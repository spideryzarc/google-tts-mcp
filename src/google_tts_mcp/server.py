import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

from mcp.server.fastmcp import FastMCP

from google_tts_mcp.config import load_config
from google_tts_mcp.partitioner import partition_text
from google_tts_mcp.api_client import GoogleTTSClient
from google_tts_mcp.audio import pcm_to_wav, combine_pcm_chunks
from google_tts_mcp.utils import get_input_basename, format_output_filename, sanitize_filename

# Configure logging
logger = logging.getLogger("google_tts_mcp")
logger.setLevel(logging.INFO)

mcp = FastMCP("Google TTS MCP")


def _setup_file_logging(output_dir: Path):
    """Sets up a file logger in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "generation.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    
    # Avoid adding duplicate handlers
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file.resolve()) for h in logger.handlers):
        logger.addHandler(file_handler)


def _write_progress_json(output_dir: Path, progress_data: Dict[str, Any]):
    """Writes real-time progress data to progress.json in the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / "progress.json"
    progress_file.write_text(json.dumps(progress_data, indent=2, ensure_ascii=False), encoding="utf-8")


@mcp.resource("config://schema")
def get_config_schema() -> str:
    """Returns the JSON schema and field descriptions for building a custom config.yaml file."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Google TTS MCP Configuration Schema",
        "description": "Schema definition and field documentation for google-tts-mcp config.yaml files.",
        "type": "object",
        "properties": {
            "generator": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "default": "google-ai-studio",
                        "description": "API provider name."
                    },
                    "model": {
                        "type": "string",
                        "default": "gemini-2.5-flash-preview-tts",
                        "description": "Google AI Studio TTS capable model identifier."
                    }
                }
            },
            "rate_limit": {
                "type": "object",
                "properties": {
                    "max_requests_per_minute": {
                        "type": "integer",
                        "default": 15,
                        "description": "Maximum allowed API requests per minute (RPM)."
                    },
                    "max_concurrent_requests": {
                        "type": "integer",
                        "default": 2,
                        "description": "Maximum parallel API connections."
                    },
                    "retry_attempts": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of retry attempts on rate limit (429/503) errors."
                    },
                    "backoff_factor": {
                        "type": "number",
                        "default": 2.0,
                        "description": "Exponential backoff multiplier in seconds."
                    }
                }
            },
            "partitioning": {
                "type": "object",
                "properties": {
                    "max_chars_per_partition": {
                        "type": "integer",
                        "default": 1300,
                        "description": "Maximum character length per chunk partition."
                    },
                    "respect_existing_delimiters": {
                        "type": "boolean",
                        "default": True,
                        "description": "If True and text contains '---', preserves existing sections if all <= max_chars."
                    }
                }
            },
            "voices": {
                "type": "object",
                "properties": {
                    "default_voice": {
                        "type": "string",
                        "default": "Puck",
                        "description": "Fallback prebuilt Google TTS voice name (e.g. Puck, Aoede, Charon, Kore, Fenrir)."
                    },
                    "speakers": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "voice_name": {"type": "string"},
                                "prompt_prefix": {"type": "string"}
                            }
                        },
                        "description": "Speaker label mapping to voice names and prompt instructions."
                    }
                }
            },
            "audio": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "default": "wav",
                        "enum": ["wav", "pcm"],
                        "description": "Output audio format."
                    },
                    "sample_rate": {
                        "type": "integer",
                        "default": 48000,
                        "description": "Target sample rate in Hz (48000 recommended for video editor sync)."
                    },
                    "sample_width_bytes": {
                        "type": "integer",
                        "default": 2,
                        "description": "Bytes per sample (2 = 16-bit pcm_s16le)."
                    },
                    "channels": {
                        "type": "integer",
                        "default": 1,
                        "description": "Number of audio channels (1 = Mono)."
                    },
                    "output_dir": {
                        "type": "string",
                        "default": "output",
                        "description": "Directory path where output WAV files will be saved."
                    },
                    "inter_partition_pause_ms": {
                        "type": "integer",
                        "default": 300,
                        "description": "Silence pause in milliseconds inserted between concatenated partitions."
                    },
                    "naming_pattern": {
                        "type": "string",
                        "default": "{input_name}_part{part_num:02d}.wav",
                        "description": "Filename pattern for individual partition audio files."
                    },
                    "combine_full": {
                        "type": "boolean",
                        "default": True,
                        "description": "If True, generates the concatenated {input_name}_complete.wav file."
                    }
                }
            }
        }
    }
    return json.dumps(schema, indent=2, ensure_ascii=False)


@mcp.resource("config://template")
def get_config_template() -> str:
    """Returns the content of config.yaml in YAML format. Raises FileNotFoundError if unavailable."""
    cwd_config = Path("config.yaml")
    pkg_root_config = Path(__file__).parent.parent.parent / "config.yaml"

    if cwd_config.is_file():
        return cwd_config.read_text(encoding="utf-8").strip()
    elif pkg_root_config.is_file():
        return pkg_root_config.read_text(encoding="utf-8").strip()

    raise FileNotFoundError("Configuration template file 'config.yaml' was not found.")


@mcp.tool()
async def check_job_progress(output_dir: Optional[str] = "output") -> str:
    """Checks the real-time progress status of an active or completed TTS audio generation job.

    Args:
        output_dir: Output directory where the job is generating files (defaults to 'output').
    """
    out_path = Path(output_dir or "output")
    progress_file = out_path / "progress.json"
    log_file = out_path / "generation.log"

    if not progress_file.is_file():
        return json.dumps({
            "status": "NO_JOB_FOUND",
            "message": f"No active or recent progress.json found in {out_path.resolve()}"
        }, ensure_ascii=False)

    try:
        progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
        if log_file.is_file():
            recent_logs = log_file.read_text(encoding="utf-8").strip().splitlines()[-10:]
            progress_data["recent_log_tail"] = recent_logs
        return json.dumps(progress_data, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to read progress log: {e}"}, ensure_ascii=False)


@mcp.tool()
async def partition_tts_file(
    file_path: str,
    max_chars_per_partition: int = 1300,
    config_path: Optional[str] = None
) -> str:
    """Inspects and dry-run partitions a .tts script file without making API calls.

    Args:
        file_path: Path to the .tts script file.
        max_chars_per_partition: Maximum allowed characters per chunk (default 1300).
        config_path: Optional custom config.yaml file path.
    """
    path = Path(file_path)
    if not path.is_file():
        return json.dumps({"error": f"File not found: {file_path}"}, ensure_ascii=False)

    config = load_config(config_path)
    text = path.read_text(encoding="utf-8")

    respect_delimiters = config.partitioning.respect_existing_delimiters
    chunks = partition_text(
        text=text,
        max_chars=max_chars_per_partition,
        respect_existing_delimiters=respect_delimiters
    )

    report = {
        "file_path": str(path.resolve()),
        "input_name": path.stem,
        "total_characters": len(text),
        "total_partitions": len(chunks),
        "max_chars_per_partition": max_chars_per_partition,
        "partitions": [
            {
                "part_num": chunk.part_num,
                "char_count": chunk.char_count,
                "paragraph_count": chunk.paragraph_count,
                "preview": chunk.text[:120] + ("..." if len(chunk.text) > 120 else "")
            }
            for chunk in chunks
        ]
    }

    return json.dumps(report, indent=2, ensure_ascii=False)


@mcp.tool()
async def validate_config(config_path: Optional[str] = None) -> str:
    """Validates the config.yaml structure and reports environment status.

    Args:
        config_path: Path to config.yaml (defaults to ./config.yaml if omitted).
    """
    try:
        config = load_config(config_path)
        has_api_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

        status = {
            "status": "VALID",
            "config_path": config_path or "config.yaml (default)",
            "environment": {
                "api_key_detected": has_api_key,
                "api_key_variable": "GEMINI_API_KEY / GOOGLE_API_KEY"
            },
            "generator": {
                "provider": config.generator.provider,
                "model": config.generator.model
            },
            "rate_limit": {
                "max_requests_per_minute": config.rate_limit.max_requests_per_minute,
                "max_concurrent_requests": config.rate_limit.max_concurrent_requests
            },
            "partitioning": {
                "max_chars_per_partition": config.partitioning.max_chars_per_partition,
                "respect_existing_delimiters": config.partitioning.respect_existing_delimiters
            },
            "audio": {
                "format": config.audio.format,
                "sample_rate": config.audio.sample_rate,
                "channels": config.audio.channels,
                "output_dir": config.audio.output_dir
            }
        }
        return json.dumps(status, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"status": "INVALID", "error": str(e)}, ensure_ascii=False)


@mcp.tool()
async def generate_tts_from_file(
    file_path: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_chars_per_partition: int = 1300,
    combine_parts: bool = True
) -> str:
    """Generates 48kHz WAV audio files from a .tts script using Google AI Studio API.

    Args:
        file_path: Path to the .tts script file (e.g. samples/aula04-script-duo.tts).
        config_path: Path to config.yaml.
        output_dir: Output directory path (defaults to config.yaml setting or ./output).
        max_chars_per_partition: Character limit per partition (default 1300).
        combine_parts: If True, also generates the full merged {input_name}_complete.wav file.
    """
    path = Path(file_path)
    if not path.is_file():
        return json.dumps({"error": f"Input file not found: {file_path}"}, ensure_ascii=False)

    config = load_config(config_path)
    text = path.read_text(encoding="utf-8")

    out_directory = Path(output_dir or config.audio.output_dir)
    _setup_file_logging(out_directory)

    chunks = partition_text(
        text=text,
        max_chars=max_chars_per_partition,
        respect_existing_delimiters=config.partitioning.respect_existing_delimiters
    )

    if not chunks:
        return json.dumps({"error": "The input script file is empty."}, ensure_ascii=False)

    total_chunks = len(chunks)
    input_basename = get_input_basename(file_path)
    start_time = time.time()

    logger.info(f"Starting TTS generation for '{path.name}': {total_chunks} partitions, {len(text)} characters.")

    progress_info = {
        "status": "PROCESSING",
        "input_file": str(path.resolve()),
        "input_name": input_basename,
        "total_partitions": total_chunks,
        "completed_partitions": 0,
        "current_partition": 1,
        "start_time": start_time,
        "elapsed_seconds": 0,
        "log_file": str((out_directory / "generation.log").resolve()),
        "partition_files": []
    }
    _write_progress_json(out_directory, progress_info)

    client = GoogleTTSClient(config)
    generated_files = []
    pcm_chunks = []

    for idx, chunk in enumerate(chunks, start=1):
        chunk_start = time.time()
        logger.info(f"[{idx}/{total_chunks}] Generating speech for chunk {chunk.part_num} ({chunk.char_count} chars)...")

        progress_info["current_partition"] = idx
        progress_info["elapsed_seconds"] = round(time.time() - start_time, 1)
        _write_progress_json(out_directory, progress_info)

        # Generate PCM bytes from Google AI Studio API
        pcm_bytes = await client.generate_speech_pcm(chunk.text)
        pcm_chunks.append(pcm_bytes)

        # Convert to 48kHz pcm_s16le WAV
        wav_bytes = pcm_to_wav(
            pcm_bytes=pcm_bytes,
            source_rate=24000,
            target_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            sample_width=config.audio.sample_width_bytes
        )

        filename = format_output_filename(
            pattern=config.audio.naming_pattern,
            input_name=input_basename,
            part_num=chunk.part_num,
            ext=config.audio.format
        )
        file_dest = out_directory / sanitize_filename(filename)
        file_dest.write_bytes(wav_bytes)

        chunk_elapsed = round(time.time() - chunk_start, 2)
        logger.info(f"[{idx}/{total_chunks}] Saved '{file_dest.name}' ({len(wav_bytes)} bytes) in {chunk_elapsed}s.")

        file_info = {
            "part_num": chunk.part_num,
            "filename": file_dest.name,
            "filepath": str(file_dest.resolve()),
            "bytes": len(wav_bytes),
            "char_count": chunk.char_count
        }
        generated_files.append(file_info)

        progress_info["completed_partitions"] = idx
        progress_info["partition_files"].append(file_info)
        progress_info["elapsed_seconds"] = round(time.time() - start_time, 1)
        _write_progress_json(out_directory, progress_info)

    complete_file_info = None
    if combine_parts and len(pcm_chunks) > 0:
        logger.info(f"Combining {len(pcm_chunks)} audio partitions into complete WAV file...")
        combined_wav_bytes = combine_pcm_chunks(
            pcm_chunks=pcm_chunks,
            pause_ms=config.audio.inter_partition_pause_ms,
            source_rate=24000,
            target_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            sample_width=config.audio.sample_width_bytes
        )
        complete_filename = f"{input_basename}_complete.wav"
        complete_dest = out_directory / sanitize_filename(complete_filename)
        complete_dest.write_bytes(combined_wav_bytes)

        logger.info(f"Saved complete combined audio: '{complete_dest.name}' ({len(combined_wav_bytes)} bytes).")

        complete_file_info = {
            "filename": complete_dest.name,
            "filepath": str(complete_dest.resolve()),
            "bytes": len(combined_wav_bytes)
        }

    total_elapsed = round(time.time() - start_time, 1)
    logger.info(f"SUCCESS: Completed TTS generation for '{path.name}' in {total_elapsed}s.")

    progress_info["status"] = "COMPLETED"
    progress_info["elapsed_seconds"] = total_elapsed
    progress_info["complete_file"] = complete_file_info
    _write_progress_json(out_directory, progress_info)

    result = {
        "status": "SUCCESS",
        "input_file": str(path.resolve()),
        "output_directory": str(out_directory.resolve()),
        "total_partitions": total_chunks,
        "elapsed_seconds": total_elapsed,
        "log_file": str((out_directory / "generation.log").resolve()),
        "progress_file": str((out_directory / "progress.json").resolve()),
        "partition_files": generated_files,
        "complete_file": complete_file_info
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
