import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

from fastmcp import FastMCP

from google_tts_mcp.config import load_config
from google_tts_mcp.partitioner import partition_text
from google_tts_mcp.api_client import GoogleTTSClient, APIRateLimitError, parse_api_keys_from_env
from google_tts_mcp.audio import pcm_to_wav, combine_pcm_chunks, wav_to_pcm
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


def _cleanup_file_logging(output_dir: Path):
    """Closes and removes file logger handlers writing to generation.log in output_dir."""
    target_log = (output_dir / "generation.log").resolve()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == target_log:
                    logger.removeHandler(handler)
                    handler.close()
            except Exception:
                pass


def _cleanup_job_artifacts(output_dir: Path, generated_files: list) -> Dict[str, Any]:
    """Cleans up generation log, progress.json, and partial .wav files upon task success."""
    cleaned_items = []

    # 1. Detach file logger handler so file locks are released
    _cleanup_file_logging(output_dir)

    # 2. Delete generation.log
    log_file = output_dir / "generation.log"
    if log_file.is_file():
        try:
            log_file.unlink()
            cleaned_items.append(log_file.name)
        except Exception as e:
            logger.warning(f"Could not remove log file '{log_file.name}': {e}")

    # 3. Delete progress.json
    progress_file = output_dir / "progress.json"
    if progress_file.is_file():
        try:
            progress_file.unlink()
            cleaned_items.append(progress_file.name)
        except Exception as e:
            logger.warning(f"Could not remove progress file '{progress_file.name}': {e}")

    # 4. Delete partial partition .wav files after combined audio is created
    for file_info in generated_files:
        filepath = file_info.get("filepath")
        if filepath:
            part_file = Path(filepath)
            if part_file.is_file():
                try:
                    part_file.unlink()
                    cleaned_items.append(part_file.name)
                except Exception as e:
                    logger.warning(f"Could not remove partition file '{part_file.name}': {e}")

    return {
        "cleaned_up": True,
        "cleaned_items": cleaned_items
    }




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
async def check_job_progress(output_dir: str = "output") -> str:
    """Checks the real-time progress status of an active or completed TTS audio generation job.

    Args:
        output_dir: Output directory where the job is generating files (defaults to 'output').
    """
    out_path = Path(output_dir)
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


async def partition_tts_file(
    file_path: str,
    config_path: Optional[str] = None
) -> str:
    """Inspects and dry-run partitions a .tts script file without making API calls.

    Args:
        file_path: Path to the .tts script file.
        config_path: Optional custom config.yaml file path.
    """
    path = Path(file_path)
    try:
        config = _validate_config_and_env(config_path, check_api_key=False)
    except Exception as err:
        return json.dumps({"error": f"Configuration validation failed: {err}"}, ensure_ascii=False)

    text = path.read_text(encoding="utf-8")

    max_chars = config.partitioning.max_chars_per_partition
    respect_delimiters = config.partitioning.respect_existing_delimiters
    chunks = partition_text(
        text=text,
        max_chars=max_chars,
        respect_existing_delimiters=respect_delimiters
    )

    report = {
        "file_path": str(path.resolve()),
        "input_name": path.stem,
        "total_characters": len(text),
        "total_partitions": len(chunks),
        "max_chars_per_partition": max_chars,
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


def _validate_config_and_env(config_path: Optional[str] = None, check_api_key: bool = True):
    """Internal helper to validate configuration structure and environment pre-requisites.

    Raises:
        FileNotFoundError: If configuration file is missing.
        ValueError: If config structure is invalid.
        EnvironmentError: If required API key environment variables are missing when live API execution is required.
    """
    config = load_config(config_path)
    if check_api_key:
        has_api_key = bool(parse_api_keys_from_env())
        if not has_api_key:
            raise EnvironmentError("API key not configured. Please set GEMINI_API_KEYS, GEMINI_API_KEY, or GOOGLE_API_KEY in the environment or .env file.")
    return config


@mcp.tool()
async def generate_tts_from_file(
    file_path: str,
    output_dir: str = "output",
    dry_run: bool = False,
    resume: bool = True,
    config_path: Optional[str] = None
) -> str:
    """Generates WAV audio files from a .tts script using Google AI Studio API (or dry-run simulation).

    Args:
        file_path: Path to the .tts script file (e.g. samples/aula04-script-duo.tts).
        output_dir: Output directory path where audio files will be generated (default 'output').
        dry_run: If True, simulates speech generation with synthetic PCM audio without invoking the Google API.
        resume: If True, skips already generated valid partition files from a previous run.
        config_path: Optional path to custom config.yaml.
    """
    path = Path(file_path)
    if not path.is_file():
        return json.dumps({"error": f"Input file not found: {file_path}"}, ensure_ascii=False)

    try:
        config = _validate_config_and_env(config_path, check_api_key=not dry_run)
    except Exception as err:
        return json.dumps({"error": f"Configuration/Environment validation failed: {err}"}, ensure_ascii=False)

    text = path.read_text(encoding="utf-8")
    script_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

    out_directory = Path(output_dir)
    _setup_file_logging(out_directory)

    chunks = partition_text(
        text=text,
        max_chars=config.partitioning.max_chars_per_partition,
        respect_existing_delimiters=config.partitioning.respect_existing_delimiters
    )

    if not chunks:
        return json.dumps({"error": "The input script file is empty."}, ensure_ascii=False)

    total_chunks = len(chunks)
    input_basename = get_input_basename(file_path)
    start_time = time.time()

    mode_label = "DRY RUN" if dry_run else "LIVE API"
    logger.info(f"Starting TTS generation ({mode_label}) for '{path.name}': {total_chunks} partitions, {len(text)} characters.")

    progress_info = {
        "status": "PROCESSING",
        "input_file": str(path.resolve()),
        "input_name": input_basename,
        "script_hash": script_hash,
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
        filename = format_output_filename(
            pattern=config.audio.naming_pattern,
            input_name=input_basename,
            part_num=chunk.part_num,
            ext=config.audio.format
        )
        file_dest = out_directory / sanitize_filename(filename)

        # Check if partition file can be resumed from disk
        if resume and file_dest.is_file() and file_dest.stat().st_size > 0:
            logger.info(f"[{idx}/{total_chunks}] Partition {chunk.part_num} already exists ('{file_dest.name}'). Resuming: skipping API call.")
            try:
                wav_bytes = file_dest.read_bytes()
                pcm_data, sample_rate = wav_to_pcm(wav_bytes)
                pcm_chunks.append((pcm_data, sample_rate))

                file_info = {
                    "part_num": chunk.part_num,
                    "filename": file_dest.name,
                    "filepath": str(file_dest.resolve()),
                    "bytes": len(wav_bytes),
                    "char_count": chunk.char_count,
                    "resumed": True
                }
                generated_files.append(file_info)

                progress_info["completed_partitions"] = idx
                progress_info["partition_files"].append(file_info)
                progress_info["elapsed_seconds"] = round(time.time() - start_time, 1)
                _write_progress_json(out_directory, progress_info)
                continue
            except Exception as read_err:
                logger.warning(f"[{idx}/{total_chunks}] Failed to read existing file '{file_dest.name}': {read_err}. Regenerating chunk...")

        # Process chunk via API or dry run
        logger.info(f"[{idx}/{total_chunks}] Generating speech for chunk {chunk.part_num} ({chunk.char_count} chars)...")

        progress_info["current_partition"] = idx
        progress_info["elapsed_seconds"] = round(time.time() - start_time, 1)
        _write_progress_json(out_directory, progress_info)

        try:
            if dry_run:
                # Generate 1 second of 24kHz 16-bit mono PCM silence for synthetic dry run testing
                pcm_bytes = b"\x00\x00" * 24000
            else:
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

            file_dest.write_bytes(wav_bytes)

            chunk_elapsed = round(time.time() - chunk_start, 2)
            logger.info(f"[{idx}/{total_chunks}] Saved '{file_dest.name}' ({len(wav_bytes)} bytes) in {chunk_elapsed}s.")

            file_info = {
                "part_num": chunk.part_num,
                "filename": file_dest.name,
                "filepath": str(file_dest.resolve()),
                "bytes": len(wav_bytes),
                "char_count": chunk.char_count,
                "resumed": False
            }
            generated_files.append(file_info)

            progress_info["completed_partitions"] = idx
            progress_info["partition_files"].append(file_info)
            progress_info["elapsed_seconds"] = round(time.time() - start_time, 1)
            _write_progress_json(out_directory, progress_info)

        except APIRateLimitError as limit_err:
            total_elapsed = round(time.time() - start_time, 1)
            logger.warning(
                f"[API LIMIT INTERRUPTED] Generation interrupted at partition {idx}/{total_chunks}: {limit_err}. Progress saved."
            )

            progress_info["status"] = "INTERRUPTED_API_LIMIT"
            progress_info["interrupted_at_partition"] = idx
            progress_info["elapsed_seconds"] = total_elapsed
            progress_info["error"] = str(limit_err)
            _write_progress_json(out_directory, progress_info)

            return json.dumps({
                "status": "INTERRUPTED",
                "reason": "API_LIMIT",
                "message": f"TTS generation interrupted at partition {idx}/{total_chunks} due to API limit ({limit_err}). Progress saved up to partition {len(generated_files)}. Call generate_tts_from_file again to resume.",
                "input_file": str(path.resolve()),
                "output_directory": str(out_directory.resolve()),
                "completed_partitions": len(generated_files),
                "total_partitions": total_chunks,
                "interrupted_at_partition": idx,
                "elapsed_seconds": total_elapsed,
                "log_file": str((out_directory / "generation.log").resolve()),
                "progress_file": str((out_directory / "progress.json").resolve()),
                "partition_files": generated_files
            }, indent=2, ensure_ascii=False)

        except Exception as gen_err:
            total_elapsed = round(time.time() - start_time, 1)
            logger.error(
                f"[UNEXPECTED ERROR] Generation failed at partition {idx}/{total_chunks}: {gen_err}."
            )

            progress_info["status"] = "INTERRUPTED_ERROR"
            progress_info["interrupted_at_partition"] = idx
            progress_info["elapsed_seconds"] = total_elapsed
            progress_info["error"] = str(gen_err)
            _write_progress_json(out_directory, progress_info)

            return json.dumps({
                "status": "INTERRUPTED",
                "reason": "ERROR",
                "message": f"TTS generation failed at partition {idx}/{total_chunks}: {gen_err}",
                "input_file": str(path.resolve()),
                "output_directory": str(out_directory.resolve()),
                "completed_partitions": len(generated_files),
                "total_partitions": total_chunks,
                "interrupted_at_partition": idx,
                "elapsed_seconds": total_elapsed,
                "log_file": str((out_directory / "generation.log").resolve()),
                "progress_file": str((out_directory / "progress.json").resolve()),
                "partition_files": generated_files
            }, indent=2, ensure_ascii=False)

    complete_file_info = None
    if len(pcm_chunks) > 0:
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

    cleanup_info = _cleanup_job_artifacts(
        output_dir=out_directory,
        generated_files=generated_files
    )

    result = {
        "status": "SUCCESS",
        "input_file": str(path.resolve()),
        "output_directory": str(out_directory.resolve()),
        "total_partitions": total_chunks,
        "elapsed_seconds": total_elapsed,
        "log_file": None,
        "progress_file": None,
        "partition_files": generated_files,
        "complete_file": complete_file_info,
        "cleaned_up": True,
        "cleanup_details": cleanup_info
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
