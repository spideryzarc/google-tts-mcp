import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any
import yaml


@dataclass
class GeneratorConfig:
    provider: str = "google-ai-studio"
    model: str = "gemini-2.5-flash-preview-tts"


@dataclass
class RateLimitConfig:
    max_requests_per_minute: int = 15
    max_concurrent_requests: int = 2
    retry_attempts: int = 3
    backoff_factor: float = 2.0


@dataclass
class PartitioningConfig:
    max_chars_per_partition: int = 1300
    respect_existing_delimiters: bool = True


@dataclass
class VoiceSpec:
    voice_name: str = "Puck"
    prompt_prefix: str = ""


@dataclass
class AudioConfig:
    format: str = "wav"
    sample_rate: int = 48000
    sample_width_bytes: int = 2
    channels: int = 1
    output_dir: str = "output"
    inter_partition_pause_ms: int = 300
    naming_pattern: str = "{input_name}_part{part_num:02d}.wav"
    combine_full: bool = True


@dataclass
class AppConfig:
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    partitioning: PartitioningConfig = field(default_factory=PartitioningConfig)
    voices: Dict[str, Any] = field(default_factory=lambda: {
        "default_voice": "Puck",
        "speakers": {}
    })
    audio: AudioConfig = field(default_factory=AudioConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads application configuration.

    Resolution Order:
    1. If `config_path` is explicitly specified, load that file.
    2. If omitted, search for 'config.yaml' in current directory or project root.
    3. If no file is found, fallback to built-in default AppConfig dataclass.
    """
    if config_path:
        target_path = Path(config_path)
    else:
        cwd_config = Path("config.yaml")
        pkg_root_config = Path(__file__).parent.parent.parent / "config.yaml"
        if cwd_config.is_file():
            target_path = cwd_config
        elif pkg_root_config.is_file():
            target_path = pkg_root_config
        else:
            target_path = cwd_config

    if not target_path.is_file():
        # Fallback to built-in default AppConfig if no file exists
        return AppConfig()

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    gen_data = data.get("generator", {})
    rate_data = data.get("rate_limit", {})
    part_data = data.get("partitioning", {})
    voices_data = data.get("voices", {})
    audio_data = data.get("audio", {})

    generator = GeneratorConfig(
        provider=gen_data.get("provider", "google-ai-studio"),
        model=gen_data.get("model", "gemini-2.5-flash-preview-tts")
    )
    rate_limit = RateLimitConfig(
        max_requests_per_minute=rate_data.get("max_requests_per_minute", 15),
        max_concurrent_requests=rate_data.get("max_concurrent_requests", 2),
        retry_attempts=rate_data.get("retry_attempts", 3),
        backoff_factor=float(rate_data.get("backoff_factor", 2.0))
    )
    partitioning = PartitioningConfig(
        max_chars_per_partition=int(part_data.get("max_chars_per_partition", 1300)),
        respect_existing_delimiters=bool(part_data.get("respect_existing_delimiters", True))
    )
    audio = AudioConfig(
        format=audio_data.get("format", "wav"),
        sample_rate=int(audio_data.get("sample_rate", 48000)),
        sample_width_bytes=int(audio_data.get("sample_width_bytes", 2)),
        channels=int(audio_data.get("channels", 1)),
        output_dir=audio_data.get("output_dir", "output"),
        inter_partition_pause_ms=int(audio_data.get("inter_partition_pause_ms", 300)),
        naming_pattern=audio_data.get("naming_pattern", "{input_name}_part{part_num:02d}.wav"),
        combine_full=bool(audio_data.get("combine_full", True))
    )

    return AppConfig(
        generator=generator,
        rate_limit=rate_limit,
        partitioning=partitioning,
        voices=voices_data if isinstance(voices_data, dict) else {},
        audio=audio
    )
