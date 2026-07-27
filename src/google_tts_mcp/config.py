import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import yaml


@dataclass
class GeneratorConfig:
    provider: str
    model: str


@dataclass
class RateLimitConfig:
    max_requests_per_minute: int
    max_requests_per_day: int
    max_concurrent_requests: int
    retry_attempts: int
    backoff_factor: float


@dataclass
class PartitioningConfig:
    max_chars_per_partition: int
    respect_existing_delimiters: bool


@dataclass
class VoiceSpec:
    voice_name: str
    prompt_prefix: str


@dataclass
class AudioConfig:
    format: str
    sample_rate: int
    sample_width_bytes: int
    channels: int
    inter_partition_pause_ms: int
    naming_pattern: str


@dataclass
class AppConfig:
    generator: GeneratorConfig
    rate_limit: RateLimitConfig
    partitioning: PartitioningConfig
    voices: Dict[str, Any]
    audio: AudioConfig


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _get_required(data: dict, section: str, key: str) -> Any:
    sec = data.get(section)
    if not isinstance(sec, dict) or key not in sec:
        raise ValueError(f"Invalid configuration: field '{section}.{key}' is required in config.yaml!")
    return sec[key]


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads application configuration strictly from config.yaml without hardcoded Python fallbacks.

    Resolution Order:
    1. Load default 'config.yaml' (from current working directory or package root). If missing, raise FileNotFoundError.
    2. If `config_path` is explicitly specified, verify existence (raise FileNotFoundError if missing) and merge over base config.
    """
    cwd_config = Path("config.yaml")
    pkg_root_config = Path(__file__).parent.parent.parent / "config.yaml"

    base_path = None
    if cwd_config.is_file():
        base_path = cwd_config
    elif pkg_root_config.is_file():
        base_path = pkg_root_config

    if not base_path:
        raise FileNotFoundError("Required configuration file 'config.yaml' was not found in project or package root!")

    with open(base_path, "r", encoding="utf-8") as f:
        base_data = yaml.safe_load(f) or {}

    user_data = {}
    if config_path:
        target_path = Path(config_path)
        if not target_path.is_file():
            raise FileNotFoundError(f"Specified configuration file '{config_path}' was not found!")
        with open(target_path, "r", encoding="utf-8") as f:
            user_data = yaml.safe_load(f) or {}

    data = _deep_merge_dicts(base_data, user_data) if user_data else base_data

    generator = GeneratorConfig(
        provider=str(_get_required(data, "generator", "provider")),
        model=str(_get_required(data, "generator", "model"))
    )
    rate_limit = RateLimitConfig(
        max_requests_per_minute=int(_get_required(data, "rate_limit", "max_requests_per_minute")),
        max_requests_per_day=int(_get_required(data, "rate_limit", "max_requests_per_day")),
        max_concurrent_requests=int(_get_required(data, "rate_limit", "max_concurrent_requests")),
        retry_attempts=int(_get_required(data, "rate_limit", "retry_attempts")),
        backoff_factor=float(_get_required(data, "rate_limit", "backoff_factor"))
    )
    partitioning = PartitioningConfig(
        max_chars_per_partition=int(_get_required(data, "partitioning", "max_chars_per_partition")),
        respect_existing_delimiters=bool(_get_required(data, "partitioning", "respect_existing_delimiters"))
    )
    audio = AudioConfig(
        format=str(_get_required(data, "audio", "format")),
        sample_rate=int(_get_required(data, "audio", "sample_rate")),
        sample_width_bytes=int(_get_required(data, "audio", "sample_width_bytes")),
        channels=int(_get_required(data, "audio", "channels")),
        inter_partition_pause_ms=int(_get_required(data, "audio", "inter_partition_pause_ms")),
        naming_pattern=str(_get_required(data, "audio", "naming_pattern"))
    )

    voices_data = data.get("voices")
    if not isinstance(voices_data, dict):
        raise ValueError("Invalid configuration: 'voices' section is required in config.yaml!")

    voices_dict = voices_data.copy()
    speakers = voices_dict.get("speakers", {})
    if isinstance(speakers, dict) and speakers:
        first_speaker = next(iter(speakers.values()))
        if isinstance(first_speaker, dict) and "voice_name" in first_speaker:
            if not voices_dict.get("default_voice"):
                voices_dict["default_voice"] = first_speaker["voice_name"]

    return AppConfig(
        generator=generator,
        rate_limit=rate_limit,
        partitioning=partitioning,
        voices=voices_dict,
        audio=audio
    )
