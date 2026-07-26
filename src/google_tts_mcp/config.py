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
    max_requests_per_day: int = 10
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
    generator: GeneratorConfig = field(default_factory=lambda: GeneratorConfig(provider="google-ai-studio", model="gemini-2.5-flash-preview-tts"))
    rate_limit: RateLimitConfig = field(default_factory=lambda: RateLimitConfig(max_requests_per_minute=15, max_requests_per_day=10, max_concurrent_requests=2, retry_attempts=3, backoff_factor=2.0))
    partitioning: PartitioningConfig = field(default_factory=lambda: PartitioningConfig(max_chars_per_partition=1300, respect_existing_delimiters=True))
    voices: Dict[str, Any] = field(default_factory=dict)
    audio: AudioConfig = field(default_factory=lambda: AudioConfig(format="wav", sample_rate=48000, sample_width_bytes=2, channels=1, output_dir="output", inter_partition_pause_ms=300, naming_pattern="{input_name}_part{part_num:02d}.wav", combine_full=True))


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads application configuration.

    Resolution Order:
    1. Load default 'config.yaml' (from current working directory or package root).
    2. If `config_path` is explicitly specified and exists, load user configuration and merge over default.
    3. If neither default nor user config is found, raise FileNotFoundError.
    """
    cwd_config = Path("config.yaml")
    pkg_root_config = Path(__file__).parent.parent.parent / "config.yaml"

    base_path = None
    if cwd_config.is_file():
        base_path = cwd_config
    elif pkg_root_config.is_file():
        base_path = pkg_root_config

    base_data = {}
    if base_path and base_path.is_file():
        with open(base_path, "r", encoding="utf-8") as f:
            base_data = yaml.safe_load(f) or {}

    user_data = {}
    if config_path:
        target_path = Path(config_path)
        if target_path.is_file():
            with open(target_path, "r", encoding="utf-8") as f:
                user_data = yaml.safe_load(f) or {}

    if not base_data and not user_data:
        raise FileNotFoundError("O arquivo de configuração 'config.yaml' não foi encontrado no projeto nem no pacote!")

    data = _deep_merge_dicts(base_data, user_data) if user_data else base_data

    gen_data = data.get("generator", {})
    rate_data = data.get("rate_limit", {})
    part_data = data.get("partitioning", {})
    voices_data = data.get("voices", {})
    audio_data = data.get("audio", {})

    model = gen_data.get("model")
    if not model:
        raise ValueError("Configuração inválida: o modelo ('generator.model') deve ser informado no config.yaml!")

    generator = GeneratorConfig(
        provider=gen_data.get("provider", "google-ai-studio"),
        model=model
    )
    rate_limit = RateLimitConfig(
        max_requests_per_minute=int(rate_data.get("max_requests_per_minute", 15)),
        max_requests_per_day=int(rate_data.get("max_requests_per_day", 10)),
        max_concurrent_requests=int(rate_data.get("max_concurrent_requests", 2)),
        retry_attempts=int(rate_data.get("retry_attempts", 3)),
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

    voices_dict = voices_data if isinstance(voices_data, dict) else {}
    if voices_dict:
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
