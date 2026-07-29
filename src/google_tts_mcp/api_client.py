import asyncio
import base64
import logging
import os
import random
import re
from dotenv import load_dotenv

from aiolimiter import AsyncLimiter
from google_tts_mcp.config import AppConfig

import time
from typing import Dict, Optional, Any, List

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger("google_tts_mcp")

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def parse_api_keys_from_env() -> list[str]:
    """Parses Gemini/Google API keys from environment variables.

    Checks GEMINI_API_KEYS, GEMINI_API_KEY, and GOOGLE_API_KEY.
    Supports comma-separated lists of keys. Returns a deduplicated list of non-empty keys.
    """
    raw_keys = []
    for var_name in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.environ.get(var_name)
        if val:
            raw_keys.extend(val.split(","))

    cleaned = []
    for k in raw_keys:
        k_clean = k.strip().strip('"').strip("'")
        if k_clean and k_clean not in cleaned:
            cleaned.append(k_clean)

    return cleaned


def detect_script_speakers(text: str, speakers_cfg: dict = None) -> list[str]:
    """Detects distinct speaker names present in the script text.

    Checks for explicit speaker prefixes like 'Speaker 1:', 'Narrator:', or keys from speakers_cfg.
    Returns a list of unique detected speaker names in order of appearance.
    """
    if not text:
        return []

    known_speakers = list(speakers_cfg.keys()) if isinstance(speakers_cfg, dict) else []
    detected = []

    for line in text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue

        found_known = False
        for spk_name in known_speakers:
            pattern = rf'^\s*(?:\[\s*{re.escape(spk_name)}\s*\]|{re.escape(spk_name)})\s*:'
            if re.search(pattern, line_str, re.IGNORECASE):
                if spk_name not in detected:
                    detected.append(spk_name)
                found_known = True
                break

        if not found_known:
            match = re.match(r'^\s*([A-Za-z0-9_ -]{1,30})\s*:', line_str)
            if match:
                name = match.group(1).strip()
                if not name.startswith("[") and name not in detected:
                    detected.append(name)

    return detected


class APIRateLimitError(RuntimeError):
    """Raised when Google TTS API rate limit or quota is exceeded and retries are exhausted."""
    def __init__(self, message: str, is_daily_quota: bool = False):
        super().__init__(message)
        self.is_daily_quota = is_daily_quota


class KeyTracker:
    """Tracks rate limiting and cooldown status for an individual API key."""
    def __init__(self, key: str, max_requests_per_minute: int):
        self.key = key
        self.limiter = AsyncLimiter(max_requests_per_minute, 60)
        self.cooldown_until: float = 0.0
        self.is_daily_exhausted: bool = False

    def is_available(self, now: Optional[float] = None) -> bool:
        if self.is_daily_exhausted:
            return False
        if now is None:
            now = time.monotonic()
        return now >= self.cooldown_until


class KeyPool:
    """Manages a pool of API keys for round-robin load balancing and failover."""
    def __init__(self, api_keys: list[str], max_requests_per_minute: int):
        self.max_rpm = max_requests_per_minute
        self.trackers = [KeyTracker(k, max_requests_per_minute) for k in api_keys]
        self._index = 0

    def get_next_available(self) -> Optional[KeyTracker]:
        now = time.monotonic()
        n = len(self.trackers)
        if n == 0:
            return None

        for _ in range(n):
            idx = self._index
            self._index = (self._index + 1) % n
            tracker = self.trackers[idx]
            if tracker.is_available(now):
                return tracker
        return None

    def get_min_cooldown_wait(self) -> float:
        now = time.monotonic()
        waits = [t.cooldown_until - now for t in self.trackers if not t.is_daily_exhausted and t.cooldown_until > now]
        return min(waits) if waits else 0.0

    def all_daily_exhausted(self) -> bool:
        return bool(self.trackers) and all(t.is_daily_exhausted for t in self.trackers)

    def active_key_count(self) -> int:
        now = time.monotonic()
        return sum(1 for t in self.trackers if t.is_available(now))


class GoogleTTSClient:
    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            from google_tts_mcp.config import load_config
            config = load_config()

        self.config = config
        self.api_keys = parse_api_keys_from_env()

        rate_cfg = config.rate_limit
        num_keys = max(1, len(self.api_keys))
        self.key_pool = KeyPool(self.api_keys, rate_cfg.max_requests_per_minute)
        self.limiter = AsyncLimiter(rate_cfg.max_requests_per_minute * num_keys, 60)
        self.semaphore = asyncio.Semaphore(rate_cfg.max_concurrent_requests * num_keys)

    @property
    def api_key(self) -> Optional[str]:
        """Provides backward-compatible access to the primary API key."""
        keys = parse_api_keys_from_env()
        return keys[0] if keys else (self.api_keys[0] if self.api_keys else None)

    def _resolve_default_voice(self, speakers_cfg: dict = None) -> str:
        """Resolves default voice dynamically from config.yaml or raises ValueError if not specified."""
        if isinstance(self.config.voices, dict):
            def_v = self.config.voices.get("default_voice")
            if def_v:
                return def_v

        if isinstance(speakers_cfg, dict) and speakers_cfg:
            for spk_info in speakers_cfg.values():
                if isinstance(spk_info, dict) and spk_info.get("voice_name"):
                    return spk_info["voice_name"]

        raise ValueError("No voice specified and no default voice ('default_voice' or speaker profile) found in config.yaml!")

    def _get_genai_client(self, api_key: Optional[str] = None):
        if not GENAI_AVAILABLE:
            raise RuntimeError(
                "The 'google-genai' package is not installed. Install it via 'pip install google-genai'."
            )
        target_key = api_key or self.api_key
        if not target_key:
            err = "API key not found! Please set GEMINI_API_KEYS, GEMINI_API_KEY, or GOOGLE_API_KEY environment variable or place it in .env."
            logger.error(f"[AUTH ERROR] {err}")
            raise RuntimeError(err)
        return genai.Client(api_key=target_key)

    async def generate_speech_pcm(self, text_chunk: str, voice_name: str = None, system_prompt: str = None) -> bytes:
        """Generates raw PCM audio bytes for a text chunk using Gemini TTS API."""
        model_name = self.config.generator.model
        speakers_cfg = self.config.voices.get("speakers", {})
        default_voice = self._resolve_default_voice(speakers_cfg)

        detected_speakers = detect_script_speakers(text_chunk, speakers_cfg)
        is_multi_speaker = (
            isinstance(speakers_cfg, dict)
            and len(speakers_cfg) > 1
            and len(detected_speakers) >= 2
            and not voice_name
        )

        target_speaker = None
        if is_multi_speaker:
            # Multi-speaker setup matching official Google AI Studio specification
            speaker_voice_configs = []
            for spk_name in detected_speakers:
                spk_info = speakers_cfg.get(spk_name, {})
                v_name = spk_info.get("voice_name") if isinstance(spk_info, dict) else None
                if not v_name:
                    v_name = default_voice
                speaker_voice_configs.append(
                    types.SpeakerVoiceConfig(
                        speaker=spk_name,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=v_name
                            )
                        )
                    )
                )
            for spk_name, spk_info in speakers_cfg.items():
                if spk_name not in detected_speakers and isinstance(spk_info, dict) and "voice_name" in spk_info:
                    speaker_voice_configs.append(
                        types.SpeakerVoiceConfig(
                            speaker=spk_name,
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=spk_info["voice_name"]
                                )
                            )
                        )
                    )
            speech_config_kwargs = {
                "multi_speaker_voice_config": types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_voice_configs
                )
            }
        else:
            # Single-speaker setup
            selected_voice = voice_name
            if not selected_voice:
                if len(detected_speakers) == 1:
                    target_speaker = detected_speakers[0]
                    spk_info = speakers_cfg.get(target_speaker, {})
                    if isinstance(spk_info, dict) and "voice_name" in spk_info:
                        selected_voice = spk_info["voice_name"]

                if not selected_voice:
                    selected_voice = default_voice

            speech_config_kwargs = {
                "voice_config": types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=selected_voice
                    )
                )
            }

        language_code = self.config.voices.get("language_code") if isinstance(self.config.voices, dict) else None
        if language_code:
            speech_config_kwargs["language_code"] = language_code

        speech_config = types.SpeechConfig(**speech_config_kwargs)

        gen_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config
        )

        effective_system_prompt = system_prompt
        if not effective_system_prompt and isinstance(self.config.voices, dict):
            v_dict = self.config.voices
            scene = v_dict.get("scene")
            context = v_dict.get("context") or v_dict.get("sample_context")

            prompt_blocks = []
            prompt_blocks.append("Read the following transcript based on the audio profile and director's note.")

            # Audio Profiles & Director Notes
            if is_multi_speaker:
                profiles = []
                for spk_name, spk_info in speakers_cfg.items():
                    if isinstance(spk_info, dict) and spk_info.get("profile"):
                        profiles.append(f"For {spk_name}: {spk_info['profile']}")
                if profiles:
                    prompt_blocks.append("# Audio Profile\n" + "\n".join(profiles))

                notes = []
                for spk_name, spk_info in speakers_cfg.items():
                    if isinstance(spk_info, dict) and spk_info.get("directors_note"):
                        notes.append(f"For {spk_name}: {spk_info['directors_note']}")
                if notes:
                    prompt_blocks.append("# Director's note\n" + "\n".join(notes))
            else:
                # Single speaker active profiles
                active_name = target_speaker or (next(iter(speakers_cfg.keys())) if isinstance(speakers_cfg, dict) and speakers_cfg else None)
                if active_name and isinstance(speakers_cfg, dict) and active_name in speakers_cfg:
                    spk_info = speakers_cfg[active_name]
                    if isinstance(spk_info, dict):
                        if spk_info.get("profile"):
                            prompt_blocks.append(f"# Audio Profile\nFor {active_name}: {spk_info['profile']}")
                        if spk_info.get("directors_note"):
                            prompt_blocks.append(f"# Director's note\nFor {active_name}: {spk_info['directors_note']}")

            # Scene & Context
            if scene and is_multi_speaker:
                prompt_blocks.append(f"## Scene:\n{scene}")

            if context:
                prompt_blocks.append(f"## Sample Context:\n{context}")

            if len(prompt_blocks) > 1:
                effective_system_prompt = "\n\n".join(prompt_blocks)

        full_contents = f"{effective_system_prompt}\n\n## Transcript:\n{text_chunk}" if effective_system_prompt else text_chunk

        retry_attempts = self.config.rate_limit.retry_attempts
        backoff_factor = self.config.rate_limit.backoff_factor

        # Dynamic refresh of env keys if updated at runtime
        env_keys = parse_api_keys_from_env()
        if env_keys and set(env_keys) != set(t.key for t in self.key_pool.trackers):
            self.api_keys = env_keys
            self.key_pool = KeyPool(self.api_keys, self.config.rate_limit.max_requests_per_minute)

        for attempt in range(1, retry_attempts + 1):
            if self.key_pool.all_daily_exhausted():
                raise APIRateLimitError(
                    f"Cota diária de todas as chaves ({len(self.key_pool.trackers)}) da API foi atingida!",
                    is_daily_quota=True
                )

            key_tracker = self.key_pool.get_next_available()
            if not key_tracker and self.key_pool.trackers:
                wait_seconds = self.key_pool.get_min_cooldown_wait()
                if wait_seconds > 0:
                    logger.warning(f"[KEY POOL COOLDOWN] All API keys in pool are in cooldown. Waiting {wait_seconds:.1f}s...")
                    await asyncio.sleep(wait_seconds + 0.1)
                    key_tracker = self.key_pool.get_next_available()

            if key_tracker:
                genai_client = self._get_genai_client(api_key=key_tracker.key)
            else:
                genai_client = self._get_genai_client()

            try:
                if key_tracker:
                    async with key_tracker.limiter:
                        async with self.semaphore:
                            loop = asyncio.get_running_loop()
                            response = await loop.run_in_executor(
                                None,
                                lambda: genai_client.models.generate_content(
                                    model=model_name,
                                    contents=full_contents,
                                    config=gen_config
                                )
                            )
                else:
                    async with self.limiter:
                        async with self.semaphore:
                            loop = asyncio.get_running_loop()
                            response = await loop.run_in_executor(
                                None,
                                lambda: genai_client.models.generate_content(
                                    model=model_name,
                                    contents=full_contents,
                                    config=gen_config
                                )
                            )

                # Extract audio PCM bytes from response
                pcm_data = self._extract_pcm_bytes(response)
                if pcm_data:
                    return pcm_data

                logger.warning(f"[API WARNING] Attempt {attempt}/{retry_attempts}: API responded but no audio payload was returned.")
                raise RuntimeError("API responded but no audio payload was returned.")

            except Exception as e:
                err_msg = str(e).lower()
                is_daily_quota = (
                    "generaterequestsperday" in err_msg or
                    "free_tier_requests" in err_msg or
                    "daily_quota" in err_msg or
                    "quota_exceeded_per_day" in err_msg
                )
                if is_daily_quota and key_tracker:
                    key_tracker.is_daily_exhausted = True
                    masked_key = key_tracker.key[:6] + "..." if len(key_tracker.key) > 6 else "key"
                    logger.warning(f"[KEY EXHAUSTED] Key {masked_key} reached daily quota. Marking key as exhausted.")
                    if not self.key_pool.all_daily_exhausted():
                        logger.info("Switching to next available API key in pool...")
                        continue
                    else:
                        raise APIRateLimitError(
                            f"Cota diária de todas as chaves da API foi atingida! Detalhes: {e}",
                            is_daily_quota=True
                        ) from e
                elif is_daily_quota:
                    logger.error(
                        f"[DAILY QUOTA EXCEEDED] Daily limit of {self.config.rate_limit.max_requests_per_day} requests/day reached on Google AI Studio API. Aborting execution immediately."
                    )
                    raise APIRateLimitError(
                        f"Cota diária da API gratuita do Gemini ({self.config.rate_limit.max_requests_per_day} requisições/dia) foi atingida! Detalhes: {e}",
                        is_daily_quota=True
                    ) from e

                is_rate_limit = "429" in err_msg or "resource_exhausted" in err_msg or "503" in err_msg
                if is_rate_limit:
                    delay_match = re.search(r'retry in (\d+(?:\.\d+)?)s', err_msg) or re.search(r"retrydelay':\s*'(\d+)s'", err_msg)
                    if delay_match:
                        sleep_time = float(delay_match.group(1)) + random.uniform(0.5, 1.5)
                    else:
                        sleep_time = max(20.0, (backoff_factor ** attempt) * 10.0 + random.uniform(0.5, 1.5))

                    if key_tracker:
                        key_tracker.cooldown_until = time.monotonic() + sleep_time
                        masked_key = key_tracker.key[:6] + "..." if len(key_tracker.key) > 6 else "key"
                        logger.warning(f"[KEY RATE LIMIT 429/503] Key {masked_key} hit rate limit. Placed on cooldown for {sleep_time:.1f}s.")

                        next_key = self.key_pool.get_next_available()
                        if next_key:
                            logger.info("Failing over to next active API key in pool immediately...")
                            continue

                    if attempt < retry_attempts:
                        logger.warning(
                            f"[RATE LIMIT 429/503] API rate limit hit. Intentional delay: backing off for {sleep_time:.1f}s before retry (Attempt {attempt}/{retry_attempts})..."
                        )
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.error(
                            f"[RATE LIMIT EXHAUSTED] Rate limit hit and retries exhausted (Attempt {attempt}/{retry_attempts}): {e}"
                        )
                        raise APIRateLimitError(
                            f"Limite de requisições da API atingido após {retry_attempts} tentativas. Detalhes: {e}",
                            is_daily_quota=False
                        ) from e
                else:
                    logger.error(
                        f"[API ERROR] Request denied or failed on attempt {attempt}/{retry_attempts}: {e}"
                    )
                    raise RuntimeError(f"Failed to generate speech with Google AI Studio API (Attempt {attempt}/{retry_attempts}): {e}") from e

        raise RuntimeError("Attempts exhausted while calling Google AI Studio API.")


    def _extract_pcm_bytes(self, response) -> bytes:
        """Extracts PCM bytes from response parts."""
        if not hasattr(response, "candidates") or not response.candidates:
            return b""

        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            if not content or not hasattr(content, "parts"):
                continue

            for part in content.parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data:
                    data = getattr(inline_data, "data", None)
                    if isinstance(data, bytes):
                        return data
                    elif isinstance(data, str):
                        return base64.b64decode(data)

        return b""
