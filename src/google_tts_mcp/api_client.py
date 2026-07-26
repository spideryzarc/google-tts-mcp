import asyncio
import base64
import logging
import os
import random
import re
from dotenv import load_dotenv

from aiolimiter import AsyncLimiter
from google_tts_mcp.config import AppConfig

# Load environment variables from .env if present
load_dotenv()

logger = logging.getLogger("google_tts_mcp")

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


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


class GoogleTTSClient:
    def __init__(self, config: Optional[AppConfig] = None):
        if config is None:
            from google_tts_mcp.config import load_config
            config = load_config()

        self.config = config
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        rate_cfg = config.rate_limit
        self.limiter = AsyncLimiter(rate_cfg.max_requests_per_minute, 60)
        self.semaphore = asyncio.Semaphore(rate_cfg.max_concurrent_requests)

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

        raise ValueError("Nenhuma voz foi especificada e nenhuma voz padrão ('default_voice' ou locutor) foi encontrada no config.yaml!")

    def _get_genai_client(self):
        if not GENAI_AVAILABLE:
            raise RuntimeError(
                "The 'google-genai' package is not installed. Install it via 'pip install google-genai'."
            )
        if not self.api_key:
            # Refresh from environment in case it was loaded dynamically
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not self.api_key:
            err = "API key not found! Please set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable or place it in .env."
            logger.error(f"[AUTH ERROR] {err}")
            raise RuntimeError(err)
        return genai.Client(api_key=self.api_key)

    async def generate_speech_pcm(self, text_chunk: str, voice_name: str = None, system_prompt: str = None) -> bytes:
        """Generates raw PCM audio bytes for a text chunk using Gemini TTS API."""
        genai_client = self._get_genai_client()
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

        for attempt in range(1, retry_attempts + 1):
            try:
                async with self.limiter:
                    async with self.semaphore:
                        # Execute API call in executor to keep async loop non-blocking
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
                    ("quota" in err_msg and ("exceeded" in err_msg or "429" in err_msg))
                )
                if is_daily_quota:
                    logger.error(
                        f"[DAILY QUOTA EXCEEDED] Daily limit of {self.config.rate_limit.max_requests_per_day} requests/day reached on Google AI Studio API. Aborting execution immediately."
                    )
                    raise RuntimeError(
                        f"Cota diária da API gratuita do Gemini ({self.config.rate_limit.max_requests_per_day} requisições/dia) foi atingida! Execução interrompida imediatamente sem tentar novamente. Detalhes do erro: {e}"
                    )

                is_rate_limit = "429" in err_msg or "resource_exhausted" in err_msg or "503" in err_msg
                if is_rate_limit and attempt < retry_attempts:
                    # Check if Google returned a specific retry delay in the error message
                    delay_match = re.search(r'retry in (\d+(?:\.\d+)?)s', err_msg) or re.search(r"retrydelay':\s*'(\d+)s'", err_msg)
                    if delay_match:
                        sleep_time = float(delay_match.group(1)) + random.uniform(0.5, 1.5)
                    else:
                        sleep_time = max(20.0, (backoff_factor ** attempt) * 10.0 + random.uniform(0.5, 1.5))

                    logger.warning(
                        f"[RATE LIMIT 429/503] API rate limit hit. Intentional delay: backing off for {sleep_time:.1f}s before retry (Attempt {attempt}/{retry_attempts})..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(
                        f"[API ERROR] Request denied or failed on attempt {attempt}/{retry_attempts}: {e}"
                    )
                    raise RuntimeError(f"Failed to generate speech with Google AI Studio API (Attempt {attempt}/{retry_attempts}): {e}")

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
