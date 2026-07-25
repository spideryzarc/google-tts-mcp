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


class GoogleTTSClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
        rate_cfg = config.rate_limit
        self.limiter = AsyncLimiter(rate_cfg.max_requests_per_minute, 60)
        self.semaphore = asyncio.Semaphore(rate_cfg.max_concurrent_requests)

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

        if isinstance(speakers_cfg, dict) and len(speakers_cfg) > 1 and not voice_name:
            # Multi-speaker setup matching official Google AI Studio specification
            speaker_voice_configs = []
            for spk_name, spk_info in speakers_cfg.items():
                if isinstance(spk_info, dict) and "voice_name" in spk_info:
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
            speech_config = types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=speaker_voice_configs
                )
            )
        else:
            # Single-speaker setup
            selected_voice = voice_name or self.config.voices.get("default_voice", "Aoede")
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=selected_voice
                    )
                )
            )
        
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

            # Audio Profiles
            profiles = []
            for spk_name, spk_info in speakers_cfg.items():
                if isinstance(spk_info, dict) and spk_info.get("profile"):
                    profiles.append(f"For {spk_name}: {spk_info['profile']}")
            if profiles:
                prompt_blocks.append("# Audio Profile\n" + "\n".join(profiles))

            # Director's notes
            notes = []
            for spk_name, spk_info in speakers_cfg.items():
                if isinstance(spk_info, dict) and spk_info.get("directors_note"):
                    notes.append(f"For {spk_name}: {spk_info['directors_note']}")
            if notes:
                prompt_blocks.append("# Director's note\n" + "\n".join(notes))

            # Scene
            if scene:
                prompt_blocks.append(f"## Scene:\n{scene}")

            # Sample Context
            if context:
                prompt_blocks.append(f"## Sample Context:\n{context}")

            if len(prompt_blocks) > 1:
                effective_system_prompt = "\n\n".join(prompt_blocks)
            else:
                # Fallback to legacy prompt_prefix if structured fields were omitted
                prefixes = []
                for spk_name, spk_info in speakers_cfg.items():
                    if isinstance(spk_info, dict) and spk_info.get("prompt_prefix"):
                        prefixes.append(f"For {spk_name}:\n{spk_info['prompt_prefix']}")
                if prefixes:
                    effective_system_prompt = "\n\n".join(prefixes)

        if effective_system_prompt:
            gen_config.system_instruction = effective_system_prompt

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
                                contents=text_chunk,
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
