import pytest
from pathlib import Path
from google_tts_mcp.api_client import detect_script_speakers, GoogleTTSClient
from google_tts_mcp.config import load_config, AppConfig

SAMPLES_DIR = Path(__file__).parent.parent / "samples"


def test_detect_script_speakers_monologue():
    speakers_cfg = {
        "Speaker 1": {"voice_name": "Kore"},
        "Speaker 2": {"voice_name": "Aoede"}
    }
    ternario_text = (SAMPLES_DIR / "ternario.tts").read_text(encoding="utf-8")
    detected = detect_script_speakers(ternario_text, speakers_cfg)
    assert len(detected) == 0, f"Expected 0 speakers in monologue, got {detected}"


def test_detect_script_speakers_duo():
    speakers_cfg = {
        "Speaker 1": {"voice_name": "Kore"},
        "Speaker 2": {"voice_name": "Aoede"}
    }
    long_duo_text = (SAMPLES_DIR / "long-duo.tts").read_text(encoding="utf-8")
    detected = detect_script_speakers(long_duo_text, speakers_cfg)
    assert "Speaker 1" in detected
    assert "Speaker 2" in detected
    assert len(detected) == 2


@pytest.mark.asyncio
async def test_dry_run_single_speaker_monologue():
    config = load_config()
    client = GoogleTTSClient(config)
    
    ternario_text = (SAMPLES_DIR / "ternario.tts").read_text(encoding="utf-8")
    detected = detect_script_speakers(ternario_text, config.voices.get("speakers", {}))
    assert len(detected) < 2, "Monologue should not trigger multi-speaker detection"


@pytest.mark.asyncio
async def test_dry_run_multi_speaker_duo():
    config = load_config()
    client = GoogleTTSClient(config)
    
    duo_text = (SAMPLES_DIR / "long-duo.tts").read_text(encoding="utf-8")
    detected = detect_script_speakers(duo_text, config.voices.get("speakers", {}))
    assert len(detected) >= 2, "Duo script should trigger multi-speaker detection"
