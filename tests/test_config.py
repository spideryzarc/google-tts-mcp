import json
import pytest
from google_tts_mcp.config import load_config
from google_tts_mcp.server import get_config_template, check_job_progress


def test_load_default_config():
    config = load_config("non_existent_config.yaml")
    assert config.generator.provider == "google-ai-studio"
    assert config.audio.sample_rate == 48000
    assert config.partitioning.max_chars_per_partition == 1300


def test_load_existing_config():
    config = load_config("config.yaml")
    assert config.generator.model == "gemini-2.5-flash-preview-tts"
    assert config.audio.channels == 1
    assert config.audio.sample_rate == 48000
    assert config.voices.get("default_voice") == "Kore"



def test_mcp_config_template_resource():
    template_yaml = get_config_template()
    assert "generator:" in template_yaml
    assert "max_chars_per_partition: 1300" in template_yaml


@pytest.mark.asyncio
async def test_check_job_progress_no_job():
    res = await check_job_progress("non_existent_output_folder_xyz")
    parsed = json.loads(res)
    assert parsed["status"] == "NO_JOB_FOUND"


def test_multi_speaker_config():
    config = load_config("config.yaml")
    from google_tts_mcp.api_client import GoogleTTSClient
    client = GoogleTTSClient(config)
    speakers = client.config.voices.get("speakers", {})
    assert "Speaker 1" in speakers
    assert "Speaker 2" in speakers
    assert speakers["Speaker 1"]["voice_name"] == "Kore"
    assert speakers["Speaker 2"]["voice_name"] == "Aoede"
    assert "scene" in client.config.voices
    assert "context" in client.config.voices
    assert client.config.rate_limit.max_requests_per_day == 10


@pytest.mark.asyncio
async def test_daily_quota_fast_fail(monkeypatch):
    config = load_config("config.yaml")
    from google_tts_mcp.api_client import GoogleTTSClient
    client = GoogleTTSClient(config)
    monkeypatch.setattr(client, "_get_genai_client", lambda: None)

    # Simulate executor raising daily quota exceeded RuntimeError
    async def mock_generate_content(*args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 10")

    monkeypatch.setattr(client, "generate_speech_pcm", mock_generate_content)

    with pytest.raises(RuntimeError) as exc_info:
        await client.generate_speech_pcm("test text")
    assert "Quota exceeded" in str(exc_info.value)
