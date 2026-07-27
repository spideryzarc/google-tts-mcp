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
    assert config.generator.model == "gemini-3.1-flash-tts-preview"
    assert config.audio.channels == 1
    assert config.audio.sample_rate == 48000
    assert config.voices.get("default_voice") == "Kore"


def test_config_user_override_merging(tmp_path):
    custom_yaml = tmp_path / "custom_config.yaml"
    custom_yaml.write_text("voices:\n  default_voice: Aoede\n", encoding="utf-8")

    config = load_config(str(custom_yaml))
    assert config.voices.get("default_voice") == "Aoede"
    # Options omitted by user naturally inherit from base config.yaml
    assert config.generator.model == "gemini-3.1-flash-tts-preview"
    assert config.audio.sample_rate == 48000



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


@pytest.mark.asyncio
async def test_transparent_api_key_validation(monkeypatch, tmp_path):
    from google_tts_mcp.server import generate_tts_from_file
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    sample_file = tmp_path / "test.tts"
    sample_file.write_text("Narrator: Teste de chave ausente.\n", encoding="utf-8")

    res_str = await generate_tts_from_file(file_path=str(sample_file), dry_run=False)
    res = json.loads(res_str)
    assert "error" in res
    assert "Chave de API não configurada" in res["error"]

