import json
import pytest
from google_tts_mcp.config import load_config
from google_tts_mcp.server import get_config_schema, get_config_template, check_job_progress


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


def test_mcp_config_schema_resource():
    schema_json = get_config_schema()
    parsed = json.loads(schema_json)
    assert parsed["title"] == "Google TTS MCP Configuration Schema"
    assert "generator" in parsed["properties"]
    assert "audio" in parsed["properties"]


def test_mcp_config_template_resource():
    template_yaml = get_config_template()
    assert "generator:" in template_yaml
    assert "max_chars_per_partition: 1300" in template_yaml


@pytest.mark.asyncio
async def test_check_job_progress_no_job():
    res = await check_job_progress("non_existent_output_folder_xyz")
    parsed = json.loads(res)
    assert parsed["status"] == "NO_JOB_FOUND"
