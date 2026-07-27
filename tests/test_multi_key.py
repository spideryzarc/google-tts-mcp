import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch

from google_tts_mcp.api_client import (
    parse_api_keys_from_env,
    KeyTracker,
    KeyPool,
    GoogleTTSClient,
    APIRateLimitError
)
from google_tts_mcp.config import load_config


def test_parse_api_keys_from_env(monkeypatch):
    # Test single key
    monkeypatch.setenv("GEMINI_API_KEY", "key_single")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert parse_api_keys_from_env() == ["key_single"]

    # Test comma-separated GEMINI_API_KEYS
    monkeypatch.setenv("GEMINI_API_KEYS", " key1 , key2 , key3 ")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert parse_api_keys_from_env() == ["key1", "key2", "key3"]

    # Test comma-separated GEMINI_API_KEY with quotes and duplicates
    monkeypatch.setenv("GEMINI_API_KEY", '"keyA", keyB, keyA, keyC ')
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    assert parse_api_keys_from_env() == ["keyA", "keyB", "keyC"]


def test_key_tracker():
    tracker = KeyTracker("test_key", max_requests_per_minute=15)
    now = time.monotonic()
    assert tracker.is_available(now) is True

    # Put on cooldown
    tracker.cooldown_until = now + 10.0
    assert tracker.is_available(now) is False
    assert tracker.is_available(now + 11.0) is True

    # Mark daily exhausted
    tracker.is_daily_exhausted = True
    assert tracker.is_available(now + 20.0) is False


def test_key_pool_round_robin():
    keys = ["key1", "key2", "key3"]
    pool = KeyPool(keys, max_requests_per_minute=15)

    # First rotation
    k1 = pool.get_next_available()
    k2 = pool.get_next_available()
    k3 = pool.get_next_available()
    assert k1.key == "key1"
    assert k2.key == "key2"
    assert k3.key == "key3"

    # Cycles back to key1
    k4 = pool.get_next_available()
    assert k4.key == "key1"


def test_key_pool_cooldown_skipping():
    keys = ["key1", "key2", "key3"]
    pool = KeyPool(keys, max_requests_per_minute=15)

    now = time.monotonic()
    # Put key1 and key2 on cooldown
    pool.trackers[0].cooldown_until = now + 30.0
    pool.trackers[1].cooldown_until = now + 15.0

    # get_next_available should pick key3 (the only non-cooldown key)
    k = pool.get_next_available()
    assert k.key == "key3"

    # min cooldown wait should be min(30, 15) = 15s (approx)
    min_wait = pool.get_min_cooldown_wait()
    assert 14.0 <= min_wait <= 16.0


@pytest.mark.asyncio
async def test_multi_key_failover(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "key_fail, key_success")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = load_config()
    client = GoogleTTSClient(config)

    assert len(client.api_keys) == 2


    # Mock response for successful call
    mock_pcm_part = MagicMock()
    mock_pcm_part.inline_data.data = b"AUDIO_DATA"
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_pcm_part]
    mock_success_response = MagicMock()
    mock_success_response.candidates = [mock_candidate]

    def mock_genai_client_constructor(api_key):
        instance = MagicMock()
        instance.api_key = api_key
        def mock_generate_content(model, contents, config):
            if api_key == "key_fail":
                raise RuntimeError("429 Resource Exhausted: retry in 20s")
            return mock_success_response
        instance.models.generate_content.side_effect = mock_generate_content
        return instance

    with patch("google_tts_mcp.api_client.GENAI_AVAILABLE", True), \
         patch("google_tts_mcp.api_client.genai.Client", side_effect=mock_genai_client_constructor):
        audio_bytes = await client.generate_speech_pcm("Test text")
        assert audio_bytes == b"AUDIO_DATA"

        # Verify key_fail was placed on cooldown
        fail_tracker = next(t for t in client.key_pool.trackers if t.key == "key_fail")
        assert fail_tracker.cooldown_until > time.monotonic()

