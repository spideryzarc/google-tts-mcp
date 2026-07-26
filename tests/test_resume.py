import json
import pytest
from pathlib import Path
from unittest.mock import patch

from google_tts_mcp.audio import pcm_to_wav, wav_to_pcm, combine_pcm_chunks
from google_tts_mcp.api_client import APIRateLimitError
from google_tts_mcp.server import generate_tts_from_file


def test_wav_to_pcm_conversion():
    # 1 second of silence at 24kHz mono PCM (48000 bytes)
    original_pcm = b"\x00\x00" * 24000
    wav_bytes = pcm_to_wav(original_pcm, source_rate=24000, target_rate=48000)
    
    extracted_pcm, rate = wav_to_pcm(wav_bytes)
    assert rate == 48000
    assert len(extracted_pcm) == 48000 * 2  # 16-bit mono = 2 bytes per sample = 96000 bytes


def test_combine_pcm_chunks_mixed():
    pcm1 = b"\x00\x00" * 100
    pcm2_tuple = (b"\x00\x00" * 200, 48000)
    
    combined_wav = combine_pcm_chunks([pcm1, pcm2_tuple], pause_ms=100)
    extracted_pcm, rate = wav_to_pcm(combined_wav)
    assert rate == 48000
    assert len(extracted_pcm) > 0


@pytest.mark.asyncio
async def test_generate_tts_from_file_resume(tmp_path):
    sample_file = tmp_path / "test_script.tts"
    sample_file.write_text(
        "Narrator: Esta é a primeira partição para testes.\n"
        "[Pause 2s]\n"
        "Narrator: Esta é a segunda partição para testes.\n",
        encoding="utf-8"
    )
    
    out_dir = tmp_path / "output"
    
    # First run (dry_run)
    res1_str = await generate_tts_from_file(
        file_path=str(sample_file),
        output_dir=str(out_dir),
        max_chars_per_partition=40,
        dry_run=True,
        resume=True
    )
    res1 = json.loads(res1_str)
    assert res1["status"] == "SUCCESS"
    assert len(res1["partition_files"]) > 0
    assert all(f["resumed"] is False for f in res1["partition_files"])

    # Second run with resume=True: should skip existing files
    res2_str = await generate_tts_from_file(
        file_path=str(sample_file),
        output_dir=str(out_dir),
        max_chars_per_partition=40,
        dry_run=True,
        resume=True
    )
    res2 = json.loads(res2_str)
    assert res2["status"] == "SUCCESS"
    assert all(f["resumed"] is True for f in res2["partition_files"])


@pytest.mark.asyncio
async def test_generate_tts_from_file_rate_limit_interruption(tmp_path):
    sample_file = tmp_path / "test_script.tts"
    sample_file.write_text(
        "Narrator: Partição 1 do teste de rate limit.\n"
        "[Pause 2s]\n"
        "Narrator: Partição 2 do teste de rate limit.\n",
        encoding="utf-8"
    )
    
    out_dir = tmp_path / "output"
    
    # Mock GoogleTTSClient.generate_speech_pcm to raise APIRateLimitError on chunk 2
    call_count = 0
    async def mock_generate_speech_pcm(text):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise APIRateLimitError("Limite de cota simulado no teste", is_daily_quota=False)
        return b"\x00\x00" * 24000

    with patch("google_tts_mcp.server.GoogleTTSClient.generate_speech_pcm", side_effect=mock_generate_speech_pcm):
        res_str = await generate_tts_from_file(
            file_path=str(sample_file),
            output_dir=str(out_dir),
            max_chars_per_partition=40,
            dry_run=False,
            resume=True
        )
        res = json.loads(res_str)
        assert res["status"] == "INTERRUPTED"
        assert res["reason"] == "API_LIMIT"
        assert res["completed_partitions"] == 1
        
        # Verify progress.json was saved with INTERRUPTED_API_LIMIT status
        progress_file = out_dir / "progress.json"
        assert progress_file.is_file()
        progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
        assert progress_data["status"] == "INTERRUPTED_API_LIMIT"
        assert progress_data["completed_partitions"] == 1
