import json
import pytest
from pathlib import Path
from unittest.mock import patch

from google_tts_mcp.api_client import APIRateLimitError
from google_tts_mcp.server import generate_tts_from_file


@pytest.mark.asyncio
async def test_cleanup_on_success_enabled(tmp_path):
    sample_file = tmp_path / "test_cleanup.tts"
    sample_file.write_text(
        "Narrator: Partição 1 para teste de limpeza.\n"
        "[Pause 1s]\n"
        "Narrator: Partição 2 para teste de limpeza.\n",
        encoding="utf-8"
    )

    out_dir = tmp_path / "output_cleaned"

    res_str = await generate_tts_from_file(
        file_path=str(sample_file),
        output_dir=str(out_dir),
        max_chars_per_partition=40,
        combine_parts=True,
        dry_run=True,
        cleanup_on_success=True
    )

    res = json.loads(res_str)
    assert res["status"] == "SUCCESS"
    assert res["cleaned_up"] is True
    assert res["log_file"] is None
    assert res["progress_file"] is None

    # Verify that generation.log and progress.json were removed
    assert not (out_dir / "generation.log").exists()
    assert not (out_dir / "progress.json").exists()

    # Verify partial WAV files were removed
    for part in res["partition_files"]:
        assert not Path(part["filepath"]).exists()

    # Verify complete WAV file exists
    complete_path = Path(res["complete_file"]["filepath"])
    assert complete_path.exists()
    assert complete_path.stat().st_size > 0


@pytest.mark.asyncio
async def test_cleanup_on_success_disabled(tmp_path):
    sample_file = tmp_path / "test_no_cleanup.tts"
    sample_file.write_text(
        "Narrator: Partição 1 para teste de retenção de arquivos.\n",
        encoding="utf-8"
    )

    out_dir = tmp_path / "output_kept"

    res_str = await generate_tts_from_file(
        file_path=str(sample_file),
        output_dir=str(out_dir),
        max_chars_per_partition=100,
        combine_parts=True,
        dry_run=True,
        cleanup_on_success=False
    )

    res = json.loads(res_str)
    assert res["status"] == "SUCCESS"
    assert res["cleaned_up"] is False

    # Verify logs, progress.json and partition files are retained
    assert (out_dir / "generation.log").exists()
    assert (out_dir / "progress.json").exists()

    for part in res["partition_files"]:
        assert Path(part["filepath"]).exists()


@pytest.mark.asyncio
async def test_no_cleanup_on_interruption(tmp_path):
    sample_file = tmp_path / "test_interrupt_cleanup.tts"
    sample_file.write_text(
        "Narrator: Partição 1 para teste de interrupção.\n"
        "[Pause 1s]\n"
        "Narrator: Partição 2 para teste de interrupção.\n",
        encoding="utf-8"
    )

    out_dir = tmp_path / "output_interrupted"

    call_count = 0
    async def mock_generate_speech_pcm(text):
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise APIRateLimitError("Erro simulado de cota", is_daily_quota=False)
        return b"\x00\x00" * 24000

    with patch("google_tts_mcp.server.GoogleTTSClient.generate_speech_pcm", side_effect=mock_generate_speech_pcm):
        res_str = await generate_tts_from_file(
            file_path=str(sample_file),
            output_dir=str(out_dir),
            max_chars_per_partition=40,
            dry_run=False,
            cleanup_on_success=True
        )

    res = json.loads(res_str)
    assert res["status"] == "INTERRUPTED"

    # On interruption, progress.json and generation.log must NOT be cleaned up
    assert (out_dir / "progress.json").exists()
    assert (out_dir / "generation.log").exists()
