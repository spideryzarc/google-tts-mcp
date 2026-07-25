import json
import os
import wave
from pathlib import Path
import pytest

from google_tts_mcp.partitioner import partition_text
from google_tts_mcp.server import partition_tts_file, generate_tts_from_file

SAMPLES_DIR = Path(__file__).parent.parent / "samples"
OUTPUT_DIR = Path(__file__).parent / "output_artifacts" / "samples_test_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_all_sample_files():
    """Returns all .tts files in samples/."""
    return sorted(list(SAMPLES_DIR.glob("*.tts")))


@pytest.mark.parametrize("sample_path", get_all_sample_files(), ids=lambda p: p.name)
def test_partition_all_samples(sample_path: Path):
    """Verifies that every script in samples/ is correctly partitioned under 1300 characters."""
    assert sample_path.exists(), f"Sample file does not exist: {sample_path}"
    text = sample_path.read_text(encoding="utf-8")
    
    chunks = partition_text(text, max_chars=1300, respect_existing_delimiters=True)
    assert len(chunks) > 0, f"No partitions generated for sample: {sample_path.name}"

    for chunk in chunks:
        assert chunk.char_count <= 1300, (
            f"Chunk {chunk.part_num} in {sample_path.name} exceeded 1300 chars ({chunk.char_count} chars)."
        )


@pytest.mark.asyncio
async def test_server_dry_run_on_samples():
    """Tests the partition_tts_file tool on all samples in samples/."""
    sample_files = get_all_sample_files()
    assert len(sample_files) > 0, "No sample files found in samples/"

    for sample_path in sample_files:
        result_json = await partition_tts_file(str(sample_path), max_chars_per_partition=1300)
        data = json.loads(result_json)

        assert "error" not in data, f"Partition error on {sample_path.name}: {data.get('error')}"
        assert data["total_partitions"] > 0
        assert data["max_chars_per_partition"] == 1300
        for part in data["partitions"]:
            assert part["char_count"] <= 1300


@pytest.mark.parametrize("sample_path", get_all_sample_files(), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_dry_run_tts_generation_all_samples(sample_path: Path):
    """Dry-run integration test: generates 48kHz WAV audio for every sample in samples/ without calling Google AI Studio API."""
    assert sample_path.exists()

    sample_out_dir = OUTPUT_DIR / f"dry_{sample_path.stem}"
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    result_json = await generate_tts_from_file(
        file_path=str(sample_path),
        output_dir=str(sample_out_dir),
        max_chars_per_partition=1300,
        combine_parts=True,
        dry_run=True
    )

    data = json.loads(result_json)
    assert data["status"] == "SUCCESS", f"Dry-run generation failed for {sample_path.name}: {data}"
    assert data["total_partitions"] > 0

    # Verify generated partition WAV files
    for part in data["partition_files"]:
        wav_path = Path(part["filepath"])
        assert wav_path.exists(), f"WAV file not created: {wav_path}"
        assert wav_path.stat().st_size > 0, f"WAV file is empty: {wav_path}"

        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == 48000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    # Verify complete concatenated WAV file
    if data.get("complete_file"):
        complete_path = Path(data["complete_file"]["filepath"])
        assert complete_path.exists()
        assert complete_path.stat().st_size > 0
        with wave.open(str(complete_path), "rb") as wf:
            assert wf.getframerate() == 48000
            assert wf.getnchannels() == 1


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_API_TESTS") != "1" or not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="Set RUN_LIVE_API_TESTS=1 and ensure API key is available to run live API tests."
)
@pytest.mark.parametrize("sample_path", get_all_sample_files(), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_live_api_tts_generation_all_samples(sample_path: Path):
    """Live API integration test: generates 48kHz WAV audio for every sample in samples/ using Google AI Studio API."""
    assert sample_path.exists()

    sample_out_dir = OUTPUT_DIR / sample_path.stem
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    result_json = await generate_tts_from_file(
        file_path=str(sample_path),
        output_dir=str(sample_out_dir),
        max_chars_per_partition=1300,
        combine_parts=True
    )

    data = json.loads(result_json)
    assert data["status"] == "SUCCESS", f"Generation failed for {sample_path.name}: {data}"
    assert data["total_partitions"] > 0

    # Verify generated partition WAV files
    for part in data["partition_files"]:
        wav_path = Path(part["filepath"])
        assert wav_path.exists(), f"WAV file not created: {wav_path}"
        assert wav_path.stat().st_size > 0, f"WAV file is empty: {wav_path}"

        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getframerate() == 48000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2

    # Verify complete concatenated WAV file
    if data.get("complete_file"):
        complete_path = Path(data["complete_file"]["filepath"])
        assert complete_path.exists()
        assert complete_path.stat().st_size > 0
        with wave.open(str(complete_path), "rb") as wf:
            assert wf.getframerate() == 48000
            assert wf.getnchannels() == 1
