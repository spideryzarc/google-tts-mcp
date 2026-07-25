import math
import struct
import wave
import io
from pathlib import Path

from google_tts_mcp.audio import resample_24k_to_48k_mono, pcm_to_wav, combine_pcm_chunks

ARTIFACTS_DIR = Path(__file__).parent / "output_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _generate_sine_wave_pcm_24k(frequency_hz: float = 440.0, duration_sec: float = 1.0, volume: float = 0.5) -> bytes:
    """Generates 24000 Hz 16-bit signed PCM audio bytes of a pure sine tone (e.g. 440Hz A4 pitch)."""
    sample_rate = 24000
    n_samples = int(sample_rate * duration_sec)
    max_amplitude = 32767 * volume

    pcm_data = bytearray()
    for i in range(n_samples):
        t = i / sample_rate
        sample_val = int(max_amplitude * math.sin(2 * math.pi * frequency_hz * t))
        pcm_data.extend(struct.pack("<h", sample_val))

    return bytes(pcm_data)


def test_resample_and_write_inspectable_wav():
    """Generates a 440Hz sine tone at 24kHz, resamples to 48kHz Mono pcm_s16le, and writes an inspectable .wav file."""
    pcm_24k = _generate_sine_wave_pcm_24k(frequency_hz=440.0, duration_sec=1.5, volume=0.4)
    resampled_pcm_48k = resample_24k_to_48k_mono(pcm_24k)

    wav_bytes = pcm_to_wav(pcm_24k, source_rate=24000, target_rate=48000, channels=1)

    out_file = ARTIFACTS_DIR / "test_440hz_tone_48k.wav"
    out_file.write_bytes(wav_bytes)

    assert out_file.exists()
    assert out_file.stat().st_size > 0

    # Verify RIFF WAV metadata
    with wave.open(str(out_file), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 48000
        # 1.5 seconds at 48000 Hz = 72000 frames
        assert wf.getnframes() == 72000


def test_combine_pcm_chunks_write_inspectable_wav():
    """Generates two distinct sine tones (440Hz and 880Hz) with 300ms silence and writes inspectable combined .wav file."""
    pcm_tone1 = _generate_sine_wave_pcm_24k(frequency_hz=440.0, duration_sec=1.0, volume=0.4)
    pcm_tone2 = _generate_sine_wave_pcm_24k(frequency_hz=880.0, duration_sec=1.0, volume=0.4)

    combined_wav_bytes = combine_pcm_chunks(
        pcm_chunks=[pcm_tone1, pcm_tone2],
        pause_ms=300,
        source_rate=24000,
        target_rate=48000,
        channels=1
    )

    out_file = ARTIFACTS_DIR / "test_combined_tones_with_pause_48k.wav"
    out_file.write_bytes(combined_wav_bytes)

    assert out_file.exists()
    assert out_file.stat().st_size > 0

    with wave.open(str(out_file), "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getnchannels() == 1
        # 1s + 0.3s + 1s = 2.3s = 110400 frames
        expected_frames = 48000 + int(48000 * 0.30) + 48000
        assert abs(wf.getnframes() - expected_frames) <= 10
