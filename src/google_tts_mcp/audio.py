import array
import io
import wave
from typing import List, Tuple


def resample_24k_to_48k_mono(pcm_bytes: bytes) -> bytes:
    """Resamples 24000 Hz 16-bit signed mono PCM bytes to 48000 Hz 16-bit signed mono PCM bytes.

    Uses linear interpolation for clean audio upsampling from 24kHz to 48kHz (1:2 ratio).
    """
    if not pcm_bytes:
        return b""

    # Interpret as signed 16-bit little-endian integers
    samples = array.array('h')
    samples.frombytes(pcm_bytes)

    n_samples = len(samples)
    if n_samples == 0:
        return b""

    resampled = array.array('h')
    # Pre-allocate capacity for 2x samples
    for i in range(n_samples - 1):
        s1 = samples[i]
        s2 = samples[i + 1]
        mid = (s1 + s2) // 2
        resampled.append(s1)
        resampled.append(mid)

    # Last sample duplicated for boundary
    resampled.append(samples[-1])
    resampled.append(samples[-1])

    return resampled.tobytes()


def generate_silence_pcm(duration_ms: int, sample_rate: int = 48000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Generates null PCM bytes representing silence for the given duration in milliseconds."""
    if duration_ms <= 0:
        return b""
    num_samples = int((sample_rate * duration_ms) / 1000.0)
    total_bytes = num_samples * channels * sample_width
    return b"\x00" * total_bytes


def pcm_to_wav(
    pcm_bytes: bytes,
    source_rate: int = 24000,
    target_rate: int = 48000,
    channels: int = 1,
    sample_width: int = 2
) -> bytes:
    """Converts raw PCM bytes into a valid RIFF WAV file format.
    If source_rate is 24000 and target_rate is 48000, performs resampling to 48kHz.
    """
    if source_rate == 24000 and target_rate == 48000 and channels == 1 and sample_width == 2:
        processed_pcm = resample_24k_to_48k_mono(pcm_bytes)
        out_rate = 48000
    else:
        processed_pcm = pcm_bytes
        out_rate = source_rate

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(out_rate)
        wf.writeframes(processed_pcm)

    return buf.getvalue()


def combine_pcm_chunks(
    pcm_chunks: List[bytes],
    pause_ms: int = 300,
    source_rate: int = 24000,
    target_rate: int = 48000,
    channels: int = 1,
    sample_width: int = 2
) -> bytes:
    """Combines multiple PCM chunks into a single concatenated WAV file with optional silence pause between chunks."""
    silence_bytes = generate_silence_pcm(pause_ms, sample_rate=target_rate, channels=channels, sample_width=sample_width)

    combined_pcm = bytearray()
    for idx, pcm in enumerate(pcm_chunks):
        if source_rate == 24000 and target_rate == 48000 and channels == 1 and sample_width == 2:
            resampled = resample_24k_to_48k_mono(pcm)
        else:
            resampled = pcm

        combined_pcm.extend(resampled)
        if idx < len(pcm_chunks) - 1 and pause_ms > 0:
            combined_pcm.extend(silence_bytes)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(target_rate)
        wf.writeframes(bytes(combined_pcm))

    return buf.getvalue()
