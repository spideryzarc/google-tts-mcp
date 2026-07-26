from pathlib import Path
from google_tts_mcp.partitioner import partition_text


def test_partition_with_valid_delimiters():
    text = "Section 1 content.\n\n---\n\nSection 2 content."
    chunks = partition_text(text, max_chars=1300, respect_existing_delimiters=True)
    assert len(chunks) == 2
    assert chunks[0].text == "Section 1 content."
    assert chunks[1].text == "Section 2 content."


def test_partition_with_exceeded_delimiters():
    # If one section exceeds max_chars, '---' should be ignored
    long_section = "A" * 1400 + "."
    text = f"Section 1.\n\n---\n\n{long_section}"
    chunks = partition_text(text, max_chars=1300, respect_existing_delimiters=True)
    assert len(chunks) > 1
    # Check that no chunk exceeds max_chars
    for chunk in chunks:
        assert chunk.char_count <= 1300


def test_long_paragraph_sentence_splitting():
    # Long paragraph without \n\n, but with sentences
    sentences = [f"Esta é a frase número {i} do parágrafo." for i in range(50)]
    long_para = " ".join(sentences)
    chunks = partition_text(long_para, max_chars=500, respect_existing_delimiters=True)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.char_count <= 500


def test_aula04_sample_partitioning():
    sample_file = Path(__file__).parent.parent / "samples" / "aula04-script-duo.tts"
    if sample_file.exists():
        text = sample_file.read_text(encoding="utf-8")
        chunks = partition_text(text, max_chars=1300, respect_existing_delimiters=True)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.char_count <= 1300


def test_balanced_partitioning():
    paragraphs = [
        "A" * 300,
        "B" * 250,
        "C" * 280,
        "D" * 310,
        "E" * 290
    ]
    text = "\n\n".join(paragraphs)
    # Total chars: 1438 chars, limit 1300 => 2 balanced chunks (~720 chars each)
    chunks = partition_text(text, max_chars=1300, respect_existing_delimiters=True)
    assert len(chunks) == 2
    # Verify that the difference in length between chunks is small (balanced split)
    diff = abs(chunks[0].char_count - chunks[1].char_count)
    assert diff < 300  # Significantly more balanced than greedy (which produced ~1150 vs ~300)


def test_sanitize_script_text():
    from google_tts_mcp.utils import sanitize_script_text
    raw = "Linha 1\r\n\r\n\r\nLinha 2 \u200b[]\n\n\n\nLinha 3"
    cleaned = sanitize_script_text(raw)
    assert "\r" not in cleaned
    assert "\u200b" not in cleaned
    assert "[]" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned == "Linha 1\n\nLinha 2\n\nLinha 3"
