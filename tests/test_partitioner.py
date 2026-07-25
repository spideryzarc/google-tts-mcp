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
