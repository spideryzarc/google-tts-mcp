import re
from dataclasses import dataclass
from typing import List


@dataclass
class PartitionChunk:
    part_num: int
    text: str
    char_count: int
    paragraph_count: int


def _split_into_sentences(text: str) -> List[str]:
    """Splits a single long paragraph into sentences based on period boundaries (. ! ? :)."""
    sentence_pattern = re.compile(r'(?<=[.!?:])\s+')
    sentences = sentence_pattern.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _split_long_sentence_fallback(sentence: str, max_chars: int) -> List[str]:
    """Fallback splitter for an unusually long single sentence exceeding max_chars."""
    words = sentence.split(" ")
    sub_chunks: List[str] = []
    current_words: List[str] = []
    current_len = 0

    for word in words:
        # If a single word itself exceeds max_chars, slice it
        if len(word) > max_chars:
            if current_words:
                sub_chunks.append(" ".join(current_words))
                current_words = []
                current_len = 0
            for i in range(0, len(word), max_chars):
                sub_chunks.append(word[i:i + max_chars])
            continue

        word_len = len(word)
        if current_words and (current_len + 1 + word_len > max_chars):
            sub_chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = word_len
        else:
            current_words.append(word)
            current_len += (1 if current_words else 0) + word_len

    if current_words:
        sub_chunks.append(" ".join(current_words))

    return sub_chunks


def _partition_paragraphs(paragraphs: List[str], max_chars: int) -> List[str]:
    """Partitions a list of paragraphs into chunks <= max_chars.
    If a single paragraph exceeds max_chars, it is split into sentences by period boundaries.
    """
    chunks: List[str] = []
    current_paragraphs: List[str] = []
    current_length = 0

    for para in paragraphs:
        para_str = para.strip()
        if not para_str:
            continue

        # If a single paragraph itself is larger than max_chars, we split it by sentences (. ! ? :)
        if len(para_str) > max_chars:
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_length = 0

            sentences = _split_into_sentences(para_str)
            curr_sent_chunk: List[str] = []
            curr_sent_len = 0
            for sent in sentences:
                if len(sent) > max_chars:
                    if curr_sent_chunk:
                        chunks.append(" ".join(curr_sent_chunk))
                        curr_sent_chunk = []
                        curr_sent_len = 0
                    sub_parts = _split_long_sentence_fallback(sent, max_chars)
                    chunks.extend(sub_parts)
                    continue

                sent_len = len(sent)
                if curr_sent_chunk and (curr_sent_len + 1 + sent_len > max_chars):
                    chunks.append(" ".join(curr_sent_chunk))
                    curr_sent_chunk = [sent]
                    curr_sent_len = sent_len
                else:
                    curr_sent_chunk.append(sent)
                    curr_sent_len += (1 if curr_sent_chunk else 0) + sent_len
            if curr_sent_chunk:
                chunks.append(" ".join(curr_sent_chunk))
            continue

        added_len = len(para_str) + (2 if current_paragraphs else 0)  # \n\n separator
        if current_length + added_len > max_chars and current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))
            current_paragraphs = [para_str]
            current_length = len(para_str)
        else:
            current_paragraphs.append(para_str)
            current_length += added_len

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    return chunks


def partition_text(text: str, max_chars: int = 1300, respect_existing_delimiters: bool = True) -> List[PartitionChunk]:
    """Partitions script text into chunks <= max_chars.

    Logic:
    1. If respect_existing_delimiters is True and text contains '---':
       - Split text by '---'.
       - Check if ALL sections resulting from '---' have len(section.strip()) <= max_chars.
       - If YES, preserve existing sections as partitions.
       - If NO, ignore '---' and fall back to paragraph-level partitioning.
    2. Paragraph-level partitioning groups paragraphs without breaking them unless a paragraph
       exceeds max_chars, in which case it is split by sentences (. ! ? :).
    """
    raw_text = text.strip()
    if not raw_text:
        return []

    # Step 1: Check existing '---' section delimiters
    if respect_existing_delimiters and "---" in raw_text:
        sections = [sec.strip() for sec in raw_text.split("---") if sec.strip()]
        if sections and all(len(sec) <= max_chars for sec in sections):
            return [
                PartitionChunk(
                    part_num=idx + 1,
                    text=sec,
                    char_count=len(sec),
                    paragraph_count=len([p for p in sec.split("\n\n") if p.strip()])
                )
                for idx, sec in enumerate(sections)
            ]

    # Step 2: Fallback or standard paragraph-level partitioning
    cleaned_text = re.sub(r'^\s*---\s*$', '', raw_text, flags=re.MULTILINE)
    paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]

    raw_chunks = _partition_paragraphs(paragraphs, max_chars)

    result_chunks: List[PartitionChunk] = []
    for idx, chunk in enumerate(raw_chunks):
        paras_in_chunk = [p for p in chunk.split("\n\n") if p.strip()]
        result_chunks.append(
            PartitionChunk(
                part_num=idx + 1,
                text=chunk,
                char_count=len(chunk),
                paragraph_count=len(paras_in_chunk)
            )
        )

    return result_chunks
