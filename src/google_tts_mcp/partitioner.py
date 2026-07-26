import math
import re
from dataclasses import dataclass
from typing import List

from google_tts_mcp.utils import sanitize_script_text


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
    """Partitions a list of paragraphs into balanced chunks <= max_chars.
    Calculates target_size per partition to ensure chunks have approximately the same size.
    If a single paragraph exceeds max_chars, it is split into sentences by period boundaries.
    """
    if not paragraphs:
        return []

    # Decompose any paragraph > max_chars into sentence units
    atomic_units: List[str] = []
    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
        if len(p_str) > max_chars:
            sents = _split_into_sentences(p_str)
            for s in sents:
                if len(s) > max_chars:
                    atomic_units.extend(_split_long_sentence_fallback(s, max_chars))
                else:
                    atomic_units.append(s)
        else:
            atomic_units.append(p_str)

    total_chars = sum(len(u) for u in atomic_units) + (len(atomic_units) - 1) * 2
    if total_chars <= max_chars:
        return ["\n\n".join(atomic_units)]

    num_chunks = math.ceil(total_chars / max_chars)
    target_size = math.ceil(total_chars / num_chunks)

    chunks: List[str] = []
    curr_units: List[str] = []
    curr_len = 0

    for idx, unit in enumerate(atomic_units):
        added_len = len(unit) + (2 if curr_units else 0)

        if curr_units:
            # Check if adding unit exceeds max_chars
            if curr_len + added_len > max_chars:
                chunks.append("\n\n".join(curr_units))
                curr_units = [unit]
                curr_len = len(unit)
                rem_units = atomic_units[idx:]
                rem_len = sum(len(u) for u in rem_units) + (len(rem_units) - 1) * 2
                rem_chunks = max(1, num_chunks - len(chunks))
                target_size = math.ceil(rem_len / rem_chunks)
                continue

            dist_before = abs(curr_len - target_size)
            dist_after = abs((curr_len + added_len) - target_size)

            # If current chunk has reached or passed target size, and adding unit makes it further:
            if curr_len >= target_size and dist_after >= dist_before:
                chunks.append("\n\n".join(curr_units))
                curr_units = [unit]
                curr_len = len(unit)
                rem_units = atomic_units[idx:]
                rem_len = sum(len(u) for u in rem_units) + (len(rem_units) - 1) * 2
                rem_chunks = max(1, num_chunks - len(chunks))
                target_size = math.ceil(rem_len / rem_chunks)
                continue

        curr_units.append(unit)
        curr_len += added_len

    if curr_units:
        chunks.append("\n\n".join(curr_units))

    return chunks


def partition_text(text: str, max_chars: int = 1300, respect_existing_delimiters: bool = True) -> List[PartitionChunk]:
    """Partitions script text into chunks <= max_chars.

    Logic:
    1. Sanitize input text (remove zero-width chars, normalize CRLF and empty lines).
    2. If respect_existing_delimiters is True and text contains '---':
       - Split text by '---'.
       - Check if ALL sections resulting from '---' have len(section.strip()) <= max_chars.
       - If YES, preserve existing sections as partitions.
       - If NO, ignore '---' and fall back to paragraph-level partitioning.
    3. Paragraph-level partitioning groups paragraphs without breaking them unless a paragraph
       exceeds max_chars, in which case it is split by sentences (. ! ? :).
    """
    raw_text = sanitize_script_text(text)
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
