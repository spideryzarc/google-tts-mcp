import re
from pathlib import Path


def get_input_basename(file_path: str) -> str:
    """Extracts the stem/basename of a file path without extension."""
    path = Path(file_path)
    return path.stem


def format_output_filename(pattern: str, input_name: str, part_num: int, ext: str = "wav") -> str:
    """Formats output filename according to pattern."""
    formatted = pattern.format(
        input_name=input_name,
        part_num=part_num,
        format=ext
    )
    return formatted


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename for cross-platform compatibility."""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)


def sanitize_script_text(text: str) -> str:
    """Sanitizes script text for optimal Gemini TTS audio synthesis.

    1. Removes invisible zero-width Unicode control characters (\u200b, \ufeff, etc.).
    2. Normalizes line breaks (consecutive empty lines >= 3 reduced to double newlines).
    3. Trims trailing/leading whitespace per line.
    4. Cleans empty/malformed brackets (e.g. '[]').
    """
    if not text:
        return ""

    # Remove zero-width spaces and non-printable control chars except \n and \t
    cleaned = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)

    # Clean empty brackets []
    cleaned = re.sub(r'\[\s*\]', '', cleaned)

    # Normalize CRLF to LF
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')

    # Clean leading/trailing whitespace on each line and collapse spaces
    lines = [re.sub(r'[ \t]+', ' ', line.strip()) for line in cleaned.splitlines()]
    cleaned = '\n'.join(lines)

    # Collapse 3+ consecutive newlines into 2 (paragraph break)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()
