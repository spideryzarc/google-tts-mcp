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
