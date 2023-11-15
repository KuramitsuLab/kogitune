from typing import Any

import re
import zlib
import math

def add_no_section(text):
    return text

LINE_PATTERN = re.compile(r'\n([^\n])')

def add_section_for_line(text):
    return LINE_PATTERN.sub(r'\n<sectioN>\1', text)

DOC_PATTERN = re.compile(r'\n\n([^\n])')

def add_section_for_doc(text):
    return DOC_PATTERN.sub(r'\n\n<sectioN>\1', text)

PYTHON_PATTERN = re.compile(r'\n(def|    def|class|async|    async|@|    @|\nif|\ntry|\n#|\n[A-Za-z0-9_]+\s=) ')

def add_section_for_python(code):
    return PYTHON_PATTERN.sub(r'\n<sectioN>\1 ', code)

MARKDOWN_PATTERN = re.compile(r'\n(#)')

def add_section_for_markdown(text):
    return MARKDOWN_PATTERN.sub(r'\n<sectioN>\1', text)

def select_section_fn(section):
    if section == 'python' or section == 'py':
        return add_section_for_python
    if section == 'markdown' or section == 'md':
        return add_section_for_markdown
    if section == 'line':
        return add_section_for_line
    if section == 'none':
        return add_no_section
    return add_section_for_doc


def compression_ratio(text:str, length_factor: float = 0.0)->float:
    encoded = text.encode("utf-8", errors='ignore')
    compressed = zlib.compress(encoded, level=9)
    encoded_length = len(encoded)
    compressed_length = len(compressed)
    ratio = compressed_length / encoded_length
    length_penalty = (
        length_factor * math.log(encoded_length) if length_factor else 0.0
    )
    score = ratio + length_penalty
    return score

def has_good_compression_ratio(text: str,
    min_score: float = 0.3, max_score: float = 0.7, length_factor: float = 0.0
) -> bool:
    """Checks if data compression (deflate) yields a desired size of data stream.

    NOTE(odashi, 2023-09-03):
    Ths judgment is based on an assumption that a "natual" sentence has an entropy
    within a certain range, and both "too simple" (low entropy) and "too complex" (high
    entropy) sentences don't reflect human's usual writing.
    This function calculates the data compression ratio (calculated by the Deflate
    algorithm) of the original stream, and compares if the resulting ratio is in-between
    the specified range.
    This criterion is somewhat sensitive against the length of the original stream (e.g.
    if the input is long, the resulting compression ratio tends to be small).
    This function also has a mechanism to consider the original length (adjusted by the
    `length_factor` parameter).

    Args:
        min_score: The lower bound of the compression ratio.
        max_score: The upper bound of the compression ratio.
        length_factor: Penalty factor of log(original_byte_length), usually set to
            something larger than 0. Using 0 falls back to a simple compression ratio.

    Returns:
        Judgment function, bound with `min` and `max`.

    Example:
        >>> judge = has_good_compression_ratio(0.1, 1.0, 0.0)
        >>> judge({"text": "LbdJA66Ufy4Pr6ffQEIo0DL60OL7kQl6y6ohAhqYKf3laCruuR"})
        False  # 1.16
        >>> judge({"text": "a" * 200})
        False  # 0.06
        >>> judge({"text": "This is a usual sentence. This sentence should pass this judgment."})
        True  # 0.92
    """
    return min_score <= compression_ratio(text, length_factor) < max_score


def split_section(text):
    for sec in text.split('<sectioN>'):
        print("==", compression_ratio(sec))
        print(sec)
