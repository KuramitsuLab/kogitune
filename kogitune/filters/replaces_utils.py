from typing import List
import regex


def RE(*patterns: List[str], flags=0):
    return regex.compile('|'.join(patterns), flags=flags)

def replace_pattern(pattern, text, replaced):
    return pattern.sub(replaced, text)

