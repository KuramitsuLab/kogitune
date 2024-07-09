from typing import List

import os
import re
import regex

def compile_words(words: List[str], prefix='', suffix=''):
    """
    Given a list of words or a single string of words separated by '|', compiles and returns a regular expression pattern that matches any of the words. Additionally, if the words list contains filenames ending in '.txt', the function reads these files and includes their contents as words. The function removes duplicates and sorts the words before compiling the pattern.

    If `prefix` or `suffix` strings are provided, they are added to the beginning and end of the compiled pattern, respectively.

    Parameters:
    - words (List[str] or str): A list of words, or a single string of words separated by '|'. Can also include filenames with '.txt' extension, whose contents will be read and included as words.
    - prefix (str, optional): A string to be added to the beginning of the compiled pattern. Defaults to an empty string.
    - suffix (str, optional): A string to be added to the end of the compiled pattern. Defaults to an empty string.

    Returns:
    - re.Pattern: A compiled regular expression pattern that matches any of the specified words, optionally enclosed between `prefix` and `suffix`.

    Note:
    - The function ensures that duplicates are removed and the final list of words is sorted before compiling the pattern.
    - If a filename is provided in the `words` list and it does not exist or cannot be read, it is ignored.
    """
    if isinstance(words, str):
        words = words.split('|')

    ws = []
    for w in words:
        if w.endswith('.txt') and os.path.isfile(w):
            with open(w) as f:
                ws.extend(s.strip() for s in f.readlines() if len(s.strip()) > 0)
        else:
            ws.append(w)
    ws = list(set(ws))
    ws.sort()
    pattern = '|'.join(re.escape(w) for w in ws)
    if len(prefix) > 0 or len(suffix) > 0:
        return regex.compile(f'{prefix}({pattern}){suffix}')
    return regex.compile(pattern)


def RE(*patterns: List[str], flags=0):
    return regex.compile('|'.join(patterns), flags=flags)

def replace_pattern(pattern, text, replaced):
    return pattern.sub(replaced, text)

