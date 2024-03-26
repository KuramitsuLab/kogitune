from typing import List
import re
import regex
import inspect

from ..adhoc_args import adhoc
VERBOSE = True


def RE(*patterns: List[str], flags=0):
    return regex.compile('|'.join(patterns), flags=flags)

def replace_pattern(pattern, text, replaced):
    if VERBOSE:
        ## 注意: inspect.stack()は比較的重い処理を行うため、
        ## パフォーマンスが重要な場面での頻繁な使用は避けるべきです。
        replaced_text = pattern.sub(replaced, text)
        if text != replaced_text:
            adhoc.log('replace', replaced, {'before': text, 'after': replaced_text})
        return replaced_text
    return pattern.sub(replaced, text)

