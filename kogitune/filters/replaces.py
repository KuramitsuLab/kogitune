from typing import List
from .base import TextFilter
from .replaces_utils import *
from .replaces_code import *
from .replaces_url import *

bar_pattern = RE(
    r'(\<[\w_]+\>)(?:\s*\1)+',
)

def replace_repeated(text, replaced=None):
    """
    text中の-----などの４連続以上の文字を短くする

    >>> replreplace_repeated("<id> <id>")
    '<id>'

    >>> place_repeated("<id> <id> <id>")
    '<id>'
    """

    text = replace_pattern(bar_pattern, text, r'\1\1\1')
    return text

FUNCMAP = {
}

def add_replace_func(name, func):
    global FUNCMAP
    try:
        func('test','')
    except BaseException as e:
        adhoc.warn('f置き換え関数が一致しないよ. {name} {func} {e}')
        return
    FUNCMAP[name] = func

def find_replace_func(pattern:str):
    if pattern in FUNCMAP:
        return FUNCMAP[pattern]
    func = globals().get(f'replace_{pattern}')
    if func is None:
        patterns = [s.replace('replace_', '') for s in globals() if s.startswith('replace_')]
        raise ValueError(f'replace_{pattern} is not found. Select pattern from {patterns}')
    return func

class Replacement(TextFilter):
    """
    置き換えフィルター
    :patterns: 'url:<URL>|date:<date>'
    """

    def __init__(self, patterns: List[str], **kwargs):
        """
        置き換えフィルターを作る
        :param patterns: 置き換える文字列パターンのリスト
        """
        if isinstance(patterns,str):
            patterns = patterns.split('|')
        self.replace_funcs = []
        for pattern in patterns:
            pattern, _, w = pattern.partition(':')
            self.replace_funcs.append((pattern, find_replace_func(pattern), w))
        self.verbose_count = kwargs.get('verbose', 0) 
    
    def __call__(self, text):
        if self.verbose > 0:
            for name, replace_fn, w in self.replace_funcs:
                replaced_text = replace_fn(text, w)
                if text != replaced_text:
                    adhoc.log('replace', name, {'before': text, 'after': replaced_text})
                    self.verbose_count -= 1
                text = replaced_text
        else:
            for _, replace_fn, w in self.replace_funcs:
                text = replace_fn(text, w)
            return replace_repeated(text)

def replace(patterns: Union(str, List[str]), **kwargs):
    return Replacement(patterns)

