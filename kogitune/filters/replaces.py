from typing import List, Union

from .commons import TextFilter, adhoc

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
        adhoc.warn('知らないパターンだね', unknown_pattern=pattern, expected_patterns=patterns)
        return None
    return func

class ReplacementFilter(TextFilter):
    """
    置き換えフィルター
    :patterns: 'url:<URL>|date:<date>'
    """

    def __init__(self, patterns: List[str], uppercase=False, verbose_count=0, **kwargs):
        """
        置き換えフィルターを作る
        patterns: 置き換える文字列パターンのリスト
        
        >>> ReplacementFilter("data|url:<URL>")

        """
        if isinstance(patterns, str):
            patterns = patterns.split('|')
        self.patterns = patterns
        self.uppercase = uppercase
        self._funcs = []
        for pattern in self.patterns:
            if ':' in patterns:
                # 'url:<URL>'
                pattern, _, replaced = pattern.partition(':')
            else:
                # コロンがない場合は、置き換えるプレースホルダーを作る
                replaced, _, _ = pattern.partition('_')
                if uppercase:
                    replaced = replaced.upper()
                replaced = f'<{replaced}>'
            replace_fn = find_replace_func(pattern)
            if replace_fn:
                self._funcs.append((pattern, replace_fn, replaced))
        self._verbose_count = verbose_count 
    
    def __call__(self, text:str, record:dict):
        if self._verbose_count > 0:
            for pattern, replace_fn, replaced in self._funcs:
                replaced_text = replace_fn(text, replaced)
                if text != replaced_text:
                    adhoc.log('filter/replace', pattern, {'before': text, 'after': replaced_text})
                    self._verbose_count -= 1
                text = replaced_text
            return replace_repeated(text)
        for _, replace_fn, replaced in self._funcs:
            text = replace_fn(text, replaced)
        return replace_repeated(text)

def replace(patterns: Union[str, List[str]], uppercase=False, verbose_count=0, **kwargs):
    return ReplacementFilter(patterns, uppercase=uppercase, verbose_count=verbose_count)

