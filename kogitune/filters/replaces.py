from typing import List, Union

from .filters import TextFilter, adhoc

from .patterns import *
from .replaces_url import *
from .replaces_code import *

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

def add_replace_fn(name, func):
    global FUNCMAP
    try:
        func('test','')
    except BaseException as e:
        adhoc.warn('f置き換え関数が一致しないよ. {name} {func} {e}')
        return
    FUNCMAP[name] = func

def find_replace_fn(pattern:str):
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
    置換フィルター
    :patterns: 'url:<URL>|date:<date>'
    """

    def __init__(self, **kwargs):
        """
        置換フィルタを作る
        """
        super().__init__()
        adhoc.aargs_from(**kwargs).record(
            'patterns|!!',
            'uppercase|=True',
            field=self, dic=self.rec,
        )
        if isinstance(self.patterns, str):
            self.patterns = self.patterns.split('|')
        self.replace_fn_list = []
        for pattern in self.patterns:
            if ':' in pattern:
                # 'url:<URL>'
                pattern, _, replaced = pattern.partition(':')
            else:
                # コロンがない場合は、置き換えるプレースホルダーを作る
                replaced, _, _ = pattern.partition('_')
                if self.uppercase:
                    replaced = replaced.upper()
                replaced = f'<{replaced}>'
            replace_fn = find_replace_fn(pattern)
            if replace_fn:
                self.replace_fn_list.append((pattern, replace_fn, replaced))
            else:
                adhoc.notice(f'replace: {pattern}は見つかりません。無視します。')
    
    def __call__(self, text:str, record:dict):
        for pattern, replace_fn, replaced in self.replace_fn_list:
            text = replace_fn(text, replaced)
        return replace_repeated(text)

def replace(patterns: Union[str, List[str]], **kwargs):
    return ReplacementFilter(patterns, **kwargs)

def replace_cli(**kwargs):
    with adhoc.aargs_from(**kwargs) as aargs:
        text_filter = ReplacementFilter(**kwargs)
        text_filter.run_for_cli(**kwargs)
