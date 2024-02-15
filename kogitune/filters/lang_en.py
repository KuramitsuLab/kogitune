from typing import Any, List
import re
import os
from collections import Counter
from ..adhocargs import AdhocArguments

# 英語の頻出単語を50個以上含む正規表現パターン
# 例: 'the', 'and', 'of', 'to', 'a', 'in', 'is', 'that', 'it', 'for', ...
# 更に多くの単語を追加

pattern_english_common_words = re.compile(
    r'\b(the|and|of|to|a|in|is|that|it|for|as|with|his|he|was|on|are|by|'
    r'that|this|at|from|but|not|or|have|an|they|which|one|you|were|her|'
    r'all|their|there|been|if|more|when|will|would|who|so|no|out|up|into|'
    r'that|like|how|then|its|our|two|more|these|want|way|look|first|also|'
    r'new|because|day|use|no|man|find|here|thing|give|many|well)\b', re.IGNORECASE
)

def contains_english(text: str) -> bool:
    """
    与えられたテキストが英文を含むかどうかを判定する
    :param text: 判定するテキスト
    :return: 英文を含む場合はTrue、そうでない場合はFalse
    """
    return bool(pattern_english_common_words.search(text))

def alpha_fraction(text: str) -> float:
    """
    頻出英単語の比率を算出する
    """
    alphas = len([c for c in text if 'A' <= c <= 'z'])
    count = len(text)
    return alphas / count if count > 0 else 0.0 

class EnglishWordCounter(object):
    """
    与えられたテキストに英単語(欧文単語)が含まれるか判定する評価関数
    """
    def __init__(self, pattern=None, unification=True, 
                 alpha_fraction=False, length_fraction=False):
        """
        与えられたテキストに英単語が含まれるか判定する評価関数を作る
        :param words: 英単語のリスト(省略した場合は GPT-4による頻出英単語)
        :param unification: 単一化
        :param alpha_fraction: 英文字における比率
        :param length_fraction: 全テキストにおける比率 
        """
        aargs=AdhocArguments.to_adhoc(aargs)
        self.unique = aargs[f'unification|={unification}']
        self.alpha_fraction = aargs[f'ja_fraction|={alpha_fraction}']
        self.length_fraction = aargs[f'length_fraction|={length_fraction}']

    def __call__(self, text):
        ws = self.pattern.findall(text)
        word_count = len(set(ws)) if self.unique else len(ws)
        if self.alpha_fraction:
            alpha_count = len([c for c in text if 'A' <= c <= 'z'])
            return word_count / alpha_count if alpha_count > 0 else 0.0 
        if self.length_fraction:
            length_count =len(text)
            return word_count / length_count if length_count > 0 else 0.0
        return word_count

# 空白の前がアルファベットであればカウントしない
pattern_whitespace = re.compile(r'[^A-Za-z\,][\s　]+')

class WhitespaceCounter(object):
    """
    与えられたテキストの空白文字を数える。ただし、空白の前がアルファベットであればカウントしません。
    """
    def __init__(self, length_fraction=False, aargs=None):
        """
        与えられたテキストの空白文字を数える評価関数を作る
        :param length_fraction: 全テキストにおける比率 
        """
        aargs=AdhocArguments.to_adhoc(aargs)
        self.length_fraction = aargs[f'length_fraction|={length_fraction}']

    def __call__(self, text):
        ws = pattern_whitespace.findall(text)
        whitespace_count = len(ws)
        if self.length_fraction:
            length_count =len(text)
            return (whitespace_count / length_count) if length_count > 0 else 0.0
        return whitespace_count
