from typing import Any, List
import re
from collections import Counter
from .commons import ScoreFunction, compile_pattern_for_words

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
    英文字の比率を算出する
    """
    alphas = len([c for c in text if 'A' <= c <= 'z'])
    count = len(text)
    return alphas / count if count > 0 else 0.0 

class AlphaFraction(ScoreFunction):
    """
    英文字の比率を算出する
    この関数は役に立ちます。
    """

    def __init__(self, **kwargs):
        """
        英文字の比率を算出する
        """
        super().__init__(**kwargs)

    def as_json(self):
        return {'score': self.name()}

    def __call__(self, text: str):
        return alpha_fraction(text)

class EnglishWordCounter(ScoreFunction):
    """
    与えられたテキストに英単語(欧文単語)が含まれるか判定する評価関数
    """
    def __init__(self, 
                 words=None, unique=False, 
                 alpha_fraction=False, 
                 length_fraction=True, **kwargs):
        """
        与えられたテキストに英単語が含まれるか判定する評価関数を作る
        :param words: 英単語のリスト(省略した場合は GPT-4による頻出英単語)
        :param unique: 単一化
        :param alpha_fraction: 英文字における比率
        :param length_fraction: 全テキストにおける比率 
        """
        super().__init__(**kwargs)
        if words:
            self.pattern = compile_pattern_for_words(words, prefix=r'\b', suffix=r'\b')
        else:
            self.pattern = pattern_english_common_words
        self.unique = unique
        self.alpha_fraction = alpha_fraction
        self.length_fraction = length_fraction

    def as_json(self):
        return {
            'score': self.name(),
            'unique': self.unique,
            'alpha_fraction': self.alpha_fraction,
            'length_fraction': self.length_fraction,
        }

    def __call__(self, text):
        ws = self.pattern.findall(text)
        word_count = len(set(ws)) if self.unique_word else len(ws)
        if self.length_fraction:
            length_count =len(text)
            return word_count / length_count if length_count > 0 else 0.0
        elif self.alpha_fraction:
            alpha_count = len([c for c in text if 'A' <= c <= 'z'])
            return word_count / alpha_count if alpha_count > 0 else 0.0 
        return word_count

# 空白の前が空白であればウントしない
pattern_whitespace = re.compile(r'[^\s　][\s　]+')

class WhitespaceCounter(ScoreFunction):
    """
    与えられたテキストの空白文字を数える。
    ただし、空白の前が空白であればカウントしません。
    """
    def __init__(self, length_fraction=False, **kwargs):
        """
        与えられたテキストの空白文字を数える評価関数を作る
        :param length_fraction: 全テキストにおける比率 
        """
        super().__init__(**kwargs)
        self.length_fraction = length_fraction

    def as_json(self):
        return {
            'score': self.name(),
            'length_fraction': self.length_fraction,
        }

    def __call__(self, text):

        ws = pattern_whitespace.findall(text)
        whitespace_count = len(ws)
        if self.length_fraction:
            length_count =len(text)
            return (whitespace_count / length_count) if length_count > 0 else 0.0
        return whitespace_count
