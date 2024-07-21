from typing import Any, List
import re
from .patterns import compile_words

# 英語の頻出単語を50個以上含む正規表現パターン
# 例: 'the', 'and', 'of', 'to', 'a', 'in', 'is', 'that', 'it', 'for', ...
# 更に多くの単語を追加

common_english_words_pattern = re.compile(
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
    return bool(common_english_words_pattern.search(text))


# 空白の前が空白であればウントしない
pattern_whitespace = re.compile(r'[^\s　][\s　]+')

# class WhitespaceCounter(ScoreFunction):
#     """
#     与えられたテキストの空白文字を数える。
#     ただし、空白の前が空白であればカウントしません。
#     """
#     def __init__(self, length_fraction=False, **kwargs):
#         """
#         与えられたテキストの空白文字を数える評価関数を作る
#         :param length_fraction: 全テキストにおける比率 
#         """
#         super().__init__(**kwargs)
#         self.length_fraction = length_fraction

#     def as_json(self):
#         return {
#             'score': self.name(),
#             'length_fraction': self.length_fraction,
#         }

#     def __call__(self, text):

#         ws = pattern_whitespace.findall(text)
#         whitespace_count = len(ws)
#         if self.length_fraction:
#             length_count =len(text)
#             return (whitespace_count / length_count) if length_count > 0 else 0.0
#         return whitespace_count
