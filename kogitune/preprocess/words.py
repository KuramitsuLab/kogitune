import re

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

def score_english(text: str) -> float:
    """
    与えられたテキストが英文を含むかどうかを判定する関数（拡張版）
    :param text: 判定するテキスト
    :return: 文字数あたりの頻出単語率
    """
    return len(pattern_english_common_words.findall(text)) / max(len(text),1)


pattern_japanese = re.compile(r'[ぁ-んァ-ヶ]')

def contains_japanese(text: str) -> bool:
    """
    与えられたテキストが英文を含むかどうかを判定する
    :param text: 判定するテキスト
    :return: 英文を含む場合はTrue、そうでない場合はFalse
    """
    return bool(re.search(pattern_japanese, text))

