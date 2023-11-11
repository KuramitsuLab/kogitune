import re

# 英語の頻出単語を50個以上含む正規表現パターン
# 例: 'the', 'and', 'of', 'to', 'a', 'in', 'is', 'that', 'it', 'for', ...
# 更に多くの単語を追加

extended_pattern = (
    r'\b(the|and|of|to|a|in|is|that|it|for|as|with|his|he|was|on|are|by|'
    r'that|this|at|from|but|not|or|have|an|they|which|one|you|were|her|'
    r'all|their|there|been|if|more|when|will|would|who|so|no|out|up|into|'
    r'that|like|how|then|its|our|two|more|these|want|way|look|first|also|'
    r'new|because|day|use|no|man|find|here|thing|give|many|well)\b'
)

# 正規表現のコンパイル
common_english_words = re.compile(extended_pattern, re.IGNORECASE)

def is_english(text):
    """
    与えられたテキストが英文を含むかどうかを判定する関数
    :param text: 判定するテキスト
    :return: 英文を含む場合はTrue、そうでない場合はFalse
    """
    return bool(common_english_words.search(text))

def score_english(text):
    """
    与えられたテキストが英文を含むかどうかを判定する関数（拡張版）
    :param text: 判定するテキスト
    :return: 文字数あたりの頻出単語率
    """
    return len(common_english_words.findall(text)) / max(len(text),1)



