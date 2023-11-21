from typing import List
import re
import os

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

def score_english(text: str, strict=False) -> float:
    """
    与えられたテキストが英文を含むかどうかを判定する関数（拡張版）
    :param text: 判定するテキスト
    :return: 文字数あたりの頻出単語率
    """
    words = len(pattern_english_common_words.findall(text))
    count = len([c for c in text if c.isalpha()]) if strict else len(text)
    return words / count if count > 0 else 0.0 


pattern_hirakata = re.compile(r'[ぁ-んァ-ヶ]')
pattern_japanese = re.compile(r'[ぁ-んァ-ヶー・\u4E00-\u9FFF\u3400-\u4DBF、。]')

def contains_japanese(text: str) -> bool:
    """
    与えられたテキストが日本語を含むかどうかを判定する

    :param text: 判定するテキスト
    :return: 日本語を含む場合はTrue、そうでない場合はFalse
    """
    return bool(re.search(pattern_hirakata, text))

def count_japanese_characters(text):
    """
    Count the number of Kanji, Hiragana, and Katakana characters in a text.

    Parameters:
    text (str): The input text.
    """
    return len(pattern_japanese.findall(text))

pattern_japanese_common_words = re.compile(
    r'(ある|あり|いた|いて|お|か|く|けど|けれど|こと|これ|この|'
    r'され|して|した|しな|する|すれ|せず|その|それ|そう|たい|たく|ため|'
    r'ついて|った|って|て|と|な|に|の|は|へ|ほど|まで|ます|ません|まし|'
    r'む|も|や|よ|ら|る|れな|わ|んだ|んで|を|が|だ|でき|です|でな|ば)')

def score_japanese(text: str, strict=False) -> float:
    """
    助詞/助動詞の出現頻度から日本語の品質をスコアつけ
    :param text: 判定するテキスト
    :param strict: 日本語に限定
    :return: スコア 0.25以上
    """
    count_commons = len(pattern_japanese_common_words.findall(text))
    if strict:
        count = count_japanese_characters(text)
        return count_commons / count if count > 0 else 0.0
    return  count_commons / len(text) if len(text) > 0 else 0.0

def read_words(filelist: List[str]):
    ws = []
    for file in filelist:
        if not os.path.isfile(file):
            ws.append(file)
            continue
        with open(file) as f:
            ws.extend(s.strip() for s in f.readlines() if len(s.strip()) > 0)
    return ws

def make_contain_ngwords_fn(words: List[str], max_allowed_num = 3):
    words = read_words(words)
    words_pattern = re.compile('|'.join(re.escape(w) for w in words))
    def has_NGwords(text):
        return len(words_pattern.findall(text)) > max_allowed_num
    return has_NGwords

def make_score_fn(words: List[str]):
    words_pattern = re.compile(r'\s(' + '|'.join(words) + r')\s')
    def score_fn(text):
        return len(words_pattern.findall(text)) / max(len(text),1)
    return score_fn

import zlib, math

def compression_ratio(text:str, length_factor: float = 0.0)->float:
    encoded = text.encode("utf-8", errors='ignore')
    compressed = zlib.compress(encoded, level=9)
    encoded_length = len(encoded)
    compressed_length = len(compressed)
    ratio = compressed_length / encoded_length
    length_penalty = (
        length_factor * math.log(encoded_length) if length_factor else 0.0
    )
    score = ratio + length_penalty
    return round(score,3)
