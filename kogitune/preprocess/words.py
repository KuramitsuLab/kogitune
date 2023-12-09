from typing import Any, List
import re
import os
from collections import Counter


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

def count_common_english_words(text: str) -> int:
    """
    与えられたテキストに頻出英単語が含まれるか判定する関数
    :param text: 判定するテキスト
    :return: 頻出単語の数
    """
    return len(pattern_english_common_words.findall(text))

def english_fraction(text: str) -> float:
    """
    頻出英単語の比率を算出する
    """
    words = len(pattern_english_common_words.findall(text))
    count = len(text)
    return words / count if count > 0 else 0.0 

def alpha_fraction(text: str) -> float:
    """
    頻出英単語の比率を算出する
    """
    alphas = len([c for c in text if 'A' <= c <= 'z'])
    count = len(text)
    return alphas / count if count > 0 else 0.0 

def score_en(text: str) -> float:
    """
    与えられたテキストの英語の品質を算出する
    :param text: 判定するテキスト
    :return: 文字数あたりの頻出単語率
    """
    words = len(pattern_english_common_words.findall(text))
    count = len([c for c in text if 'A' <= c <= 'z'])
    return words / count if count > 0 else 0.0 

def score_english(text: str, strict=False) -> float:
    """
    与えられたテキストの英語の品質を算出する
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
    テキストに日本語を含むかどうかを判定する

    :param text: 判定するテキスト
    :return: 日本語を含む場合はTrue、そうでない場合はFalse
    """
    return bool(re.search(pattern_hirakata, text))

def count_japanese_characters(text):
    """
    漢字/カタカナ/ひらがなの数を数える

    :param text: 判定するテキスト
    :return: 漢字/カタカナ/ひらがなの数
    """
    return len(pattern_japanese.findall(text))

def japanese_fraction(text: str) -> float:
    """
    テキスト中の漢字/カタカナ/ひらがなの比率を算出する
    """
    count_commons = count_japanese_characters(text)
    return  count_commons / len(text) if len(text) > 0 else 0.0

pattern_japanese_common_words = re.compile(
    r'(ある|あり|いた|いて|お|か|く|けど|けれど|こと|これ|この|'
    r'され|して|した|しな|する|すれ|せず|その|それ|そう|たい|たく|ため|'
    r'ついて|った|って|て|と|な|に|の|は|へ|ほど|まで|ます|ません|まし|'
    r'む|も|や|よ|ら|る|れな|わ|んだ|んで|を|が|だ|でき|です|でな|ば)')

def count_common_japanese_words(text: str) -> int:
    """
    助詞/助動詞の出現数を数える
    :param text: 判定するテキスト
    :return: 助詞/助動詞の出現数
    """
    return len(pattern_japanese_common_words.findall(text))

def score_ja(text: str) -> float:
    """
    助詞/助動詞の出現頻度から日本語の品質をスコアつけ
    :return: スコア 0.25以上
    """
    count_commons = len(pattern_japanese_common_words.findall(text))
    count = count_japanese_characters(text)
    return count_commons / count if count > 0 else 0.0


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

def compile_words_pattern(words: List[str], prefix='', suffix=''):
    ws = []
    for w in words:
        if '.' in w and os.path.isfile(w):
            with open(w) as f:
                ws.extend(s.strip() for s in f.readlines() if len(s.strip()) > 0)
        else:
            ws.append(w)
    ws = list(set(ws))
    ws.sort()
    pattern = '|'.join(re.escape(w) for w in ws)
    return re.compile(f'{prefix}{pattern}{suffix}'), len(ws)

def count_words_pattern(words: List[str]):
    pattern, n = compile_words_pattern(words)
    if n > 100:
        counters = Counter()
    else:
        counters = None
    def count_ngwords(text: str):
        nonlocal counters, pattern, n
        results = set(pattern.findall(text))
        if counters is not None and len(results) > 1:
            counters.update(results)
            if len(counters) >= max(n/8, 100):
                #print(counters.most_common())
                ws = [w for w, i in counters.most_common()]
                pattern = re.compile('|'.join(re.escape(w) for w in ws))
                counters = None
        return len(results)
    return count_ngwords


def remove_footnote(words: List[str]):
    pattern, n = compile_words_pattern(words, prefix=r'\n(', suffix=r'\s*\n')
    def remove(text) -> dict[str, Any]:
        matched = pattern.search(text)
        if matched:
            text = text[: matched.start()]
        return text
    return remove

def remove_wikipedia_footnote():
    footnote_sections: list[str] = [
        "脚注",
        "関連項目",
        "日本国内の関連項目",
        "出典",
        "出典・脚注",
        "参照",
        "外部リンク",
        "参考文献",
        "その他関連事項",
        "Footnotes",
        "See also",
        "Further reading",
        "Bibliography",
        "References",
        "Notes",
        "Citations",
        "Sources",
        "External links",
    ]
    return remove_footnote(footnote_sections)
