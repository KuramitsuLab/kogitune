from typing import Any, List, Optional
import re
import os

from .base import TextFilter
from ..adhocargs import AdhocArguments

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

def compile_words_pattern(words: List[str], prefix='', suffix=''):
    if isinstance(words, str):
        words = words.split('|')

    ws = []
    for w in words:
        if w.endswith('.txt') and os.path.isfile(w):
            with open(w) as f:
                ws.extend(s.strip() for s in f.readlines() if len(s.strip()) > 0)
        else:
            ws.append(w)
    ws = list(set(ws))
    ws.sort()
    pattern = '|'.join(re.escape(w) for w in ws)
    return re.compile(f'{prefix}{pattern}{suffix}')

pattern_japanese_common_words = re.compile(
    r'(ある|あり|いた|いて|お|か|く|けど|けれど|こと|これ|この|'
    r'され|して|した|しな|する|すれ|せず|その|それ|そう|たい|たく|ため|'
    r'ついて|った|って|て|と|な|に|の|は|へ|ほど|まで|ます|ません|まし|'
    r'む|も|や|よ|ら|る|れな|わ|んだ|んで|を|が|だ|でき|です|でな|ば|。)')

class JapaneseWordCounter:
    """
    与えられたテキストに日本語単語が含まれるか判定する評価関数
    """
    def __init__(self, 
                 words: Optional[List[str]] = None, 
                 unification=False, 
                 ja_fraction=True, 
                 length_fraction=False, aargs=None):
        """
        与えられたテキストに日本語単語が含まれるか判定する評価関数を作る
        :param words: 日本語単語のリスト(省略した場合は助詞)
        :param unification: 単一化
        :param ja_fraction: 漢字/ひらがな/かたかな文字における比率
        :param length_fraction: 全テキストにおける比率 
        """
        aargs=AdhocArguments.to_adhoc(aargs)
        words = words or aargs['words']
        if words:
            self.pattern = compile_words_pattern(words)
        else:
            self.pattern = pattern_japanese_common_words
        self.unique = aargs[f'unification|={unification}']
        self.ja_fraction = aargs[f'ja_fraction|={ja_fraction}']
        self.length_fraction = aargs[f'length_fraction|={length_fraction}']

    def __call__(self, text):
        ws = self.pattern.findall(text)
        word_count = len(set(ws)) if self.unique else len(ws)
        if self.length_fraction:
            length_count =len(text)
            return (word_count / length_count) if length_count > 0 else 0.0
        elif self.ja_fraction:
            ja_count = count_japanese_characters(text)
            return (word_count / ja_count) if ja_count > 0 else 0.0 
        return word_count


pattern_wikipedia_footnote = compile_words_pattern([
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
    ])

class FootnoteFilter(TextFilter):
    """
    テキストから脚注を取り除くフィルター
    """

    def __init__(self, footnote_words: List[str] = None):
        """
        テキストから脚注を取り除くフィルターを作る
        :param footnote_words: 脚注の先頭(省略した場合は、Wikipedia 脚注)
        """
        if isinstance(footnote_words, list):
            self.pattern = compile_words_pattern(footnote_words)
        else:
            self.pattern = pattern_wikipedia_footnote

    def __call__(self, text):
        matched = self.pattern.search(text)
        if matched:
            text = text[: matched.start()]
        return text

