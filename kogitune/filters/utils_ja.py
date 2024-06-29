from typing import Any, List, Optional
import re

from .commons import TextFilter, ScoreFunction, compile_pattern_for_words

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

class JapaneseCounter(CharCounter):
    def __init__(self, regex:str, unique=False, **kwargs) -> None:
        super().__init__(unique=unique, **kwargs)
        self.pattern = re.compile(regex)

    def __call__(self, text:str) -> int:
        if self.unique:
            return len(set(self.pattern.findall(text)))
        return len(self.pattern.findall(text))


pattern_japanese_common_words = compile_pattern_for_words(
    r'(ある|あり|いた|いて|お|か|く|けど|けれど|こと|これ|この|' + 
    r'され|して|した|しな|する|すれ|せず|その|それ|そう|たい|たく|ため|' +
    r'ついて|った|って|て|と|な|に|の|は|へ|ほど|まで|ます|ません|まし|' +
    r'む|も|や|よ|ら|る|れな|わ|んだ|んで|を|が|だ|でき|です|でな|ば|。)'
)

class JapaneseWordCounter(ScoreFunction):
    """
    与えられたテキストに日本語単語が含まれるか判定する評価関数
    """
    def __init__(self, 
                 words: Optional[List[str]] = None, 
                 unique=False, 
                 japanese_fraction=False, 
                 length_fraction=True, **kwargs):
        """
        与えられたテキストに日本語単語が含まれるか判定する評価関数を作る
        :param words: 日本語単語のリスト(省略した場合は助詞)
        :param unique: 単一化
        :param japanese_fraction: 漢字/ひらがな/かたかな文字における比率
        :param length_fraction: 全テキストにおける比率 
        """
        self.__init__(**kwargs)
        if words:
            self.pattern = compile_pattern_for_words(words)
        else:
            self.pattern = pattern_japanese_common_words
        self.unique = unique
        self.length_fraction = length_fraction
        self.japanese_fraction = japanese_fraction

    def as_json(self):
        return {
            'score': self.name(),
            'unique': self.unique,
            'length_fraction': self.length_fraction,
            'japanese_fraction': self.japanese_fraction,
        }

    def __call__(self, text):
        ws = self.pattern.findall(text)
        word_count = len(set(ws)) if self.unique else len(ws)
        if self.length_fraction:
            length_count =len(text)
            return (word_count / length_count) if length_count > 0 else 0.0
        elif self.japanese_fraction:
            ja_count = count_japanese_characters(text)
            return (word_count / ja_count) if ja_count > 0 else 0.0 
        return word_count

pattern_wikipedia_footnote = compile_pattern_for_words([
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

    def __init__(self, words: List[str] = None):
        """
        テキストから脚注を取り除くフィルターを作る
        :param footnote_words: 脚注の先頭(省略した場合は、Wikipedia 脚注)
        """
        if isinstance(words, list):
            self.pattern = compile_pattern_for_words(words)
        else:
            self.pattern = pattern_wikipedia_footnote

    def __call__(self, text):
        matched = self.pattern.search(text)
        if matched:
            text = text[: matched.start()]
        return text

