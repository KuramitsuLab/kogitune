from typing import Optional
import zlib, math
from collections import Counter

from .commons import ScoreFunction
import kogitune.adhocs as adhoc

class TextLength(ScoreFunction):
    """
    文字列長による評価関数
    この関数は役に立ちます。
    """

    def __init__(self, **kwargs):
        """
        文字列長による評価関数を作成する
        """
        super().__init__(**kwargs)

    def name(self):
        return self.__class__.__name__

    def as_json(self):
        return {'score': self.name()}

    def __call__(self, text: str):
        return len(text)


class TokenizerCompression(ScoreFunction):
    """
    トークンナイザーの圧縮率による評価関数
    """

    def __init__(self, 
                 tokenizer: str = None, 
                 head=None, length=None, 
                 chars_per_tokens=False, 
                 **kwargs):
        """
        トークンナイザーの圧縮率（1トークン辺りの文字数）による評価関数を作る
        :param tokenizer: トークンナイザー(もしくはトークンナイザー名)   
        :param head: 指定された先頭の文字数だけチェックする（デフォルトは全体）
        """
        super().__init__(**kwargs)
        self.tokenizer = adhoc.load_tokenizer(tokenizer=tokenizer)
        self.chars_per_tokens = chars_per_tokens
        self.head = head
        self.length = length

    def as_json(self):
        return {
            'score': self.name(), 
            'tokenizer': self.tokenizer.name_or_path, 
            'chars_per_tokens': self.chars_per_tokens, 
        }

    def __call__(self, text):
        if self.head:
            text = text[:self.head]
        text_length = len(text)
        if text_length == 0:
            return 1
        token_length = len(self.tokenizer.encode(text))
        return text_length / token_length 

class TokenizerEntropy(ScoreFunction):
    """
    任意のトークンリストのエントロピーを計算でき、それによりトークンの分布がどの程度多様か、
    またはどの程度予測可能かが分かります。
    エントロピーが高いほど、トークンの分布は多様で予測が難しいと言えます。
    逆にエントロピーが低い場合、トークンの分布は比較的均一で予測が容易です。
    :param tokenizer:
    """

    def __init__(self, tokenizer=None, **kwargs):
        """
        トークンナイザーによるエントロピー評価関数を作る
        :param tokenizer: トークンナイザー(もしくはトークンナイザー名)   
        """
        super().__init__(tokenizer=None, **kwargs)
        self.tokenizer = adhoc.load_tokenizer(tokenizer=tokenizer)

    def as_json(self):
        return {
            'score': self.name(),
            'tokenizer': self.tokenizer.name_or_path, 
        }

    def __call__(self, text):
        tokens = self.tokenizer.encode(text)
        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        # Calculate entropy
        entropy = 0
        for count in token_counts.values():
            probability = count / total_tokens
            entropy -= probability * math.log(probability, 2)
        return entropy


class ZLibCompression(ScoreFunction):
    """
    Zlib圧縮率による評価関数
    """
    def __init__(self, length_factor=0.0, **kwargs):
        """
        Zlib圧縮率による評価関数をつくる
        :param length_factor: 
        """
        super().__init__(**kwargs)
        self.length_factor = length_factor

    def as_json(self):
        return {
            'score': self.name(),
            'length_factor': self.length_factor, 
        }

    def __call__(self, text):
        encoded = text.encode("utf-8", errors='ignore')
        encoded_length = len(encoded)
        if encoded_length == 0:
            return 0.0
        compressed = zlib.compress(encoded, level=9)    
        compressed_length = len(compressed)
        ratio = compressed_length / encoded_length
        length_penalty = (
            self.length_factor * math.log(encoded_length) if self.length_factor else 0.0
        )
        score = ratio + length_penalty
        return score
