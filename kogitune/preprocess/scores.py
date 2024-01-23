import zlib, math
from collections import Counter

def _load_tokenizer(tokenizer: str, legacy=False):
    from transformers import AutoTokenizer
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer, legacy=legacy, trust_remote_code=True, use_fast=False)
    return tokenizer

class CharacterPerToken(object):

    def __init__(self, tokenizer: str, length=None, head=None, legacy=False):
        self.tokenizer = _load_tokenizer(tokenizer, legacy=legacy)
        self.head = head or length

    def __call__(self, text):
        if self.head:
            text = text[:self.head]
        text_length = len(text)
        token_length = len(self.tokenizer.encode(text))
        return text_length / token_length 




def zlib_ratio(text:str, length_factor: float = 0.0)->float:
    encoded = text.encode("utf-8", errors='ignore')
    encoded_length = len(encoded)
    if encoded_length == 0:
        return 0.0
    compressed = zlib.compress(encoded, level=9)    
    compressed_length = len(compressed)
    ratio = compressed_length / encoded_length
    length_penalty = (
        length_factor * math.log(encoded_length) if length_factor else 0.0
    )
    score = ratio + length_penalty
    return score

class token_entropy(object):
    """
    任意のトークンリストのエントロピーを計算でき、それによりトークンの分布がどの程度多様か、
    またはどの程度予測可能かが分かります。
    エントロピーが高いほど、トークンの分布は多様で予測が難しいと言えます。
    逆にエントロピーが低い場合、トークンの分布は比較的均一で予測が容易です。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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

    