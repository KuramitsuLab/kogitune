from typing import Optional
import zlib, math
from collections import Counter
import numpy as np
import pandas as pd

from .base import TextFilter

DEFAULT_PERCENTILES = [0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95]

class MaxMinFilter(TextFilter):
    """
    評価関数の最大値と最小値からフィルターする
    """
    def __init__(self, 
                 score_fn, 
                 min_value=None, 
                 max_value=None, 
                 minq=None, maxq=None, 
                 record_name:str = None,
                 window_size = 10000, 
                 histogram: Optional[str] = None,
                 funcname: Optional[str] = None,
                 percentiles = DEFAULT_PERCENTILES,
                 extract_samples=[],
                 verbose=0):
        """
        評価関数フィルタを作る
        :param min_value: 許容される最小値（省略した場合は全て許容される）
        :param max_value: 許容される最大値（省略した場合は全て許容される）
        :param minq: 最小値のパーセンタイル（省略した場合は、パーセンタイル補正はしない）
        :param maxq: 最大値のパーセンタイル（省略した場合は、パーセンタイル補正はしない）
        :param record_name: 評価値を記録するときのエントリー名
        :param histogram: Histogram 名
        """
        super().__init__(verbose=verbose)
        self.score_fn = score_fn
        self.min_value = min_value
        self.max_value = max_value
        self.minq = minq
        self.maxq = maxq
        self.record_name = record_name
        self.values = []
        self.window_size = window_size
        self.recheck_factor = 100
        if funcname:
            self.funcname = funcname
        elif hasattr(score_fn, 'name'):
            self.funcname = score_fn.name
        elif hasattr(score_fn, '__name__'):
            self.funcname = score_fn.__name__
        else:
            self.funcname = score_fn.__class__.__name__
        self.histogram = histogram
        self.percentiles = percentiles
        self.extract_samples = set(round(x,2) for x in extract_samples)

    def filter(self, text):
        value = self.score_fn(text)
        if self.record is not None:
            if self.record_name:
                self.record[self.record_name] = round(value,5)
            if len(self.values) < self.window_size:
                self.values.append(value)
                if len(self.values) % self.recheck_factor == 1:
                    self.recheck()
                    self.recheck_factor *= 2
                if len(self.values) == self.window_size:
                    _describe(self.values, self.funcname, self.histogram, self.percentiles)
            if len(self.extract_samples) > 0:
                key = round(value, 2)
                if key in self.extract_samples:
                    print(f'{self.funcname}[sample={value:.5f}]', text)
                    self.extract_samples.discard(key)
                else:
                    key = round(value, 1)
                    if key in self.extract_samples:
                        print(f'{self.funcname}[sample={value:5f}]', text)
                        self.extract_samples.discard(key)

        if (self.min_value and self.min_value > value):
            return None
        if (self.max_value and self.max_value < value):
            return None
        return text
    
    def recheck(self):
        a = np.array(self.values)
        if self.minq is not None:
            min_value = self.min_value
            self.min_value = round(np.percentile(a, self.minq),5)
            if self.funcname:
                print(f'{self.funcname}[{self.recheck_factor}] min_value: {min_value} => {self.min_value}')
        if self.maxq is not None:
            max_value = self.max_value
            self.max_value = round(np.percentile(a, self.maxq),5)
            if self.funcname:
                print(f'{self.funcname}[{self.recheck_factor}] max_value: {max_value} => {self.max_value}')

def _describe(values, funcname, histogram, percentiles=DEFAULT_PERCENTILES):
    caption = histogram or funcname
    if funcname is None:
        return
    df = pd.DataFrame({caption: values})
    print(df.describe(percentiles=percentiles))
    if histogram is None:
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.displot(df, stat='probability')
        filename = caption.replace(' ', '_').replace('/', '_')
        print(f'Saving Histgram {filename}.png Data {filename}.csv')
        df.to_csv(f'{filename}.csv', index=None)
        plt.savefig(filename)
        plt.clf()
    except:
        pass


class TokenizerCompression(object):
    """
    トークンナイザーの圧縮率による評価関数
    """

    def __init__(self, tokenizer: str, 
                 length=None, head=None, chars_per_tokens=False, legacy=False):
        """
        トークンナイザーの圧縮率による評価関数を作る
        :param tokenizer: トークンナイザー(もしくはトークンナイザー名)   
        :param head: 指定された先頭の文字数だけチェックする（デフォルトは全体）   
        :param chars_per_tokens: 圧縮率の計算を1トークン辺りの文字数(chars/tokens)にする。
        """
        self.tokenizer = _load_tokenizer(tokenizer, legacy=legacy)
        self.head = head or length
        self.chars_per_tokens = chars_per_tokens

    def __call__(self, text):
        if self.head:
            text = text[:self.head]
        text_length = len(text)
        if text_length == 0:
            return 1
        token_length = len(self.tokenizer.encode(text))
        if self.chars_per_tokens:
            return text_length / token_length 
        return text_length / token_length

class TokenizerEntropy(object):
    """
    任意のトークンリストのエントロピーを計算でき、それによりトークンの分布がどの程度多様か、
    またはどの程度予測可能かが分かります。
    エントロピーが高いほど、トークンの分布は多様で予測が難しいと言えます。
    逆にエントロピーが低い場合、トークンの分布は比較的均一で予測が容易です。
    """

    def __init__(self, tokenizer: str, length=None, head=None, legacy=False):
        """
        トークンナイザーによるエントロピー評価関数を作る
        :param tokenizer: トークンナイザー(もしくはトークンナイザー名)   
        """
        self.tokenizer = _load_tokenizer(tokenizer, legacy=legacy)

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

def _load_tokenizer(tokenizer: str, legacy=False):
    from transformers import AutoTokenizer
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer, legacy=legacy, trust_remote_code=True, use_fast=False)
    return tokenizer


class ZLibCompression(object):
    """
    Zlib圧縮率による評価関数
    """
    def __init__(self, length_factor: float = 0.0):
        """
        Zlib圧縮率による評価関数をつくる
        """
        self.length_factor = length_factor

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


    