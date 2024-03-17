from typing import Optional
import zlib, math, json
from collections import Counter
import numpy as np
import pandas as pd

from .base import TextFilter

from ..adhoc_args import parse_path_arguments, configurable_tokenizer, verbose_print

class MaxMinScoreFunction(object):
    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            verbose_print(f'Unused [{self.name()}]', kwargs)

    def name(self):
        return self.__class__.__name__

    def as_json(self):
        return None

    def __repr__(self):
        return json.dumps(self.as_json(), indent=2)

    def __call__(self, text: str):
        return len(text)

class TextLength(MaxMinScoreFunction):
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


class TokenizerCompression(MaxMinScoreFunction):
    """
    トークンナイザーの圧縮率による評価関数
    """

    def __init__(self, 
                 tokenizer: str = None, 
                 head=None, length=None, 
                 chars_per_tokens=False, 
                 **kwargs):
        """
        トークンナイザーの圧縮率による評価関数を作る
        :param tokenizer: トークンナイザー(もしくはトークンナイザー名)   
        :param head: 指定された先頭の文字数だけチェックする（デフォルトは全体）   
        :param chars_per_tokens: 圧縮率の計算を1トークン辺りの文字数(chars/tokens)にする。
        """
        super().__init__(**kwargs)
        self.tokenizer = configurable_tokenizer(tokenizer=tokenizer)
        self.chars_per_tokens = chars_per_tokens
        self.head = head
        self.length = length
        self.zlib_fraction = zlib_fraction

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
        if self.zlib_fraction:
            encoded = text.encode("utf-8", errors='ignore')
            compressed = zlib.compress(encoded, level=9)    
            text_length = len(compressed)
        if self.chars_per_tokens:
            return text_length / token_length 
        return token_length / text_length

class TokenizerEntropy(MaxMinScoreFunction):
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
        super().__init__(**kwargs)
        self.tokenizer = configurable_tokenizer(tokenizer=tokenizer)

    def as_json(self):
        return {
            'score': 'TokenizerEntropy', 
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


class ZLibCompression(MaxMinScoreFunction):
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
            'score': 'ZLibCompression', 
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


function_map = {
    'zlib': ZLibCompression
}

def score_function(score_path):
    if isinstance(score_path, MaxMinScoreFunction):
        return score_path
    path, args = parse_path_arguments(score_path)
    if path in function_map:
        func = function_map[path]
    else:
        ns = globals()
        if path in ns:
            func = ns[path]
    return func(**args)

## MaxMin

DEFAULT_PERCENTILES = [0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95]

class MaxMinFilter(TextFilter):
    """
    評価関数の最大値と最小値からフィルターする
    """
    def __init__(self, 
                 score_path, 
                 min_value=None, 
                 max_value=None, 
                #  minq=None, 
                #  maxq=None, 
                 record_key = None,
                 histogram_size = 0, 
                 save_to=None,
                 percentiles = DEFAULT_PERCENTILES,
                 **kwargs):
        """
        評価関数フィルタを作る
        :param min_value: 許容される最小値 (inclusive)（省略した場合は全て許容される）
        :param max_value: 許容される最大値 (inclusive)（省略した場合は全て許容される）
        :param minq: 最小値のパーセンタイル（省略した場合は、パーセンタイル補正はしない）
        :param maxq: 最大値のパーセンタイル（省略した場合は、パーセンタイル補正はしない）
        :param record_name: 評価値を記録するときのエントリー名
        :param histogram_size: ヒストグラムを保存したいときのサンプル数
        :param save_to: ヒストグラムの保存先
        """
        super().__init__(**kwargs)
        self.score_func = score_function(score_path)
        self.min_value = min_value
        self.max_value = max_value
        # self.minq = minq
        # self.maxq = maxq
        self.record_key = record_key
        self.funcname = record_key or self.score_func.name()
        self.histogram_size = histogram_size
        self.save_to = save_to
        if save_to is not None and histogram_size == 0:
            histogram_size = 10000
        if histogram_size > 0:
            self.values = []
            self.percentiles = percentiles

    def as_json(self):
        return {
            'maxmin': self.score_func.as_json(),
            'max_value': self.max_value,
            'min_value': self.min_value,
        }

    def __call__(self, text: str, record: dict) -> Optional[str]:
        value = self.score_fn(text)
        if self.record_key:
            record[self.record_key] = round(value,5)
        if self.histogram_size > 0:
            self.values.append(value)
            if len(self.values) == self.window_size:
                _describe(self.values, self.funcname, self.percentiles)
        if (self.min_value and self.min_value > value):
            #record['drop'] = 'DROP[{self.funcname}:{value}>]\n{text}\n'
            return None
        if (self.max_value and self.max_value < value):
            # if self.verbose > 0:
            #     self.debug_print(f'DROP[{self.funcname}:{value}>]\n{text}\n')
            return None
        return text

    # def filter(self, text):
    #     value = self.score_fn(text)
    #     if self.record is not None:
    #         if self.record_name:
    #             self.record[self.record_name] = round(value,5)
    #         if len(self.values) < self.window_size:
    #             self.values.append(value)
    #         if len(self.extract_samples) > 0:
    #             key = round(value, 2)
    #             if key in self.extract_samples:
    #                 print(f'{self.funcname}[sample={value:.5f}]', text)
    #                 self.extract_samples.discard(key)
    #             else:
    #                 key = round(value, 1)
    #                 if key in self.extract_samples:
    #                     print(f'{self.funcname}[sample={value:5f}]', text)
    #                     self.extract_samples.discard(key)
    #     if (self.min_value and self.min_value > value):
    #         if self.verbose > 0:
    #             self.debug_print(f'DROP[{self.funcname}:{value}<]\n{text}\n')
    #         return None
    #     if (self.max_value and self.max_value < value):
    #         if self.verbose > 0:
    #             self.debug_print(f'DROP[{self.funcname}:{value}>]\n{text}\n')
    #         return None
    #     return text
    
    # def recheck(self):
    #     a = np.array(self.values)
    #     if self.minq is not None:
    #         min_value = self.min_value
    #         self.min_value = round(np.percentile(a, self.minq),5)
    #         if self.funcname:
    #             print(f'{self.funcname}[{self.recheck_factor}] min_value: {min_value} => {self.min_value}')
    #     if self.maxq is not None:
    #         max_value = self.max_value
    #         self.max_value = round(np.percentile(a, self.maxq),5)
    #         if self.funcname:
    #             print(f'{self.funcname}[{self.recheck_factor}] max_value: {max_value} => {self.max_value}')

def _describe(values, funcname, percentiles=DEFAULT_PERCENTILES, filename=None):
    df = pd.DataFrame({funcname: values})
    print(df.describe(percentiles=percentiles))
    if filename is not None:
        df.to_csv(f'{filename}.csv', index=None)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.displot(df, stat='probability')
            plt.savefig(filename)
            verbose_print(f'ヒストグラムを保存しました: {filename}.png')
            plt.clf()
        except:
            pass

"""

class MaxMinFilter(TextFilter):
    def __init__(self, 
                 score_fn, 
                 min_value=None, 
                 max_value=None, 
                 minq=None, 
                 maxq=None, 
                 window_size = 10000, 
                 record_name:str = None,
                 histogram: Optional[str] = None,
                 funcname: Optional[str] = None,
                 percentiles = DEFAULT_PERCENTILES,
                 extract_samples=[],
                 verbose=0):
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
            if self.verbose > 0:
                self.debug_print(f'DROP[{self.funcname}:{value}<]\n{text}\n')
            return None
        if (self.max_value and self.max_value < value):
            if self.verbose > 0:
                self.debug_print(f'DROP[{self.funcname}:{value}>]\n{text}\n')
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

"""
