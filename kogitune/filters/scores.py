from typing import Optional
import pandas as pd
import urllib.parse

from .commons import TextFilter

from .scores_base import ScoreFunction, TokenCount, TokenFraction, CharactorCount, CharactorFraction, AlphaCount
from .OLDscores_text import TextLength, ZLibCompression, TokenizerEntropy
from .utils_en import EnglishWordCounter
from .utils_ja import JapaneseWordCounter

import kogitune.adhocs as adhoc

SCORE_FUNCTION_MAP = {
    'text-length': TextLength,
    'zlib': ZLibCompression,
    'char': CharactorCount,
    'char-fraction': CharactorFraction,
    'token': TokenCount,
    'token-fraction': TokenFraction,
    'token-entropy': TokenizerEntropy,
    'alpha-fraction': AlphaFraction,
    'word-en': EnglishWordCounter,
    'word-ja': JapaneseWordCounter,
}

def load_eval_fn(score_path):
    if isinstance(score_path, ScoreFunction):
        return score_path
    path, args = adhoc.parse_path_args(score_path)
    if '.' in path:
        func = adhoc.load_class(path)
        if not issubclass(func, ScoreFunction):
            raise TypeError(f'{path} is not a subclass of ScoreFunction')
        return func(**args)
    if path in SCORE_FUNCTION_MAP:
        func = SCORE_FUNCTION_MAP[path]
        return func(**args)
    else:
        adhoc.warn(unknown_score_path=score_path, expected=list(SCORE_FUNCTION_MAP.keys()))

## MaxMin

DEFAULT_PERCENTILES = [0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95]

class MaxMinFilter(TextFilter):
    """
    評価関数の最大値と最小値からフィルターする
    """
    def __init__(self, 
                 text_eval, 
                 min_value=None, 
                 max_value=None, 
                 record_key = None,
                 histogram_sample = 0, 
                 save_to=None,
                 percentiles = DEFAULT_PERCENTILES,
                 **kwargs):
        """
        評価関数フィルタを作る
        :param min_value: 許容される最小値 (inclusive)（省略した場合は全て許容される）
        :param max_value: 許容される最大値 (inclusive)（省略した場合は全て許容される）
        # :param minq: 最小値のパーセンタイル（省略した場合は、パーセンタイル補正はしない）
        # :param maxq: 最大値のパーセンタイル（省略した場合は、パーセンタイル補正はしない）
        :param record_name: 評価値を記録するときのエントリー名
        :param histogram_sample: ヒストグラムを保存したいときのサンプル数
        :param save_to: ヒストグラムの保存先
        """
        super().__init__(**kwargs)
        self._score_func = load_eval_fn(self.eval)
        self._record_key = record_key
        self._funcname = record_key or self._score_func.name()
        self._histogram_sample = histogram_sample
        self._save_to = save_to
        if self._save_to is not None and histogram_sample == 0:
            self._histogram_sample = 10000
        if self._histogram_sample > 0:
            self._values = []
            self._percentiles = percentiles

    def __call__(self, text: str, record: dict) -> Optional[str]:
        value = self._score_func(text)
        if self._record_key:
            record[self._record_key] = round(value,5)
        if self._histogram_sample > 0:
            self._values.append(value)
            if len(self._values) == self._histogram_sample:
                _describe(self._values, self._funcname, self._percentiles, self._save_to)
        if (self.min_value and self.min_value > value):
            #record['drop'] = 'DROP[{self.funcname}:{value}>]\n{text}\n'
            return None
        if (self.max_value and self.max_value < value):
            # if self.verbose > 0:
            #     self.debug_print(f'DROP[{self.funcname}:{value}>]\n{text}\n')
            return None
        return text

def _describe(values, funcname, percentiles=DEFAULT_PERCENTILES, filename=None):
    df = pd.DataFrame({funcname: values})
    print(df.describe(percentiles=percentiles))
    if filename is None:
        adhoc.print(f"ヒストグラムの詳細を知りたいときは、save_to='file.csv' を設定しよう")
    else:
        df.to_csv(f'{filename}.csv', index=None)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.displot(df, stat='probability')
            plt.savefig(filename)
            adhoc.print(f'ヒストグラムを保存しました: {filename}.png')
            plt.clf()
        except:
            pass

# def urlencode(d:dict):
#     return urllib.parse.urlencode(d)

def encode_path_arguments(path:str, args:dict, without_keys:list):
    args = args.copy()
    if isinstance(path, str):
        if isinstance(without_keys, str):
            without_keys = without_keys.split('|')
        for key in without_keys:
            args.pop(key, None)
        if len(args) > 0:
            return f'{path}?{urllib.parse.urlencode(args)}'
    return path

def maxmin(_score, **kwargs):
    args = kwargs
    if 'min' in args and 'min_value' not in args:
        args['min_value'] = args.pop('min')
    if 'max' in args and 'max_value' not in args:
        args['max_value'] = args.pop('max')
    score_path = encode_path_arguments(_score, args, 
                'min_value|max_value|record_key|histogram_sample|save_to|percentiles')
    return MaxMinFilter(score_path=score_path, **args)

