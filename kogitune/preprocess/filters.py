from typing import Optional, List
import sys
import json
from kogitune.file_utils import zopen, filelines

import numpy as np
import pandas as pd

class Filter(object):
    def __init__(self, verbose=0):
        """
        verbose は指定した個数だけデバック出力する
        """
        self.verbose = verbose
        self.record = None

    def __call__(self, text: str) -> Optional[str]:
        if self.verbose > 0:
            filtered_text = self.filter(text)
            self.debug_print(f'[{self.verbose}] {repr(self)}')
            print('|'+text.replace('\n', '\n|'))
            print('==>')
            print(filtered_text)
            self.verbose -= 1
            return filtered_text
        return self.filter(text)
 
    def filter(self, text: str)-> Optional[str]:
        return text

    def read_jsonl(self, filename: str, N=-1, output_path=None):
        w = None
        if isinstance(output_path, str):
            w = zopen(output_path, 'wt')
        else:
            self.verbose = 10
        for line in filelines(filename, N=N):
            text = json.loads(line)['text']
            text = self(text)
            if text:
                if w:
                    print(json.dumps({'text': text}, ensure_ascii=False), file=w)
                else:
                    self.debug_print(text)

    def debug_print(self, *args):
        if self.verbose > 0:
            print('🦊', *args)
            self.verbose -= 1


class ComposeFilter(Filter):
    def __init__(self, *filters):
        super().__init__(verbose=0)
        self.filters = filters

    def __call__(self, text):
        for f in self.filters:
            text = f(text)
            if text is None:
                return None
        return text

class ChoiceFilter(ComposeFilter):
    def __init__(self, *filters):
        super().__init__(*filters)

    def __call__(self, text):
        for f in self.filters:
            text2 = f(text)
            if text2 is not None:
                return text2
        return None

class ExtractFilter(ComposeFilter):
    def __init__(self, extract_fn, *filters):
        super().__init__(*filters)
        self.extract_fn = extract_fn

    def __call__(self, text):
        doc, text = self.extract_fn(text)
        for f in self.filters:
            if f(doc) is None:
                return None
        return text

class LineByLineFilter(ComposeFilter):
    def __init__(self, *filters, sep='\n'):
        super().__init__(*filters)
        self.sep = sep

    def __call__(self, text):
        lines = []
        for line in text.split(self.sep):
            for f in self.filters:
                line = f(line)
                if line is None:
                    break
            if line:
                lines.append(line)
            else:
                lines.append('')
        if len(lines) == 0:
            return None
        return self.sep.join(lines)
    
DEFAULT_PERCENTILES = [0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95]

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
        from matplotlib.ticker import PercentFormatter
        sns.displot(df, stat='probability')
        filename = caption.replace(' ', '_').replace('/', '_')
        print(f'Saving Histgram {filename}.png Data {filename}.csv')
        df.to_csv(f'{filename}.csv', index=None)
        plt.savefig(filename)
        plt.clf()
    except:
        pass

class PercentileFilter(Filter):
    def __init__(self, 
                 score_fn, 
                 min_value=None, 
                 max_value=None, 
                 minq=None, maxq=None, 
                 histogram: Optional[str] = None,
                 funcname: Optional[str] = None,
                 window_size = 10000, 
                 percentiles = DEFAULT_PERCENTILES,
                 extract_samples=[],
                 verbose=0):
        """
        histogram: Histogram 名
        """
        super().__init__(verbose=verbose)
        self.score_fn = score_fn
        self.min_value = min_value
        self.max_value = max_value
        self.minq = minq
        self.maxq = maxq
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
                print(f'{self.funcname}[{self.recheck_factor}] min_value: {max_value} => {self.max_value}')
    



# class ZLibFilter(PercentileFilter):  
#     def __init__(self, length_factor = 0.0,
#                   min_value=None, max_value=None, minq=10, maxq=90, verbose=0):
#         super().__init__(min_value=min_value, max_value=max_value, 
#                          minq=minq, maxq=maxq, verbose=verbose)
#         self.length_factor = length_factor

#     def filter(self, text:str):
#         encoded = text.encode("utf-8", errors='ignore')
#         compressed = zlib.compress(encoded, level=9)
#         encoded_length = len(encoded)
#         compressed_length = len(compressed)
#         ratio = compressed_length / encoded_length
#         length_penalty = (
#             self.length_factor * math.log(encoded_length) if self.length_factor else 0.0
#         )
#         return ratio + length_penalty

import unicodedata

class UnicodeFilter(Filter):

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def filter(self, text):
        return unicodedata.normalize('NFKC', text)


class DuplicatedLineFilter(Filter):

    def __init__(self, prefix_length=8, verbose=0):
        super().__init__(verbose=verbose)
        self.prefix_length=prefix_length

    def filter(self, text):
        lines = ['']
        for line in text.split('\n'):
            prev = lines[-1]
            if self.prefix_length < len(line) < 80 and prev.startswith(line[:self.prefix_length]):
                if len(line) > len(prev):
                    lines[-1] = line
                continue
            if len(line.strip()) == 0 and prev == '':
                continue
            if 1 < len(prev) < 40 and not prev.endswith('。') and len(line) > 80 and line.count('。') > 1:
                if lines[-1] != '':
                    lines[-1] = f'\n{prev}'
            lines.append(line)
        return '\n'.join(lines[1:])

