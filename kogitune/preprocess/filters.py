from typing import Optional, List
import numpy as np
import pandas as pd
import zlib
import math
import re

class Filter(object):
    def __init__(self, verbose=0):
        self.verbose = verbose

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

    def debug_print(self, *args):
        if self.verbose > 0:
            print('🦊', *args)

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
    def __init__(self, sep='\n', *filters):
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


class PercentileFilter(Filter):
    def __init__(self, score_fn, caption=None,
                 min_value=None, max_value=None, minq=None, maxq=None, 
                 stats_window = 1000, verbose=0):
        super().__init__(verbose=verbose)
        self.score_fn = score_fn
        self.min_value = min_value
        self.max_value = max_value
        self.minq = minq
        self.maxq = maxq
        self.values = []
        self.stats_window = stats_window
        self.caption = caption if caption else score_fn.__name__

    def filter(self, text):
        value = self.score_fn(text)
        if len(self.values) < self.stats_window:
            self.values.append(value)
            if len(self.values) % 100 == 10:
                 self.reestimate()
            if len(self.values) == self.stats_window:
                self.describe()

        if (self.min_value and self.min_value > value):
            return None
        if (self.max_value and self.max_value < value):
            return None
        return text
    
    def reestimate(self):
        a = np.array(self.values)
        if self.minq is not None:
            self.min_value = np.percentile(a, self.minq)
        if self.maxq is not None:
            self.max_value = np.percentile(a, self.maxq)

    def describe(self):
        df = pd.DataFrame({self.caption: self.values})
        print(df.describe(percentiles=[0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 0.9, 0.95]))

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

