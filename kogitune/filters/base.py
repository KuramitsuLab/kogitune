from typing import Optional, List
import json
import os
import re

from ..adhoc_args import AdhocArguments, adhoc
from kogitune.utils_file import zopen, filelines, read_multilines, rename_with_linenum, list_filenames

from multiprocess import Pool

_dummy_record = {}

class TextFilter(object):
    """
    テキストフィルターの規定クラス
    """
    def __init__(self, **kwargs):
        """
        新しいテキストフィルタを作る
        """
        self.kwargs = kwargs

    def name(self):
        return self.__class__.__name__

    def as_json(self):
        return self.kwargs

    def __call__(self, text: str, record: dict) -> Optional[str]:
        return text

    def __repr__(self):
        return json.dumps(self.as_json(), indent=2)

    def save_as_json(self, filename: str):
        with open(filename, 'w') as w:
            json.dump(self.as_json(), w, indent=2)

    def to_report(self, section):
        pass

    def from_jsonl(self, filename: str, output_path:str=None, N=-1, num_workers=1):
        if num_workers == 1 or output_path is None:
            self._from_jsonl_single(filename, N=N, output_path=output_path)
        else:
            self._from_jsonl_multi(filename, output_path=output_path, N=N, num_workers=num_workers)

    def _from_jsonl_single(self, filenames: str, N=-1, output_path=None):
        filenames = list_filenames(filenames)
        adhoc.setlog('filter', input_files=filenames, filter_config =self.as_json())
        w = None
        if isinstance(output_path, str):
            w = zopen(output_path, 'wt')
        c=0
        n=0
        for line in filelines(filenames, N=N, line_reader='jsonl'):
            record = {}
            line = self(line, record)
            n+=1
            if line:
                record['text'] = line
                c+=1
                if w:
                    print(json.dumps(record, ensure_ascii=False), file=w)
                else:
                    if c < 100:
                        print(record)
        if output_path:
            newpath = rename_with_linenum(output_path, N=c, ext='json')
            adhoc.print(f'完了//Complete: {newpath} {c}/{n} {c/n:.3f}')
            adhoc.setlog('filter', output_file = newpath, total=n, filtered=c)
            adhoc.save_log(newpath)

    def invoke_as_multi(self, text):
        return self(text, _dummy_record)

    def _from_jsonl_multi(self, filenames: str, output_path:str=None, N=-1, num_workers=1):
        filenames = list_filenames(filenames)
        adhoc.setlog('filter', input_files=filenames, filter_config =self.as_json())
        c=0
        n=0
        with zopen(output_path, 'wt') as w:
            with Pool(num_workers) as pool:
                for lines in read_multilines(filenames, N=N, bufsize=1000 * num_workers, line_reader='jsonl'):
                    lines = pool.map(self.invoke_as_multi, lines)
                    n += len(lines)
                    for text in lines:
                        if text:
                            c+=1
                            print(json.dumps({'text': text}, ensure_ascii=False), file=w)
        newpath = rename_with_linenum(output_path, N=c, ext='json')
        adhoc.print(f'完了//Complete: {newpath} {c}/{n} {c/n:.3f}')
        adhoc.setlog('filter', output_file = newpath, total=n, filtered=c)
        adhoc.save_log(newpath)

class ComposeFilter(TextFilter):
    """
    テキストフィルタを合成する
    :param filters:
    """
    def __init__(self, *filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = tuple(filters)

    def __call__(self, text: str, record: dict) -> Optional[str]:
        for f in self.filters:
            text = f(text, record)
            if text is None:
                return None
        return text

    def as_json(self):
        return [e.as_json() for e in self.filters]


class ChoiceFilter(TextFilter):
    """
    テキストフィルタを合成する
    :param filters:
    """
    def __init__(self, *filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = tuple(filters)

    def as_json(self):
        return ['choice'] + [e.as_json() for e in self.filters]

    def __call__(self, text: str, record: dict) -> Optional[str]:
        for f in self.filters:
            text2 = f(text, record)
            if text2 is not None:
                return text2
        return None


# class ExtractFilter(ComposeFilter):
#     def __init__(self, extract_fn, *filters):
#         super().__init__(*filters)
#         self.extract_fn = extract_fn

#    def __call__(self, text: str, record: dict = None) -> Optional[str]:
#         doc, text = self.extract_fn(text)
#         for f in self.filters:
#             if f(doc) is None:
#                 return None
#         return text


class ScoreFunction(object):
    def __init__(self, **kwargs):
        if len(kwargs) > 0:
            adhoc.print(f'Unused [{self.name()}]', kwargs)

    def name(self):
        return self.__class__.__name__

    def as_json(self):
        return None

    def __repr__(self):
        return json.dumps(self.as_json(), indent=2)

    def __call__(self, text: str):
        return len(text)

def compile_pattern_for_words(words: List[str], prefix='', suffix=''):
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
    if len(prefix) > 0 or len(suffix) > 0:
        re.compile(f'{prefix}({pattern}){suffix}')
    return re.compile(pattern)


