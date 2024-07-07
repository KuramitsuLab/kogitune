from typing import Optional, List, Union, Any
import os
import sys
import json
import re

import kogitune.adhocs as adhoc

from kogitune.stores.files import (
    zopen, 
    filelines, 
    read_multilines, 
    rename_linenum, 
    list_filenames
)

from multiprocess import Pool

_dummy_record = {}

def is_json_valuable(v):
    if isinstance(v, (bool, int, float, str)) or v is None:
        return True
    if isinstance(v, (list, tuple)):
        for x in v:
            if not is_json_valuable(x):
                return False
        return True
    if isinstance(v, dict):
        for key, value in v.items():
            if not isinstance(key, str) or not is_json_valuable(value):
                return False
        return True
    return False

def jsonsafe(v):
    if hasattr(v, 'as_json'):
        return v.as_json()
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [jsonsafe(x) for x in v]
    if isinstance(v, dict):
        return {f'{key}': jsonsafe(value) for key, value in v.items()}
    return None

class TextFilter(object):
    """
    テキストフィルターの規定クラス
    """
    def __init__(self, *args, **kwargs):
        """
        新しいテキストフィルタを作る
        """
        self.rec = {}

    def __call__(self, text: str, record: dict) -> Optional[str]:
        return text

    def describe(self):
        pass

    def __repr__(self):
        return repr(self.rec)

    def as_json(self):
        cls = self.__class__
        return {
            'class_path' : f'{cls.__module__}.{cls.__name__}',
            'args': self.rec,
        }

    def save_config(self, filepath: str):
        with open(filepath, 'w') as w:
            json.dump(self.as_json(), w, indent=2)

    def transform(self, sample_texts: Union[str, List[str]], return_text_only=False):
        """
        サンプルテキストをフィルターしてみる。
        """
        if isinstance(sample_texts, str):
            sample_texts = [sample_texts]
        result = []
        for text in sample_texts:
            data={}
            filtered_text = self(text, data)
            result.append({
                'input': text,
                'output': filtered_text,
                'is_same': text == filtered_text,
                'details': data,
            })
        if return_text_only:
            result = [d['output'] for d in result]            
        return result[0] if len(result) == 1 else result
    
    def from_jsonl(self, files: Union[str|List[str]], N=-1, num_workers=1, **kwargs):
        kwargs = dict(
            files = files, 
            N=N,
            num_workers=num_workers,
        ) | kwargs
        with adhoc.from_kwargs(**kwargs) as aargs:
            files = list_filenames(aargs['files|!!'])
            N = aargs['head|N|=-1']
            num_workers = aargs['num_workers|=1']
            output_file = aargs['output_file|output_path|output']
            adhoc.notice('フィルタ', input_files=files, filter_config=self.as_json())
            if output_file is None:
                result = self._from_jsonl_single(files, N=100, output_file=output_file)
                adhoc.notice(f"先頭100件だけフィルタしたよ! output_file='file.jsonl'で最後まで保存できるからね")
                return result
            prefix = aargs['prefix|=']
            output_file = f'{prefix}{output_file}'
            with adhoc.start_timer() as timer:
                if num_workers == 1:
                    result = self._from_jsonl_single(files, N=N, output_file=output_file)
                else:
                    result = self._from_jsonl_multi(files, N=N, output_file=output_file, num_workers=num_workers)
                timer.notice('フィルタ、無事完了。お疲れ様', **result)
                if aargs['with_linenum|=True']:
                    output_file = rename_linenum(output_file, N=result['remaining'], rename=True)
                adhoc.saved(output_file, '前処理済みファイル//pre-processed file')
                result['output_file'] = output_file
            return result

    def _from_jsonl_single(self, filenames: str, N=-1, output_file=None):
        w = None
        if isinstance(output_file, str):
            w = zopen(output_file, 'wt')
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
                    print(record)
        return dict(total=n, remaining=c)

    def invoke_as_multi(self, text):
        return self(text, _dummy_record)

    def _from_jsonl_multi(self, filenames: str, output_file:str=None, N=-1, num_workers=1):
        filenames = list_filenames(filenames)
        c=0
        n=0
        with zopen(output_file, 'wt') as w, Pool(num_workers) as pool:
            for lines in read_multilines(filenames, N=N, bufsize=1000 * num_workers, line_reader='jsonl'):
                lines = pool.map(self.invoke_as_multi, lines)
                n += len(lines)
                for text in lines:
                    if text:
                        c+=1
                        print(json.dumps({'text': text}, ensure_ascii=False), file=w)
        return dict(total=n, remaining=c)

    def run_for_cli(self, **kwargs):
        with adhoc.open_log_file('.', 'filter_log.txt') as log:
            result = self.from_jsonl(**kwargs)
            self.describe()
            output_file = result.pop('output_file')
            if output_file:
                adhoc.saved(f'{output_file}_log.txt', 'ログ',
                            rename_from='filter_log.txt')
            adhoc.report_saved_files()

class ComposeFilter(TextFilter):
    """
    テキストフィルタを合成する
    :param filters:
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = tuple(args)

    def as_json(self):
        cls = self.__class__
        return {
            'class_path' : f'{cls.__module__}.{cls.__name__}',
            'args': [f.as_json() for f in self.filters]
        }

    def __call__(self, text: str, record: dict) -> Optional[str]:
        for filter in self.filters:
            text = filter(text, record)
            if text is None:
                return None
        return text

    def describe(self, file=sys.stdout):
        for filter in self.filters:
            filter.describe(file)

def generate_filter(expression):
    if isinstance(expression, TextFilter):
        return expression
    if isinstance(expression, dict):
        return adhoc.instantiate_from_dict(expression, check=TextFilter)
    return TextFilter(unknown_expression=f'{expression}')

def compose(*filters):
    if len(filters) == 1:
        return generate_filter(filters[0])
    return ComposeFilter(*(generate_filter(e) for e in filters))
 
class ChoiceFilter(ComposeFilter):
    """
    テキストフィルタを合成する
    :param filters:
    """
    def __init__(self, args: List[TextFilter]):
        super().__init__()
        self.filters = tuple(args)

    def __call__(self, text: str, record: dict) -> Optional[str]:
        for f in self.filters:
            text2 = f(text, record)
            if text2 is not None:
                return text2
        return None

def choice(*filters):
    if len(filters) == 1:
        return generate_filter(filters[0])
    return ChoiceFilter(*(generate_filter(e) for e in filters))

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

def compile_pattern_for_words(words: List[str], prefix='', suffix=''):
    """
    Given a list of words or a single string of words separated by '|', compiles and returns a regular expression pattern that matches any of the words. Additionally, if the words list contains filenames ending in '.txt', the function reads these files and includes their contents as words. The function removes duplicates and sorts the words before compiling the pattern.

    If `prefix` or `suffix` strings are provided, they are added to the beginning and end of the compiled pattern, respectively.

    Parameters:
    - words (List[str] or str): A list of words, or a single string of words separated by '|'. Can also include filenames with '.txt' extension, whose contents will be read and included as words.
    - prefix (str, optional): A string to be added to the beginning of the compiled pattern. Defaults to an empty string.
    - suffix (str, optional): A string to be added to the end of the compiled pattern. Defaults to an empty string.

    Returns:
    - re.Pattern: A compiled regular expression pattern that matches any of the specified words, optionally enclosed between `prefix` and `suffix`.

    Note:
    - The function ensures that duplicates are removed and the final list of words is sorted before compiling the pattern.
    - If a filename is provided in the `words` list and it does not exist or cannot be read, it is ignored.
    """
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


