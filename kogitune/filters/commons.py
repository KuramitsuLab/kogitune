from typing import Optional, List, Union, Any
import json
import os
import re

import kogitune.adhocs as adhoc

from kogitune.stores.files import (
    zopen, filelines, 
    read_multilines, 
    rename_with_linenum, 
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

def as_json(v):
    if hasattr(v, 'as_json'):
        return v.as_json()
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [as_json(x) for x in v]
    if isinstance(v, dict):
        return {key:as_json(value) for key,value in v.items()}
    return None

def instantiate_json(v, namespace:dict):
    if isinstance(v, dict) and 'class_name' in v:
        class_name = v.pop()
        if class_name in namespace:
            return namespace[class_name](**v)
    if isinstance(v, (list, tuple)):
        return [instantiate_json(x, namespace) for x in v]
    return v


class TextFilter(object):
    """
    テキストフィルターの規定クラス
    """
    def __init__(self, **kwargs):
        """
        新しいテキストフィルタを作る
        """
        self._kwargs = kwargs
        self.rec = {}

    def setups(self, kwargs, *keys):
        for key in keys:
            key, value = adhoc.get_key_value(kwargs, key)
            self.rec[key] = value
            if not hasattr(self, key):
                setattr(self, key, value)

    def name(self):
        return self.__class__.__name__

    def as_json(self):
        json_data = {
            'class_name': self.__class__.__name__,
        }
        for key, value in self.__dict__.items():
            if not key.startswith('_') and is_json_valuable(value):
                json_data[key] = as_json(value)
        return json_data

    def __call__(self, text: str, record: dict) -> Optional[str]:
        return text

    def __repr__(self):
        return json.dumps(self.as_json(), indent=2)

    def save_as_json(self, filename: str):
        with open(filename, 'w') as w:
            json.dump(self.as_json(), w, indent=2)

    def to_report(self, section):
        pass

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
    
    def from_jsonl(self, filenames: str, output_path:str=None, **kwargs):
        filenames = list_filenames(filenames)
        with adhoc.from_kwargs(**kwargs) as aargs:
            N = aargs['head|N|=-1']
            num_workers = aargs['num_workers|=1']
            adhoc.notice('フィルタ', input_files=filenames, filter_config=self.as_json())
            adhoc.start_time('filter')
            if num_workers == 1 or output_path is None:
                result = self._from_jsonl_single(filenames, N=N, output_path=output_path)
            else:
                result = self._from_jsonl_multi(filenames, output_path=output_path, N=N, num_workers=num_workers)
            # adhoc.notice('フィルタ、無事完了。お疲れ様', **result)
            adhoc.end_time('filter', message='フィルタ、無事完了。お疲れ様', total=result.pop('total'), **result)

    def _from_jsonl_single(self, filenames: str, N=-1, output_path=None):
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
        result = dict(total=n, filtered=c)
        if output_path:
            newpath = rename_with_linenum(output_path, N=c, ext='json')
            result['output_file'] = newpath
        return result

    def invoke_as_multi(self, text):
        return self(text, _dummy_record)

    def _from_jsonl_multi(self, filenames: str, output_path:str=None, N=-1, num_workers=1):
        filenames = list_filenames(filenames)
        #adhoc.setlog('filter', input_files=filenames, filter_config =self.as_json())
        c=0
        n=0
        with zopen(output_path, 'wt') as w, Pool(num_workers) as pool:
            for lines in read_multilines(filenames, N=N, bufsize=1000 * num_workers, line_reader='jsonl'):
                lines = pool.map(self.invoke_as_multi, lines)
                n += len(lines)
                for text in lines:
                    if text:
                        c+=1
                        print(json.dumps({'text': text}, ensure_ascii=False), file=w)
        newpath = rename_with_linenum(output_path, N=c, ext='json')
        adhoc.notice(f'完了//Complete: {newpath} {c}/{n} {c/n:.3f}', 
                     output_file = newpath, total=n, filtered=c)
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

class ChoiceFilter(TextFilter):
    """
    テキストフィルタを合成する
    :param filters:
    """
    def __init__(self, *filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = tuple(filters)

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
        self.rec = {'class': self.name()}
        if len(kwargs) > 0:
            adhoc.print(f'Unused [{self.name()}]', kwargs)

    def __call__(self, text: str):
        return len(text)

    def name(self):
        return self.__class__.__name__

    def get_value(self, key, **kwargs):
        return adhoc.get_value(key, self.rec, kwargs)

    def as_json(self):
        return self.rec

    def __repr__(self):
        return json.dumps(self.as_json(), indent=2)

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


