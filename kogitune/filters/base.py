from typing import Optional
import json
# import os
# from kogitune.adhocargs import adhoc_parse_arguments
from kogitune.file_utils import zopen, filelines, read_multilines, rename_with_linenum

from tqdm import tqdm
from multiprocess import Pool

class TextFilter(object):
    """
    テキストフィルターの規定クラス
    """
    def __init__(self, verbose=0):
        """
        新しいテキストフィルタを作る

        :param verbose: 指定した個数だけデバック出力する
        """
        self.verbose = verbose
        self.record = None

    def __call__(self, text: str) -> Optional[str]:
        if self.verbose > 0:
            # 指定した個数だけデバック出力する
            filtered_text = self.filter(text)
            self.debug_print(f'[{self.verbose}] {repr(self)}')
            print('|'+text.replace('\n', '\n|'))
            print('==>')
            print(filtered_text)
            self.verbose -= 1
            return filtered_text
        return self.filter(text)

    def set_record(self, record):
        self.record = record

    def filter(self, text: str)-> Optional[str]:
        return text

    def debug_print(self, *args):
        if self.verbose > 0:
            print('🦊', *args)
            self.verbose -= 1

    def from_jsonl(self, filename: str, output_path:str=None, N=-1, num_workers=1):
        if num_workers == 1 or output_path is None:
            return self._from_jsonl_single(filename, N=N, output_path=output_path)
        c=0
        n=0
        with zopen(output_path, 'wt') as w:
            with Pool(num_workers) as pool:
                for lines in read_multilines(filename, N=N, bufsize=10000 * num_workers, tqdm=tqdm):
                    lines = pool.map(self, lines)
                    n += len(lines)
                    for text in lines:
                        if text:
                            c+=1
                            print(json.dumps({'text': text}, ensure_ascii=False), file=w)
        newpath = rename_with_linenum(output_path, N=c, ext='json')
        print(f'Complete: {newpath} {c}/{n} {c/n:.3f}')

    def _from_jsonl_single(self, filename: str, N=-1, output_path=None):
        w = None
        if isinstance(output_path, str):
            w = zopen(output_path, 'wt')
        else:
            self.verbose = 10
        c=0
        n=0
        for lines in read_multilines(filename, N=N, tqdm=tqdm):
            for text in lines:
                record = {}
                self.set_record(record) # レコーダをセットする
                text = self(text)
                n+=1
                if text:
                    record['text'] = text
                    c+=1
                    if w:
                        print(json.dumps(record, ensure_ascii=False), file=w)
                    else:
                        self.debug_print(record)
        if output_path:
            newpath = rename_with_linenum(output_path, N=c, ext='json')
            print(f'Complete: {newpath} {c}/{n} {c/n:.3f}')

    # def run_as_main(self):
    #     args = adhoc_argument_parser()
    #     output_path = args['output_path']
    #     num_workers = args['num_workers|=1']
    #     N = args['N|=-1']
    #     for file in args.files:
    #         self.from_jsonl(file, output_path=output_path, N=N, num_workers=num_workers)


class ComposeFilter(TextFilter):
    """
    テキストフィルタを合成する
    """
    def __init__(self, *filters):
        super().__init__(verbose=0)
        self.filters = filters

    def __call__(self, text):
        for f in self.filters:
            text = f(text)
            if text is None:
                return None
        return text

    def set_record(self, record):
        self.record = record
        for f in self.filters:
            if isinstance(f, TextFilter):
                f.set_record(record)


class ChoiceFilter(TextFilter):
    def __init__(self, *filters):
        super().__init__(verbose=0)

    def __call__(self, text):
        for f in self.filters:
            text2 = f(text)
            if text2 is not None:
                return text2
        return None

    def set_record(self, record):
        self.record = record
        for f in self.filters:
            if isinstance(f, TextFilter):
                f.set_record(record)



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



