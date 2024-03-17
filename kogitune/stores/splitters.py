from typing import List, Union, Tuple
import numpy as np
import pandas as pd
from multiprocessing import Pool

from transformers import AutoTokenizer

from ..adhoc_args import configurable_tokenizer, adhoc_log
from ..tokenizers import *
from ..commons import *
from ..utils_file import *
from .store import DatasetStore
from .section import find_section_fn

empty_tokens = []

class TextBlockSpliter(object):
    def __init__(self, tokenizer, aargs):
        self.tokenizer = tokenizer
        self.blocks = []
        self.extra_tokens=empty_tokens
        self.block_size = aargs.get('max_length|block_size', 2048)
        self.section = 'elastic'
        self.trancate_size = aargs.get('trancate|trancate_size', self.block_size // 8)
        self.padding_size = aargs.get('padding|padding_size', self.block_size // 8)
        self.max_tokens = aargs['max_tokens']
        self.min_tokens = aargs['min_tokens|=2']
        self.record = Recorder(aargs)
    
    def as_json(self):
        return dict(
            class_name = self.__class__.__name__,
            section = 'pack' if self.trancate_size == 0 and self.padding_size == 0 else 'none',
            block_size = self.block_size,
            trancate_size = self.trancate_size,
            padding_size = self.padding_size,
            max_tokens = self.max_tokens,
            min_tokens = self.min_tokens,
        )

    def encode(self, text, eos=True):
        if eos:
            tokens = self.tokenizer.encode(text)
            tokens_length = len(tokens)-1
        else:
            tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            tokens_length = len(tokens)
        if self.min_tokens is not None and tokens_length <= max(self.min_tokens, 2):
            self.record.trancated(tokens_length)
            return []
        if self.record.tokenized_and_filtered(len(text), tokens_length):
            return []
        if self.max_tokens is not None and tokens_length > self.max_tokens:
            self.record.trancated(max(tokens_length-self.block_size, 0))
            tokens = tokens[:self.block_size]
        return tokens

    def pad(self, length):
        if length == 0:
            return []
        pad_id = self.tokenizer.eos_token_id
        self.record.padded(length)
        if length == 1:
            return [pad_id]
        # 予想しやすいpad作る
        return [self.tokenizer.vocab_size-length] + [pad_id]*(length-1)

    def append_block(self, tokens:List[int]):
        assert len(tokens) == self.block_size
        self.blocks.append(tokens) #np.array(tokens, dtype=np.int32))
        self.record.blocked()

    def try_block(self, tokens:List[int], unused_tokens:List[int]=empty_tokens) -> List[int]:
        if len(unused_tokens) == 0:
            self.record.count_head()
        else:
            tokens = unused_tokens + tokens
        work_size = self.block_size
        if len(tokens) < work_size:
            return tokens
        for i in range(0, len(tokens) - work_size + 1, work_size):  
            segmented = tokens[i : i + work_size]
            self.append_block(segmented)
        unused_size = len(tokens) % work_size
        return empty_tokens if unused_size == 0 else tokens[-unused_size:]
    
    def try_trancate(self, extra_tokens):
        if len(extra_tokens) <= self.trancate_size:
            self.record.trancated(len(extra_tokens))
            return empty_tokens
        return extra_tokens
    
    def append_text(self, text):
        tokens = self.encode(text)
        extra_tokens = self.try_block(tokens, unused_tokens=self.extra_tokens)
        unused_size = self.block_size - len(extra_tokens)
        if 0 < unused_size <= self.padding_size:
            ## パッディングする
            self.append_block(extra_tokens+self.pad(unused_size))
            self.extra_tokens = empty_tokens
        else:
            self.extra_tokens = self.try_trancate(extra_tokens)

    def __call__(self, args: Union[dict, List[str]]):
        self.blocks = []
        if isinstance(args, dict):
            self.record = args['record']
            self.extra_tokens = args['extra_tokens']
            for text in args['docs']:
                self.append_text(text)
            args['docs'] = None
            args['record'] = self.record
            args['extra_tokens'] = self.extra_tokens
            args['blocks'] = self.blocks
            return args
        for text in args:
            self.append_text(text)
        return self.blocks

    def report_to(self, logs: dict):
        logs['class'] = self.__class__.__name__
        for attr in dir(self):
            v = getattr(self, attr)
            if isinstance(v, (int, float, str, bool)):
                logs[attr] = v
        logs['record'] = {}
        self.record.report_to(logs['record'])

class TextPacker(TextBlockSpliter):
    def __init__(self, tokenizer, aargs):
        super().__init__(tokenizer, aargs)
        self.trancate_size = 0
        self.padding_size = 0

class SectionSplitter(TextBlockSpliter):
    def __init__(self, tokenizer, section_fn, aargs):
        super().__init__(tokenizer, aargs)
        self.section = aargs.get('section', 'doc')
        self.section_fn = section_fn
        self.overlap_factor = aargs.get('overlap_factor|overlap', 0.25)

    def as_json(self):
        return dict(
            class_name = self.__class__.__name__,
            section=self.section,
            block_size = self.block_size,
            trancate_size = self.trancate_size,
            padding_size = self.padding_size,
            overlap_factor = self.overlap_factor,
            max_tokens = self.max_tokens,
            min_tokens = self.min_tokens,
        )

    def append_text(self, text:str):
        sections = self.section_fn(text)
        block_size = self.block_size
        extra = self.extra_tokens
        for i, sec in enumerate(sections):
            tokens = self.encode(sec, eos = (i == len(sections)-1))
            if len(extra) + len(tokens) < block_size:
                extra = extra + tokens
                continue
            unused_size = block_size - len(extra)
            if (unused_size / block_size) < self.overlap_factor:
                self.append_block(extra+tokens[:unused_size])
                self.record.overlapped(unused_size)
                extra = self.try_block(tokens)
            else:
                extra = self.try_block(tokens, unused_tokens=extra)
            extra = self.try_trancate(extra)
        self.extra_tokens = extra

def find_splitter(tokenizer, aargs):
    data_type = aargs['datatype|data_type|=text']
    if data_type == 'text':
        section = aargs['section|=pack']
        if section is None or section == 'none':
            return TextBlockSpliter(tokenizer, aargs)
        if section == 'pack':
            return TextPacker(tokenizer, aargs)
        section_fn = find_section_fn(section)
        if section_fn is None:
            return TextBlockSpliter(tokenizer, aargs)
        return SectionSplitter(tokenizer, section_fn, aargs)
    else: # ファインチューニング用
        # splitter = SimpleTextSplitter(tokenizer, args)
        raise NotImplementedError(f'datatype={data_type}')

class Recorder(object):
    def __init__(self, args: AdhocArguments, rank=0):
        self.rank = rank
        self.head_count = 0
        self.block_count = 0
        self.trancated_size = 0
        self.filtered_size = 0
        self.padding_size = 0
        self.overlapped_size = 0
        self.chars_size = 0
        self.tokens_size = 0
        self.cpt_counts = None
        self.cpt_q = int(100 * 0.02)
        self.max_cpt = args['max_cpt']
        self.min_cpt = args['min_cpt']
        self._init_counters()

    def _init_counters(self):
        if self.max_cpt is None and self.min_cpt is None and self.cpt_counts is None:
            self.cpt_counts = []
        return self

    def _update_maxmin(self):
        if self.cpt_counts is not None and len(self.cpt_counts) > 0:
            self.max_cpt = np.percentile(np.array(self.cpt_counts), 100 - self.cpt_q)
            self.min_cpt = np.percentile(np.array(self.cpt_counts), self.cpt_q)

    def tokenized_and_filtered(self, chars_length, tokens_length)-> bool:
        """
        トークンナイザーの圧縮率が極端な外れ値の場合はフィルターする
        """
        self.chars_size += chars_length
        self.tokens_size += tokens_length
        chars_per_tokens = chars_length / tokens_length
        if self.cpt_counts is not None:
            self.cpt_counts.append(chars_per_tokens)
            if len(self.cpt_counts) % 1000 == 999:
                self._update_maxmin()
            if len(self.cpt_counts) == 5000:
                self._update_maxmin()
                self.cpt_counts = None
        if self.max_cpt is not None and chars_per_tokens > self.max_cpt:
            self.filtered(tokens_length)
            return True
        if self.min_cpt is not None and chars_per_tokens < self.min_cpt:
            self.filtered(tokens_length)
            return True
        return False

    def count_head(self):
        self.head_count += 1

    def blocked(self):
        self.block_count += 1

    def trancated(self, size):
        self.trancated_size += size

    def filtered(self, size):
        self.filtered_size += size

    def padded(self, size):
        self.padding_size += size

    def overlapped(self, size):
        self.overlapped_size += size

    def as_json(self, merge={}):
        logs = {}
        logs['source_chars'] = self.chars_size + merge.get('source_chars', 0)
        logs['source_tokens'] = self.tokens_size + merge.get('source_tokens', 0)
        logs['trancated'] = self.trancated_size + merge.get('trancated', 0)
        logs['filtered'] = self.filtered_size + merge.get('padding', 0)
        logs['block'] = self.block_count + merge.get('block', 0)
        logs['head'] = self.head_count + merge.get('head', 0)
        logs['padding'] = self.padding_size + merge.get('padding', 0)
        logs['overlapped'] = self.overlapped_size + merge.get('overlapped', 0)
        if self.max_cpt:
            logs['max_cpt'] = max(self.max_cpt, merge.get('max_cpt', 0))
            logs['min_cpt'] = min(self.min_cpt, merge.get('min_cpt',100))
        return logs

def report_record(logs, block_size):
    chars = logs['source_chars']
    tokens = logs['source_tokens']
    filtered = int(logs['filtered'])
    trancated =int(logs['trancated'])
    block = logs['block']
    head = logs['head']
    block_tokens = block * block_size
    padding = logs['padding'] 
    overlapped = logs['overlapped'] 
    max_cpt = logs.get('max_cpt',100)
    min_cpt = logs.get('min_cpt', 0)
    verbose_print(f'訓練データセットの結果')
    print(f'文字数//chars {format_unit(chars, scale=1000)} トークン数//tokens {format_unit(tokens, scale=1000)} 適合率(cpt) {chars/tokens:.4f}')
    print(f'フィルタ//filtered {format_unit(filtered, scale=1000)}トークン {filtered*100/tokens:.2f}% 区間 [{min_cpt:.3f}, {max_cpt:.3f}]')
    print(f'切り詰め//trancated {format_unit(trancated, scale=1000)}トークン {trancated*100/tokens:.2f}% ')
    print(f'有効トークン数 {block_tokens}/{tokens} {block_tokens*100/tokens:.2f}% ')
    print(f'先頭ブロック/section {head}/{block} {head*100/block:.2f}% ブロック長 {block_size} x {block} = {block_size*block}')
    print(f'パディング//padding {padding}/{block_tokens} {padding*100/block_tokens:.2f}%')
    print(f'オーバーラップ//overlap {overlapped}/{block_tokens} {overlapped*100/block_tokens:.2f}%')

## Store 用

def get_store_path(filenames, tokenizer, aargs):
    store_path = aargs['store_path|store_dir|store']
    if store_path is None:
        filebase = get_filebase(filenames[0])
        tokenizer_name = tokenizer_id(tokenizer)
        store_path=f'{tokenizer_name}/{filebase}'
        return store_path
    else:
        if '/' in store_path:
            filebase = store_path.replace('/', '_')
        else:
            filebase = store_path
            tokenizer_name = tokenizer_id(tokenizer)
            store_path=f'{tokenizer_name}/{filebase}'
            aargs['store_path'] = store_path
            verbose_print(f'保存先/Saving To.. {store_path}')

def store_files(filenames: List[str], tokenizer=None, **kwargs):
    """
    ファイルからローカルストアを構築する
    :param filenames: ファイル名、もしくはファイル名のリスト
    """
    filenames = list_filenames(filenames)
    adhoc_log('store', 'input_files', filenames)
    with AdhocArguments.from_main(**kwargs) as aargs:
        tokenizer = configurable_tokenizer(tokenizer=tokenizer)
        splitter = find_splitter(tokenizer, aargs)
        adhoc_log('store', 'splitter', splitter.as_json(), message='確認してよ')
        adhoc_log('store', 'tokenizer', tokenizer_as_json(tokenizer))

        store_path = get_store_path(filenames, tokenizer, aargs)
        adhoc_log('store', 'store_path', store_path, message='保存先//Saving To..')
 
        store = DatasetStore(store_path, aargs=aargs)

        num_workers = aargs['num_workers|=1']
        N=aargs['head|N|=-1']
        if num_workers == 1:
            for docs in read_multilines(filenames, N=N, bufsize=1024):
                blocks = splitter(docs)
                store.append(blocks)
            record_logs = splitter.record.as_json()
        else:
            pool = Pool(num_workers)
            func_args = []
            for i in range(num_workers):
                func_args.append({
                    'record': Recorder(aargs, rank=i),
                    'extra_tokens': empty_tokens,
                    'docs': None,
                    'blocks': None,
                })
            for batch in read_multilines(filenames, N=N, bufsize=1024 * num_workers):
                batch_size = len(batch) // num_workers
                for i in range(num_workers):
                    func_args[i]['docs'] = batch[batch_size*i:batch_size*(i+1)]
                    func_args[i]['blocks'] = None
                func_args = pool.map(splitter, func_args)
                for i in range(num_workers):
                    store.append(func_args[i]['blocks'])
            pool.close()
            record_logs={}
            for a in func_args:
                a['record'].as_json(merge=record_logs)
        report_record(record_logs, splitter.block_size)
        adhoc_log('store', 'result', record_logs)
        store.save(tokenizer, skip_validation=aargs['skip_validation'])
    return str(os.path.abspath(store_path))


# def make_local_store(filename:str, tokenizer, args:dict):
#     if 'cache_dir' in args and 'store_path' not in args:
#         filebase = get_filebase(filename)
#         args['store_path'] = safe_join_path(args['cache_dir'], filebase)
#     args['tokenizer'] = tokenizer
#     configurable_store(filename, args=args)
#     return str(os.path.abspath(args['store_path']))
