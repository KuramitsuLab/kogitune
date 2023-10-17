from typing import List
import random
import json
import numpy as np
import pandas as pd

import hashlib
from transformers import AutoTokenizer

from .tokenizers import load_tokenizer, find_ellipsis_token_id, find_newline_token_id, find_token_id
from .commons import *
from .file_utils import *

DEFAULT_BLOCK_SIZE=2048

def _record_tokens(counts):
    if len(counts) == 0:
        return {'total': 0}
    data = np.array(counts)
    return {
        'total': int(np.sum(data)),
        'mean': float(np.mean(data)),
        'std': float(np.var(data)) ** 0.5,
        'max': int(np.max(data)),
        '75%': int(np.percentile(data, 75)),
        'median': int(np.median(data)),
        '25%': int(np.percentile(data, 25)),
        'min': int(np.min(data)),
    }

def _update_fn(blocks: List[List[int]]):
    return blocks


empty_tokens = []


class DefaultSplitter(object):

    def __init__(self, tokenizer, block_size, **kwargs):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split_args = kwargs
        self.pad_token_id = getint(kwargs, 'pad_token_id', tokenizer.pad_token_id)
        self.eos_token_id = getint(kwargs, 'eos_token_id', tokenizer.eos_token_id)
        self.ellipsis_token_id = find_ellipsis_token_id(tokenizer)
        # self.output_token_id = None # <sftの場合>
        self.trancate_size=getint(kwargs, 'trancate_size', 0)
        self.sep = kwargs.get('sep', None)
        self.token_counts = []
        self.text_length_count = 0
        self.trimmed_size = 0
        self.padded_size = 0
    
    def about(self):
        return dict(name=self.__class__.__name__)

    def split(self, text:str, blocks: List[List[int]]):
        raise NotImplemented()

    def flush(self, blocks: List[List[int]]):
        pass

    def split_iter(self, iterator, update_fn=_update_fn):
        blocks = []
        for text in iterator:
            self.split(text, blocks)
            blocks = update_fn(blocks)
        self.flush(blocks)
        blocks = update_fn(blocks)
        return blocks

    def encode_and_count(self, text):
        self.text_length_count += len(text)
        tokens = self.tokenizer.encode(text)
        self.token_counts.append(len(tokens)-1)
        return tokens

    def resize_token_counts(self, size):
        self.token_counts[-1] += size

    def report(self, logs: dict = None, verbose=True):
        token_count = sum(self.token_counts)
        if logs:
            logs['n_tokens'] = token_count
            if self.text_length_count > 0:
                logs['n_chars'] = self.text_length_count
                logs['tokens_per_char'] = token_count / self.text_length_count
            logs['tokens'] = _record_tokens(self.token_counts)
        if verbose:
            print(pd.DataFrame({'tokens': self.token_counts}).describe())
        if self.padded_size > 0:
            if verbose:
                print(f'padding: {self.padded_size:,} {self.padded_size*100/token_count:.2f}%')
            if logs:
                logs['padding_rate'] = self.padded_size / token_count


class SimpleTextBlockSplitter(DefaultSplitter):
    def __init__(self, tokenizer, block_size, **kwargs):
        super().__init__(tokenizer, block_size, **kwargs)
        self.extra_tokens=empty_tokens
        self.prefix='pre'

    def split(self, text:str, blocks: List[List[int]]):
        tokens = self.encode_and_count(text)
        work_size = self.block_size
        tokens = self.extra_tokens + tokens
        for i in range(0, len(tokens) - work_size + 1, work_size):  
            segmented = tokens[i : i + work_size]
            blocks.append(segmented)
        extra_size = len(tokens) % work_size
        if extra_size == 0:
            self.extra_tokens = empty_tokens
        else:
            self.extra_tokens = tokens[-extra_size:]

    def report(self, logs: dict = None, verbose=True):
        super().report(logs, verbose=verbose)
        if logs:
            logs['block_size'] = self.block_size


class MultiTextBlockSplitter(DefaultSplitter):
    def __init__(self, tokenizer, block_size, **kwargs):
        super().__init__(tokenizer, block_size, **kwargs)
        self.prefix='pre'
        self.work_size = getint(kwargs, 'work_size', 512)
        self.pad_size = getint(kwargs, 'pad_size', 64)
        # 警告が出る場合があるので、padding を変更する
        self.pad_token_id = find_newline_token_id(tokenizer)
        self.extra_tokens = empty_tokens
        self.work_buffers = []
        self.lazy_buffers = []
        self.heads = [[] for _ in range(self.work_size//self.pad_size)]
        self.match_count = 0
        self.sep_counts = []

    def resize_as_dividable(self, tokens, size, trancate_size=0, padding_size=None):
        extra_size = len(tokens) % size 
        if extra_size == 0:
            return tokens
        if extra_size <= trancate_size:
            tokens = tokens[:-extra_size]
            if len(tokens)>2:
                tokens[-1] = self.eos_token_id
                if self.ellipsis_token_id:
                    tokens[-2] = self.ellipsis_token_id
            return tokens
        if padding_size is None or (size - extra_size) < padding_size:
            self.padded_size += (size - extra_size)
            return tokens + [self.pad_token_id] * (size - extra_size)
        return tokens[:(len(tokens)//size)*len(tokens)]

    def tokenize_nonsep(self, text:str):
        tokens = self.tokenizer.encode(text)
        self.token_counts.append(len(tokens))
        tokens = self.resize_as_dividable(tokens, self.pad_size, trancate_size=self.trancate_size)
        return tokens

    def tokenize_sep(self, text:str):
        text_blocks = text.split(self.sep)
        tokenizer = self.tokenizer
        chunks = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line)) for line in text_blocks]
        chunks[-1] = tokenizer.build_inputs_with_special_tokens(chunks[-1])
        counts = [len(c) for c in chunks]
        self.token_counts.append(sum(counts))
        self.sep_counts.extend(counts)
        tokens = []
        chunk_bufs = []
        split_size = self.pad_size * 2
        for chunk in chunks:
            prev_len = len(chunk_bufs)
            chunk_bufs.extend(chunk)
            if len(chunk_bufs) < split_size:
                continue
            if len(chunk_bufs) % self.pad_size == 0:
                tokens.extend(chunk_bufs)
                chunk_bufs=[]
            else:
                splitted = self.resize_as_dividable(chunk_bufs, self.pad_size, trancate_size=self.pad_size, padding_size=0)
                tokens.extend(splitted)
                if prev_len > 0:
                    chunk_bufs = chunk
                else:
                    chunk_bufs = []
        return self.resize_as_dividable(tokens+chunk_bufs, self.pad_size, self.trancate_size)

    def add_buffer(self, blocks: List[List[int]], tokens: List[int]):
        assert(len(tokens) == self.work_size)
        if len(self.extra_tokens) > 0:
            self.lazy_buffers.append(tokens)
        elif len(self.lazy_buffers) > 0:
            lazy_buffers = self.lazy_buffers
            self.lazy_buffers = []
            for lazy_tokens in lazy_buffers:
                self.add_buffer(blocks, lazy_tokens)

        self.work_buffers.extend(tokens)
        if len(self.work_buffers) == self.block_size:
            blocks.append(self.work_buffers)
            self.work_buffers = []

    def push_head(self, tokens):
        index = len(tokens) // self.pad_size
        self.heads[index].append(tokens)

    def pop_head(self, extra_size):
        index = extra_size // self.pad_size
        if len(self.heads[index]) > 0:
            return self.heads[index].pop()
        return None

    def split(self, text:str, blocks: List[List[int]]):
        if self.sep is not None and self.sep in text:
            tokens = self.tokenize_sep(text)
        else:
            tokens = self.tokenize_nonsep(text)

        work_size = self.work_size
        if len(tokens) < work_size:
            head = self.pop_head(work_size - len(tokens) % work_size)
            if head:
                self.add_buffer(blocks, head+tokens)
            else:
                self.push_head(tokens)
            return
        tokens = self.extra_tokens + tokens
        for i in range(0, len(tokens) - work_size + 1, work_size):  
            segmented = tokens[i : i + work_size]
            self.add_buffer(blocks, segmented)

        extra_size = len(tokens) % work_size
        if extra_size == 0: # 最後の分割が揃っていればおしまい
            self.extra_tokens = empty_tokens
            return
        extra_tokens = tokens[-extra_size:]
        head = self.pop_head(work_size - extra_size)
        if head:
            self.add_buffer(blocks, extra_tokens+head)
            self.extra_tokens = empty_tokens
        else:
            self.extra_tokens = extra_tokens

    def flush(self, blocks: List[List[int]]):
        heads = []
        for hh in self.heads:
            heads.extend(hh)
        # print(pd.DataFrame({'heads': [len(h) for h in heads]}).describe())
        random.shuffle(heads)
        tokens = []
        for t in [self.extra_tokens]+heads:
            tokens.append(t)
        work_size = self.work_size
        for i in range(0, len(tokens) - work_size + 1, work_size):  
            segmented = tokens[i : i + work_size]
            self.add_buffer(blocks, segmented)

    def report(self, logs: dict = None, verbose=True):
        super().report(logs, verbose=verbose)
        if logs:
            logs['block_size'] = self.block_size
            logs['work_size'] = self.work_size

class TextPairSplitter(DefaultSplitter):
    def __init__(self, tokenizer, block_size, **kwargs):
        super().__init__(tokenizer, block_size, **kwargs)
        self.prefix=''
        self.output_sep_token_id = find_token_id(tokenizer, 
                                                 kwargs.get('output_sep', '<outpuT>'), 
                                                 kwargs.get('sep', '<seP>'), 
                                                 '<nL>')

    def report(self, logs: dict = None, verbose=True):
        super().report(logs, verbose=verbose)
        if self.output_sep_token_id is not None:
            logs['output_sep_token_id'] = self.output_sep_token_id            
        if self.sep is not None:
            logs['corpus_sep'] = self.sep            

    def trancate_pair(self, inputs: List[int], labels: List[int], blocks: List[List[int]]):
        if self.block_size is not None:
            if len(labels) > self.block_size:
                # ラベルの方が大きい場合は諦める
                return
            if len(inputs)+len(labels) > self.block_size:
                trimmed_size = self.block_size - len(labels)
                if len(inputs) - trimmed_size > self.trancate_size:
                    # 諦める                
                    return
                self.trimmed_size += len(inputs) - trimmed_size
                inputs = inputs[:trimmed_size]
            else:
                self.padded_size += self.block_size - (len(inputs)+len(labels))
        index = len(inputs)
        inputs[-1] = self.output_sep_token_id
        blocks.append(inputs+labels+[index])

    def split(self, text:str, blocks: List[List[int]]):
        t = text.split(self.sep)
        if len(t)==2:
            inputs = self.encode_and_count(t[0])        
            labels = self.encode_and_count(t[1])
            self.trancate_pair(inputs, labels, blocks)
        else:
            print(self.sep, text)
            raise ValueError(f'In text, the {self.sep} token is required.')

class SimpleTextSplitter(TextPairSplitter):
    def __init__(self, tokenizer, block_size, **kwargs):
        super().__init__(tokenizer, block_size, **kwargs)
        # セパレータ字句があるか調べる
        # self.sep: コーパス上のセパレータ
        # self.output_sep トークンナイザのセパレータ
        self.output_sep_token_id = None
        self.output_sep = str(kwargs.get('output_sep', self.sep))
        if self.output_sep:
            self.output_sep_token_id = find_token_id(tokenizer, self.output_sep)
            if self.output_sep_token_id == tokenizer.unk_token_id:
                verbose_print(f'undefined token {self.output_sep} in {tokenizer.name_or_path}')
                self.output_sep_token_id = None            

    def trancate_text(self, inputs: List[int], labels: List[int], blocks: List[List[int]]):
        if self.block_size is not None:
            inputs_size = len(inputs)
            if inputs_size > self.block_size:
                if inputs_size - self.block_size > self.trancate_size:
                    # 切り詰めが大きすぎる
                    return
                half_size = self.block_size // 2
                prefix = inputs[:half_size]
                suffix = inputs[-half_size:]
                if self.ellipsis_token_id:
                    prefix[-1] = self.ellipsis_token_id
                inputs = prefix + suffix
                self.trimmed_size += (inputs_size - len(inputs))
            else:
                self.padded_size += self.block_size - inputs_size
        blocks.append(inputs)

    def split(self, text:str, blocks: List[List[int]]):
        if self.sep:
            text = text.replace(self.sep, self.output_sep, 1)
        inputs = self.encode_and_count(text)
        if self.output_sep_token_id is not None:
            index = inputs.find(self.output_sep_token_id)
            if index == -1:
                sep = self.sep or self.output_sep
                raise ValueError(f'{sep} が見つかりません')
            self.trancate_pair(inputs[:index+1], inputs[index+1:], blocks)
        else:
            self.trancate_text(inputs, blocks)

def new_TextSplitter(tokenizer, training_type, format='simple', block_size=None, **kwargs):
    splitter = None
    if training_type.startswith('pre'):
        if block_size is None:
            verbose_print(f"block_sizeの指定がないため、block_size={DEFAULT_BLOCK_SIZE}にします。")
            block_size = DEFAULT_BLOCK_SIZE
        if format=='multi':
            splitter = MultiTextBlockSplitter(tokenizer, block_size, **kwargs)
        if splitter is None:
            if format != 'simple':
                verbose_print(f"format={format}は、サポートされていません。")
        splitter = SimpleTextBlockSplitter(tokenizer, block_size, **kwargs)
    elif training_type.startswith('seq2seq'):
        # if splitter is None:
        #     if format != 'simple':
        #         verbose_print(f"format={format}は、サポートされていません。")
        if 'sep' not in kwargs:
            kwargs['sep'] = '<outpuT>'
        splitter = TextPairSplitter(tokenizer, block_size, **kwargs)
    else: # ファインチューニング用
        # if splitter is None:
        #     if format != 'simple':
        #         verbose_print(f"format={format}は、サポートされていません。")
        splitter = SimpleTextSplitter(tokenizer, block_size, **kwargs)
    return splitter


## Store 用

def record_tokenizer(tokenizer: AutoTokenizer):
    allvoc = ''.join(tokenizer.get_vocab().keys())
    sha256 = hashlib.sha256(allvoc.encode()).hexdigest()
    return dict(
        name_or_path=tokenizer.name_or_path,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
        hash=sha256, 
        vocab_size=tokenizer.vocab_size)


N_CHUNKS = 4096

class DatasetStore(object):
    def __init__(self, store_path, prefix, block_size, **kwargs):
        self.store_path = safe_dir(store_path)
        self.prefix = prefix
        self.block_size = block_size
        self.config = {}
        self.file_ext = kwargs.get("file_ext", "npz")
        self.n_chunks = kwargs.get("n_chunks", N_CHUNKS)
        self.shuffle = kwargs.get("shuffle", False)
        self.chunkseq = 0
        self.bufs = []
        self.n_items = 0
        self.n_tokens = 0
        self.chunk_files = []

    def save_config(self):
        self.config.update(dict(
            n_items=self.n_items,
            n_tokens = self.n_tokens,
            chunkseq=len(self.chunk_files),
            n_chunks=self.n_chunks,
            file_ext=self.file_ext,
            shuffle=self.shuffle,
        ))
        file_checks = make_chunk_filelist(self.store_path, self.chunk_files)
        if file_checks:
            self.config['files'] = file_checks
        
        config_file = safe_join_path(self.store_path, f'{self.prefix}config.json')
        with open(config_file, "w") as w:
            json.dump(self.config, w)
    
    def save(self, save_config=True):
        if len(self.bufs) > 0:
            chunk_file = chunkseq_to_filename(self.chunkseq, self.prefix, self.file_ext)
            save_chunk_file(self.store_path, chunk_file, self.bufs)
            self.chunk_files.append(chunk_file)
            self.n_items += len(self.bufs)
            if len(self.bufs) == self.n_chunks:
                self.chunkseq += 1
                self.bufs = []
        if save_config:
            self.save_config()
            verbose_print(f'トークン数: {self.n_tokens:,} 件数: {self.n_items:,}')

    def append(self, block: List[int]):
        self.bufs.append(np.array(block, dtype=np.int32))
        self.n_tokens += len(block)
        if len(self.bufs) == self.n_chunks:
            self.save(save_config=False)

    def extend(self, blocks: List[List[int]]):
        for block in blocks:
            self.append(block)
        return []   


def split_to_store(filename, N=-1,
                   desc=None,
                   tokenizer_path=DEFAULT_TOKENIZER, 
                   training_type='',
                   format='simple', 
                   split='train',
                   block_size=None, # DEFAULT_BLOCKSIZE 
                   store_path=None, 
                   verbose=True, histogram=False,
                   split_args={}):
    
    if isinstance(tokenizer_path, str):
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        tokenizer = tokenizer_path

    filebase = get_filebase(filename)
    if store_path is None:
        _, _, tokenizer_name = tokenizer_path.rpartition('/')
        store_path=f'{tokenizer_name}/{filebase}'
        verbose_print(f'Saving To.. {store_path}')

    splitter = new_TextSplitter(tokenizer, training_type,
                                format=format, block_size=block_size, 
                                **split_args)
    
    prefix = f'{(splitter.prefix+split)}_'
    store = DatasetStore(store_path, prefix, block_size, **split_args)

    if desc:
        store.config['desc'] = desc
    store.config['tokenizer_path'] = str(tokenizer.name_or_path)
    store.config['tokenizer'] = record_tokenizer(tokenizer)
    store.config['splitter'] = splitter.about()

    iterator = file_iterator(filename, N=N)
    splitter.split_iter(iterator=iterator, update_fn=store.extend)
    splitter.report(store.config, verbose=verbose)
    store.save()
    print(store.config)
    if histogram:
        make_histogram(tokenizer, store_path, store.chunk_files, verbose=verbose)


def make_histogram(tokenizer, store_path, chunk_files, verbose=True):
    token_ids = list(range(0, tokenizer.vocab_size))
    vocabs = tokenizer.convert_ids_to_tokens(token_ids)
    counts = [0] * tokenizer.vocab_size
    csv_file = f'{store_path.replace("/", "_")}.csv'
    from tqdm import tqdm
    for chunk_file in tqdm(chunk_files):
        chunks = load_chunk_file(store_path, chunk_file)
        for chunk in chunks:
            for token_id in chunk:
                counts[token_id] += 1
    df = pd.DataFrame({'tokens': vocabs, 'counts': counts})
    if verbose:
        print(df['counts'].describe())
    df.to_csv(csv_file)
    verbose_print(f"字句の出現頻度を'{csv_file}'に保存しました。")

def make_local_store(filename, tokenizer, block_size, args):
    filebase = get_filebase(filename)
    store_path = safe_join_path(args['cache_dir'],filebase)
    split_to_store(filename, 
                   tokenizer_path=tokenizer, 
                   for_pretraining=args.get('training_type', ''),
                   format='simple', 
                   split='train',
                   block_size=block_size, 
                   store_path=store_path,
                   verbose=False, histogram=False,
                   kwargs={})
    return str(os.path.abspath(store_path))