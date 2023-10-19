from typing import Any, List

import os
import random

import json
import shutil
from urllib.parse import urlparse, parse_qs
import hashlib

from collections import deque

from filelock import FileLock
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForLanguageModeling

from .commons import *
from .file_utils import *
from .tokenizers import *
from .splitters import *

_ID = 0
def random_name():
    global _ID
    _ID+= 1
    return f'Cache{_ID-1}'

# 設定ファイル

DEFAULT_BLOCK_SIZE = 2096
DEFAULT_MAX_LENGTH = 4096
N_CHUNKS = 4096

# ChunkedDataset

class _DummyFileLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

def _FileLock(lockfile: str):
    # lockがNoneなら何もしない
    return _DummyFileLock() if lockfile is None else FileLock(lockfile)


def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1

def url_to_hash(url):
    return hashlib.md5(url.encode()).hexdigest()


class ChunkedDataset(Dataset):
    def __init__(self, url:str, url_args: dict, split='train', block_size=None, prefetch=1):
        """
        block_size=Noneのときは、再分割しない (ファインチューニングは再分割しない)
        """
        self.url = safe_dir(url)
        self.url_args = url_args
        self.split = split
        self.block_size = block_size
        self.cache_dir = safe_join_path(url_args['cache_dir'], url_to_hash(url))
        self.lock_file = url_args['lock_file']
        if url_args['training_type'].startswith('pre'):
            self.split_prefix = f"pretrain_"
        else:
            self.split_prefix = f"{split}_"
        self.load_config()
        self.queue = deque(maxlen=64)
        self.cache = {}
        self.prefetch=1
        if self.prefetch > 0 and self.n_items > 0:
            self.try_prefetch(0)

    def get_valid_dataset(self, split='valid'):
        dataset = ChunkedDataset(self.url, self.url_args, split=split, block_size=self.block_size)
        if len(dataset) > 0:
            return dataset
        return None

    def load_config(self):
        with _FileLock(self.lock_file):
            config_file = resolve_file(self.url, f'{self.split_prefix}config.json', self.cache_dir)
        try:
            with open(config_file) as f:
                config = json.load(f)
        except BaseException as e:
            verbose_print(f'見つかりません {self.url} ({config_file})')
            config = dict(n_items=0, n_tokens=0)
        # 設定
        self.file_ext = config.get("file_ext", "npz")
        self.n_tokens = config.get('n_tokens', 0)
        self.n_items = config.get("n_items", 0)
        self.n_chunks = config.get("n_chunks", N_CHUNKS)
        self.tokenizer_path = config.get("tokenizer_path", DEFAULT_TOKENIZER)
        self.max_chunkseq = max(config.get("chunkseq", 1), 1)  # 古いデータ用
        self.n_subblocks = 1
        if self.block_size is not None and 'block_size' in config and self.block_size < config['block_size']:
            self.n_subblocks = config['block_size'] // self.block_size
            if self.n_subblocks > 1:
                self.n_chunks = self.n_chunks * self.n_subblocks
                verbose_print(f'{self.url} は、{self.n_subblocks}個に再分割されます')
        self.is_seq2seq = 'output_sep_token_id' in config
        self.config = config
        return config

    def __len__(self):
        return self.n_items * self.n_subblocks

    def try_prefetch(self, chunkseq):
        chunk_file = chunkseq_to_filename(chunkseq % self.max_chunkseq, self.split_prefix, self.file_ext)
        resolve_file(self.url, chunk_file, self.cache_dir, sync=False)

    def get_chunks(self, chunk_file):
        if chunk_file in self.cache:
            return self.cache[chunk_file]
        with _FileLock(self.lock_file):
            chunk_file2 = resolve_file(self.url, chunk_file, self.cache_dir)
            chunks = load_chunk_file(chunk_file2)
            chunks = self.resize_chunks(chunks)
        if chunks is None:
            # エラーで落ちるくらいなら、キャッシュのデータで学習を続ける
            chunks = self.cache[self.queue[0]]
        # if self.shuffle:
        #     random.shuffle(chunks)
        if len(self.queue) == 64:
            older = self.queue.popleft()
            if older in self.cache:
                del self.cache[older]
        self.queue.append(chunk_file)
        self.cache[chunk_file] = chunks
        return chunks

    def resize_chunks(self, chunks):
        if chunks and self.n_subblocks > 1:
            newchunks=[]
            for chunk in chunks:
                splits = np.array_split(chunk, self.n_subblocks)
                newchunks.extend(splits)
            assert len(newchunks) == len(chunks) * self.n_subbloks
            return newchunks
        return chunks

    def __getitem__(self, index):
        i = index
        chunkseq = i // self.n_chunks
        chunk_file = chunkseq_to_filename(chunkseq, self.split_prefix, self.file_ext)
        chunks = self.get_chunks(chunk_file)
        if self.prefetch > 0 and index % self.n_chunks == 0:
            self.try_prefetch(chunkseq+self.prefetch)
        return chunks[i % self.n_chunks]
    

class DistributedIndexer(Dataset):
    def __init__(self, dataset, url_args:dict):
        self.dataset = dataset
        self.url_args = url_args
        self.count = 0
        self.dataset_size = len(dataset)
        self.valid_dataset = None

        # offset をパラメータから調整する
        self.offset = url_args.get('start', 0)
        if isinstance(self.offset, float) and self.offset < 1.0:
            self.offset = int(self.offset * self.dataset_size)
        self.offset = self.offset % self.dataset_size
        
        # length をパラメータから調整する
        self.length = url_args.get('length', self.dataset_size)
        if isinstance(self.length, float) and self.length < 1.0:
            self.length = int(self.length * self.dataset_size)
        if self.length > self.dataset_size:
            self.length = self.length % self.dataset_size
        self.epoch = 1
        # DistributedSamplerと同様に均等に分割して各プロセスが同じデータを読まないようにする
        self.sublength = self.length // get_world_size()
        if get_world_size() > 1:
            verbose_print('ランクごとに再配置します')
            self.offset = self.offset + (get_rank() * self.sublength)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        index = (self.offset + self.count) % self.dataset_size
        self.count += 1
        if self.count >= self.sublength:
            self.count = 0
            self.epoch += 1
        return self.dataset[index]

    def get_valid_dataset(self, valid_split=0.1):
        if self.valid_dataset:
            return self.valid_dataset
        self.valid_dataset = self.dataset.get_valid_dataset()
        if self.valid_dataset:
            return self.valid_dataset
        # 訓練データを 9:1 に分割して検証データを作る
        valid_dataset = DistributedIndexer(self.dataset, self.url_args)
        valid_dataset.length = int(self.length * valid_split)
        self.length -= valid_dataset.length
        sublength = valid_dataset.sublength
        valid_dataset.sublength = int(self.sublength * valid_split)
        self.sublength -= valid_dataset.sublength
        valid_dataset.offset += (sublength - valid_dataset.sublength)
        self.valid_dataset = valid_dataset
        return self.valid_dataset

    # def get_num_of_tokens(self):
    #     if len(self.dataset) == self.length:
    #         return self.n_tokens
    #     return self.n_tokens * self.length // len(self.dataset)


def build_inputs_for_clm(data, max_length):
    return torch.tensor(data[:max_length].astype(np.int64), dtype=torch.long)

def parse_url_list(url_list=[]):
    if isinstance(url_list, str):
        if os.path.exists(url_list):
            with open(url_list) as f:
                return [url.strip() for url in f.readlines() if url.strip() != '' and not url.startswith('#')]
        return url_list.split('|')
    return url_list


def convert_to_number(value):
    """
    文字列を可能ならば整数または浮動小数点数に変換する。
    変換できない場合はそのままの文字列を返す。
    """
    lower_string = str(value).lower()
    if lower_string == 'true':
        return True
    if lower_string == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return str(value)

def _recalculate_length(mixed):
    dd={}
    n_items = 0
    for ds in mixed:
        if id(ds) not in dd:
            dd[id(ds)] = ds
            n_items += len(ds)
    return n_items

class MixingDataset(Dataset):
    def __init__(self, mixed, n_items, build_fn, max_length):
        self.mixed = mixed
        self.mixing = len(self.mixed)
        self.n_items = n_items
        self.build_fn = build_fn
        self.max_length = max_length

    def __len__(self):
        return self.n_items

    def __getitem__(self, index):
        data = self.mixed[index % self.mixing][index]
        return self.build_fn(data, self.max_length)

    def get_valid_dataset(self, valid_split=0.1):
        mixed = []
        for ds in self.mixed:
            ds = ds.get_valid_dataset(valid_split)
            mixed.append(ds)
        train_size = _recalculate_length(self.mixed)
        valid_size = _recalculate_length(mixed)
        verbose_error(f'訓練データ {train_size} 検証データ {valid_size}')
        self.n_items = min(train_size, self.n_items)
        return MixingDataset(mixed, min(valid_size, self.n_items // 4), self.build_fn, self.max_length)


class DataComposer(MixingDataset):
    def __init__(self, 
                 url_list, training_type="pre", split = 'train', 
                 block_size=None, max_length = DEFAULT_MAX_LENGTH,
                 build_fn=build_inputs_for_clm, 
                 cache_dir = DEFAULT_CACHE_DIR, 
                 random_seed=None, shuffle=True,
                 tokenizer=None, 
                 use_filelock=True, cleanup=True,
                 prefetch=1, test_run=None):
        self.training_type = training_type
        self.split = split
        self.block_size=block_size
        self.max_length = max_length
        self.cache_dir = f'{safe_dir(cache_dir)}/{random_name()}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock_file = f'{self.cache_dir}/lock.me' if use_filelock else None
        self.random_seed=getint_from_environ('KG_RANDOM_SEED', random_seed, 42)
        self.tokenizer_path = None
        self.prepare_data(parse_url_list(url_list), block_size, tokenizer)
        self.cleanup = False if get_rank() > 0 else cleanup
        self.prefetch = prefetch
        self.build_fn = build_fn
        # テスト実行
        test_run = getint_from_environ('KG_TEST_RUN', test_run, None)
        if test_run and isinstance(test_run, int):
            verbose_print('反復を減らして、テスト実行します', test_run)
            self.n_items = min(test_run, self.n_items)

    def parse_url_args(self, url):
        parsed_url = urlparse(url)
        if len(parsed_url.scheme):
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        else:
            base_url = f"{parsed_url.path}"
        params = parse_qs(parsed_url.query)
        url_args = {k: convert_to_number(v[0]) for k, v in params.items()}
        url_args['training_type'] = self.training_type
        if 'split' not in url_args:  # 個別のsplitの指定がなければ... 
            url_args['split'] = self.split
        url_args['prefix'] = f'{self.training_type}{url_args["split"]}_'
        url_args['cache_dir'] = self.cache_dir
        url_args['lock_file'] = self.lock_file
        return base_url, url_args

    def prepare_data(self, urls, block_size, tokenizer=None):
        self.n_items = 0
        self.n_tokens = 0
        datasets = []
        for url in urls:
            url, url_args = self.parse_url_args(url)
            if url.endswith('.gz') or url.endswith('.jsonl') or url.endswith('.txt'):
                tokenizer = self.prepare_tokenizer(tokenizer)
                url = make_local_store(url, tokenizer, block_size, url_args)
            dataset = ChunkedDataset(url, url_args, split=self.split, block_size=block_size)
            if self.check_tokenizer(url, dataset) == False:
                continue
            if self.training_type.startswith('seq2') and not dataset.is_seq2seq:
                verbose_print(f'** {url} は、seq2seqに対応していません。無視して学習を続けます。')
                continue
            if len(dataset) == 0:
                verbose_print(f'** {url} は、スキップして学習を続けます。')
                continue
            dataset = DistributedIndexer(dataset, url_args)
            datasets.append(dataset)
            self.n_items += len(dataset)
            # n_tokens = ds.get_num_of_tokens()
            # verbose_print(f'{url} トークン数: {format_unit(n_tokens)} {n_tokens:,} 件数: {len(ds):,}')
        if self.n_items > 0:
            self.blend_data(datasets)

    def blend_data(self, datasets: List[DistributedIndexer]):
        lens = [len(ds) for ds in datasets]
        total = sum(lens)
        mixer_base = (total // min(lens))+1
        lens = [int((dlen * mixer_base) / total) for dlen in lens]
        verbose_print('ミキサーパターン:', lens)
        self.mixed = []
        for dlen, ds in zip(lens, datasets):
            self.mixed.extend([ds]*dlen)
        random.seed(self.random_seed)
        random.shuffle(self.mixed)
        self.mixing = len(self.mixed)

    def prepare_tokenizer(self, tokenizer=None):
        if tokenizer is not None:
            return tokenizer
        if self.tokenizer_path is None:
            verbose_print(f'トークンナイザーの指定がないので DEFAULT_TOKENIZER={DEFAULT_TOKENIZER}を使います')
            self.tokenizer_path = DEFAULT_TOKENIZER
        return load_tokenizer(self.tokenizer_path)

    def check_tokenizer(self, url, dataset):
        if self.tokenizer_path is None:
            self.tokenizer_path = dataset.tokenizer_path
        elif self.tokenizer_path != dataset.tokenizer_path:
            verbose_error(f'警告: トークンナイザーが一致しません。{self.tokenizer_path}')
            verbose_error(f'    {dataset.tokenizer_path} @{url}')
            verbose_print(f'** {url} は、スキップして学習を続けます。')
            return False
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.n_items = 0
        self.mixed = None
        if self.cleanup and os.path.isdir(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
                verbose_print('Cleaned up', self.cache_dir)
            except:
                pass

    # def __len__(self):
    #     return self.n_items

    # def __getitem__(self, idx):
    #     mix = len(self.mixer)
    #     item = self.mixer[idx % mix][idx]
    #     return self.build_fn(item, self.max_length)




class PretrainComposer(DataComposer):
    def __init__(self, url_list, block_size, **kwargs):
        kwargs['training_type'] = 'pre'
        kwargs['split'] = 'train'
        kwargs['max_length'] = block_size
        DataComposer.__init__(self, url_list, block_size=block_size, **kwargs)

    def get_collator(self):
        tokenizer = load_tokenizer(self.tokenizer_path)
        return DataCollatorForLanguageModeling(tokenizer, mlm=False)

def build_inputs_for_instruct(data, max_length):
    # version = data[0] % CHUNK_MAGIC
    return torch.tensor(data[1:max_length+1].astype(np.int64), dtype=torch.long)

def build_inputs_attn_for_instruct(data, max_length):
    # version = data[0] % CHUNK_MAGIC
    input_ids = torch.tensor(data[1:max_length+1].astype(np.int64), dtype=torch.long)
    attention_mask=torch.ones(len(data), dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask, 
    }


class FinetuneComposer(DataComposer):
    def __init__(self, url_list, split="train", **kwargs):
        kwargs['training_type'] = ''
        if 'block_size' in kwargs: # ブロックサイズは、事前学習用のパラメータ
            kwargs['max_length'] = kwargs['block_size']
            del kwargs['block_size']
        DataComposer.__init__(self, url_list, split=split, **kwargs)
        if kwargs.get('use_attention_mask', False):
            self.build_fn = kwargs.get('build_fn', build_inputs_attn_for_instruct)
        else:
            self.build_fn = kwargs.get('build_fn', build_inputs_for_instruct)

    def get_collator(self):
        tokenizer = load_tokenizer(self.tokenizer_path)
        return DataCollatorForLanguageModeling(tokenizer, mlm=False)


class DP(object):
    def __init__(self, tokenizer, lambda_=20):
        self.lambda_ = lambda_
        self.eos_token_id = tokenizer.eos_token_id
        self.extra_ids = find_extra_ids(tokenizer)
        self.newline_id = find_newline_token_id(tokenizer)

    def __call__(self, data, max_length):
        index = 0
        start = 0
        size = min(max_length, len(data))
        for i, length in enumerate(np.random.poisson(self.lambda_, 1000), start=1):
            start = start + max(1, length) * i
            if start >= size:
                break
            if data[start] != self.eos_token_id or data[start] != self.newline_id:
                data[start] = self.extra_ids[index]
                index+=1
        return torch.tensor(data[:max_length].astype(np.int64), dtype=torch.long)

