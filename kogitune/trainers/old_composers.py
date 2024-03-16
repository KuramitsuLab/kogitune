from typing import Any, List

import os
import random

import json
import shutil
import hashlib

from collections import deque

from filelock import FileLock
import numpy as np

import torch
from torch.utils.data import Dataset

from ..commons import *
from ..utils_file import *
from ..tokenizers import *
from ..stores.store import Metastore
from ..splitters import make_local_store

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

def url_to_hash(url):
    return hashlib.md5(url.encode()).hexdigest()

def local_cache_dir(cache_dir, url):
    hash = url_to_hash(url)
    if hash in cache_dir:
        return cache_dir
    return safe_join_path(cache_dir, hash)

class TensorArrayDataset(Dataset):
    def __init__(self, url:str, prefix: str, args: dict):
        self.url = safe_dir(url)
        self.prefix = prefix if prefix.endswith('_') else f'{prefix}_'
        self.args = args
        # Noneのときは、再分割しない (ファインチューニングは再分割しない)
        self.block_size = args.get('block_size', None)
        self.cache_dir = local_cache_dir(args['cache_dir'], url)
        self.lock_file = args['lock_file']
        self.load_config()
        self.queue = deque(maxlen=64)
        self.cache = {}
        self.prefetch=args.get('prefetch', 1)

    def load_config(self):
        with _FileLock(self.lock_file):
            config_file = resolve_file(self.url, f'{self.prefix}config.json', self.cache_dir)
        try:
            with open(config_file) as f:
                config = json.load(f)
        except BaseException as e:
            verbose_print(f'見つかりません {self.url} ({config_file})')
            config = dict(n_items=0, n_tokens=0)
        # 設定
        self.n_tokens = config.get('n_tokens', 0)
        self.n_items = config.get("n_items", 0)
        self.n_chunks = config.get("n_chunks", N_CHUNKS)
        self.tokenizer_path = config.get("tokenizer_path", DEFAULT_TOKENIZER)
        if 'files' in config:
            self.chunk_files = list(config['files'].keys())
        self.compressed = config.get("compressed", None)
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

    def get_valid_dataset(self, split='valid'):
        valid_prefix=Metastore(self.url, self.cache_dir).find_valid(self.config, split)
        if valid_prefix:
            dataset = TensorArrayDataset(self.url, valid_prefix, self.args)
            return dataset
        return None

    def get_chunks(self, chunk_file):
        if chunk_file in self.cache:
            return self.cache[chunk_file]
        with _FileLock(self.lock_file):
            chunk_file2 = resolve_file(self.url, chunk_file, self.cache_dir, self.compressed)
            chunks = load_chunk_file(chunk_file2, subblocks=self.n_subblocks)
        if chunks is None:
            # エラーで落ちるくらいなら、キャッシュのデータで学習を続ける
            chunks = self.cache[self.queue[0]]
        if len(self.queue) == 64:
            older = self.queue.popleft()
            if older in self.cache:
                del self.cache[older]
        self.queue.append(chunk_file)
        self.cache[chunk_file] = chunks
        return chunks

    def __getitem__(self, index):
        chunk_index = index // self.n_chunks
        chunk_file = self.chunk_files[chunk_index]
        chunks = self.get_chunks(chunk_file)
        if self.prefetch > 0 and index % self.n_chunks == 0:
            self.try_prefetch(index+(self.prefetch*self.n_chunks))
        return chunks[index % self.n_chunks]

    def try_prefetch(self, index):
        if self.prefetch > 0:
            chunk_index = index // self.n_chunks
            chunk_file = self.chunk_files[chunk_index % len(self.chunk_files)]
            resolve_file(self.url, chunk_file, self.cache_dir, self.compressed, sync=False)

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

def verbose_print(*args, **kwargs):
    fox_face = '🦊' * (get_rank() + 1)
    print(fox_face, *args, **kwargs)


class DistributedIndexer(Dataset):
    def __init__(self, dataset: TensorArrayDataset, args: dict):
        self.dataset = dataset
        self.args = args
        self.dataset_size = len(dataset)
        self.valid_dataset = None

        # offset をパラメータから調整する
        self.offset = args.get('start', 0)
        if isinstance(self.offset, float) and self.offset < 1.0:
            self.offset = int(self.offset * self.dataset_size)
        self.offset = self.offset % self.dataset_size
        
        # length をパラメータから調整する
        self.length = args.get('length', self.dataset_size)
        if isinstance(self.length, float) and self.length < 1.0:
            self.length = int(self.length * self.dataset_size)
        if self.length > self.dataset_size:
            self.length = self.length % self.dataset_size
        
        self.epoch = 0
        self.count = 0
        
        # DistributedSamplerと同様に均等に分割して各プロセスが同じデータを読まないようにする
        self.sublength = self.length // get_world_size()
        if get_world_size() > 1:
            self.offset = self.offset + (get_rank() * self.sublength)
            verbose_print(f'データ再配置: rank={get_rank()}, offset={self.offset}, length={self.sublength}')
        dataset.try_prefetch(self.offset)

    def skip(self):
        self.count += 1
        if self.count >= self.sublength:
            self.count = 0
            self.epoch += 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        index = (self.offset + self.count) % self.dataset_size
        self.count += 1
        if self.count >= self.sublength:
            self.count = 0
            self.epoch += 1
        return self.dataset[index]

    def report(self, max_length):
        iterations = self.sublength * self.epoch + self.count
        total_tokens = iterations * max_length
        verbose_print(f'{self.dataset.url}: 反復数{iterations:,} トークン数{total_tokens:,}')
        return iterations
    
    def get_valid_dataset(self, valid_split=0.1):
        if self.valid_dataset:
            return self.valid_dataset
        valid_dataset = self.dataset.get_valid_dataset()
        if valid_dataset:
            self.valid_dataset = DistributedIndexer(valid_dataset)
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
        self.global_count = 0
        self.n_items = n_items
        self.build_fn = build_fn
        self.max_length = max_length

    def __len__(self):
        return self.n_items

    def __getitem__(self, index):
        data = self.mixed[self.global_count % self.mixing][index]
        self.global_count += 1
        return self.build_fn(data, self.max_length)

    def skip(self, index):
        self.mixed[self.global_count % self.mixing].skip()
        self.global_count += 1


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

class DefaultCollator(object):
    def __init__(self, args):
        self.is_seq2seq = args.get('data_type', 'text') == 'seq2seq'

    def __call__(self, data, max_length):
        if self.is_seq2seq:
            version = data[0] % CHUNK_MAGIC
            if version == 1:
                index = (data[0] // CHUNK_MAGIC) + 1
                inputs = data[1:index]
                labels = data[index:]
                return {
                    "input_ids": torch.tensor(inputs.astype(np.int64), dtype=torch.long),
                    "attention_mask": torch.ones(len(inputs), dtype=torch.long),
                    "labels": torch.tensor(labels.astype(np.int64), dtype=torch.long),
                }
            else:
                data = data[1:]
        return {
            "input_ids": torch.tensor(data.astype(np.int64), dtype=torch.long),
            "attention_mask": torch.ones(len(data), dtype=torch.long),
        }

class DataComposer(MixingDataset):
    def __init__(self, url_list, max_length, 
                 cache_dir = None, cleanup=False, 
                 use_filelock=True, random_seed=None, shuffle=True,
#                 build_fn=build_inputs_for_clm, 
                 tokenizer=None, restart=None, test_run=None, **args):
        self.max_length = max_length
        self.data_type = get_dict_multi_keys(args, 'data_type', 'text')
        self.split = get_dict_multi_keys(args, 'split', 'train')
 
        # キャッシュ
        cache_dir = get_environ('KG_CACHE_DIR|CACHE_DIR', None, param_specified=cache_dir)
        if cache_dir is None:
            self.cache_dir = safe_join_path('.', get_filename_by_pid('cache'))
            self.cleanup = False if get_rank() > 0 else True
        else:
            self.cache_dir = safe_dir(cache_dir)
            self.cleanup = False if get_rank() > 0 else cleanup
        if os.path.isdir(self.cache_dir):
            verbose_print(f'既に存在するキャッシュ {self.cache_dir} を使います。')
            self.cleanup = False
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock_file = safe_join_path(self.cache_dir, get_filename_by_pid('cache')) if use_filelock else None

        self.random_seed=getint_environ('KG_RANDOM_SEED|RANDOM_SEED', 42, param_specified=random_seed)
        self.tokenizer_path = None
        self.datasets = []
        self.prepare_data(parse_url_list(url_list), tokenizer)
        self.build_fn = DefaultCollator(args)

        # テスト実行
        restart = getint_environ('KG_START', 0, param_specified=restart)
        self.global_count = 0
        if restart > 0:
            verbose_print(f'{restart}回(イテレーション)から、継続学習します')
            for i in range(restart):
                self.skip(i)
        test_run = getint_environ('KG_TEST_RUN|TEST_RUN', None, param_specified=test_run)
        if test_run and isinstance(test_run, int):
            verbose_print(f'イテレーションを {test_run} 回に減らして、テスト実行します')
            self.n_items = min(test_run, self.n_items)

    def prepare_data(self, urls, tokenizer=None):
        global_args = {
            'cache_dir': self.cache_dir,
            'lock_file': self.lock_file,
            'data_type': self.data_type,
            'split': self.split,
            'max_length': self.max_length,
        }
        self.n_items = 0
        self.n_tokens = 0
        for url in urls:
            url, args = parse_url_args(url, global_args)
            if url.endswith('.gz') or url.endswith('.zst') or url.endswith('.jsonl') or url.endswith('.txt'):
                tokenizer = self.prepare_tokenizer(tokenizer)
                url = make_local_store(url, tokenizer, args)
            args['cache_dir'] = local_cache_dir(args['cache_dir'], url)
            prefix = Metastore(url, args['cache_dir']).find_better_size(args)
            if prefix is None:
                verbose_print(f'データセット {url} には、適切なデータがありません')
                print('CHECKME', args)
                continue
            dataset = TensorArrayDataset(url, prefix, args)
            if len(dataset) == 0:
                verbose_print(f'データセット {url} は、スキップして学習を続けます。')
                continue
            if self.check_tokenizer(url, dataset) == False:
                continue
            verbose_print(f'{url} {prefix} トークン数: {format_unit(dataset.n_tokens)} {dataset.n_tokens:,} 件数: {len(dataset):,}')
            dataset = DistributedIndexer(dataset, args)
            self.datasets.append(dataset)
            self.n_items += len(dataset)
        if self.n_items > 0:
            self.blend_data(self.datasets)
        else:
            raise ValueError('🦊 データセットが空っぽです。')

    def blend_data(self, datasets: List[DistributedIndexer]):
        dataset_lengths = [len(ds) for ds in datasets]
        if len(dataset_lengths) > 1:
            #dataset_lengths = [1009, 100, 300, 100, 45, 32, 29]
            total = sum(dataset_lengths)
            base = total / min(dataset_lengths) * 7
            lens = [max(int((dlen * base) / total),1) for dlen in dataset_lengths]
            verbose_print(f'最大: {max(dataset_lengths):,} 最小: {min(dataset_lengths):,} 混成比率: {lens}')
        else:
            lens=[1]
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

    def get_tokenizer(self):
        return self.prepare_tokenizer(None)

    def check_tokenizer(self, url, dataset):
        if self.tokenizer_path is None:
            self.tokenizer_path = dataset.tokenizer_path
        elif self.tokenizer_path != dataset.tokenizer_path:
            verbose_error(f'警告: トークンナイザーが一致しません。{self.tokenizer_path}')
            verbose_error(f'    {dataset.tokenizer_path} @{url}')
            verbose_print(f'** {url} は、スキップして学習を続けます。')
            return False
        return True
    
    def report(self):
        total_tokens = self.global_count * self.max_length
        verbose_print(f'イテレーション {self.global_count:,} トークン数 {format_unit(total_tokens)} {total_tokens:,}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.n_items = 0
        self.mixed = None
        self.report()
        if self.cleanup and os.path.isdir(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
                verbose_print('Cleaned up', self.cache_dir)
            except:
                pass
