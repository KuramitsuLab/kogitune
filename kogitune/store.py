from typing import List, Union, Tuple
import random
import json
import numpy as np

from .tokenizers import *
from .commons import *
from .file_utils import *

## meta

class Metastore(object):
    def __init__(self, index_file_or_url=None, cache_dir=None):
        self.index_file = None
        self.metadata  = {}
        if index_file_or_url:
            self.load(index_file_or_url, cache_dir)

    def load(self, index_file_or_url, cache_dir=None):
        if cache_dir is not None:
            index_file = resolve_file(index_file_or_url, 'kogitune.json', cache_dir)
        else:
            index_file = self.index_file = index_file_or_url
        if os.path.exists(index_file):
            with open(index_file, "r") as f:
                self.metadata = json.load(f)
        return self

    def save(self):
        if self.index_file:
            with open(self.index_file, "w") as f:
                json.dump(self.metadata, f, indent=2)

    def update(self, store_path, prefix, config):
        index_file = safe_join_path(store_path, f'kogitune.json')
        self.load(index_file)
        lists = self.metadata.get('prefixes', {})
        summary = {k:v for k,v in config.items() if isinstance(v, (int, float, bool, str))}
        lists[prefix] = summary
        self.metadata['prefixes'] = lists
        self.save()

    def guess_data_type(self, args: dict):
        if 'data_type' in args and args['data_type'] is not None:
            return args['data_type']
        lists = self.metadata.get('prefixes', None)
        if lists is not None:
            for _, config in lists.items():
                if 'data_type' in config:
                    verbose_print('指定がなかったので data_type="{config['data_type']}"を採用します。')
                    return config['data_type']
        return None

    def guess_max_length(self, args: dict):
        if 'max_length' in args and args['max_length'] is not None:
            return args['max_length']
        lists = self.metadata.get('prefixes', None)
        if lists is not None:
            for _, config in lists.items():
                if 'max_length' in config:
                    return config['max_length']
        return -1

    def find_better_size(self, args: dict):
        if 'prefix' in args:
            return args['prefix']
        prefixes = self.metadata.get('prefixes', None)
        if prefixes is None: # older version
            return 'pretrain' if args.get('data_type', 'text') else 'train'
        req_max_length = args['max_length']
        data_type = self.guess_data_type(args)
        split = args.get('split', 'train')
        selected_prefix = None
        selected_max_length = 0
        for prefix, config in prefixes.items():
            if config['data_type'] == data_type and config['split'] == split:
                max_length = config.get('max_length', -1)
                if req_max_length <= max_length and max_length > selected_max_length:
                    selected_max_length = max_length
                    selected_prefix = prefix
        if not selected_prefix:
            for prefix, config in prefixes.items():
                if config['data_type'] == data_type and config['split'] == split:
                    max_length = config.get('max_length', -1)
                    if max_length >= selected_max_length:
                        selected_max_length = max_length
                        selected_prefix = prefix
            if selected_max_length > 0:
                verbose_print(f"ブロック長が小さすぎます。max_length={selected_max_length}が適切です。")
        return selected_prefix

    def find_valid(self, base, split='valid'):
        prefixes = self.metadata.get('prefixes', {})
        if prefixes is not None:
            data_type = self.guess_data_type(base)
            max_length = config.get('max_length', -1)
            for prefix, config in prefixes.items():
                if config['split'] == split and config['data_type'] == data_type and config['max_length'] == max_length:
                    return prefix
        return None


N_CHUNKS = 4096

class DatasetStore(object):
    def __init__(self, store_path, prefix, args):
        self.store_path = safe_dir(store_path)
        self.block_size = args.get('block_size', None)
        self.prefix = prefix
        self.config_file = safe_join_path(self.store_path, f'{self.prefix}_config.json')
        self.file_ext = args.get("file_ext", "npz")
        self.n_chunks = args.get("n_chunks", N_CHUNKS)
        self.config = {}
        self.chunk_files = []
        self.chunkseq = 0
        self.chunks = []
        self.shuffle_n = args.get("shuffle", 0)
        self.n_items = 0
        self.n_tokens = 0
        self.max_length = 0
        self.mix_length = DEFAULT_MAX_LENGTH*10
        self.clear_files()

    def clear_files(self):
        if not os.path.exists(self.config_file):
            return
        with open(self.config_file, "r") as f:
            config = json.load(f)
            if 'files' not in config:
                return
            for file in config['files'].keys():
                filepath = safe_join_path(self.store_path, file)
                if os.path.exists(filepath):
                    os.remove(filepath)
                if os.path.exists(f'{filepath}.zst'):
                    os.remove(f'{filepath}.zst')

    def shuffle_chunk(self, N=4):
        if len(self.chunk_files) > N:
            files = random.sample(self.chunk_files, N)
            buffers=[]
            filedata=[]
            for file in files:
                chunks = load_chunk_file(self.store_path, file)
                if chunks is None:
                    return False
                buffers.extend(chunks)
                filedata.append((file, len(buffers), len(buffers)+len(chunks)))
            reminder = len(buffers)
            buffers.extend(self.chunks)
            random.shuffle(buffers)
            for file, start, end in filedata:
                save_chunk_file(self.store_path, file, buffers[start:end])
            self.chunks = buffers[reminder:]
        return True

    def save_chunk(self):
        if len(self.chunks) > 0:
            if self.shuffle_n > 0:
                self.shuffle_chunk(N=self.shuffle_n)
            chunk_file = chunkseq_to_filename(len(self.chunk_files), self.prefix, self.file_ext)
            save_chunk_file(self.store_path, chunk_file, self.chunks)
            self.chunk_files.append(chunk_file)
            self.n_items += len(self.chunks)
            self.chunks = []

    def append(self, block: List[int]):
        self.chunks.append(np.array(block, dtype=np.int32))
        self.n_tokens += len(block)
        self.max_length = max(len(block), self.max_length)
        self.min_length = min(len(block), self.mix_length)
        if len(self.chunks) == self.n_chunks:
            self.save_chunk()

    def extend(self, blocks: List[List[int]]):
        for block in blocks:
            self.append(block)
        return []   

    def check_files(self, validation=True):
        d = {}
        for chunk_file in tqdm(self.chunk_files, desc='File validation'):
            filepath = safe_join_path(self.store_path, chunk_file)
            checks = {
                'filesize': get_filesize(filepath), 
                'sha1': get_file_sha1(filepath)
            }
            d[chunk_file] = checks
            if validation:
                if not load_chunk_file('', filepath):
                    verbose_print(f'unloaded and broken chunk file {chunk_file}')
                    return None
                if not check_chunk_file(self.store_path, chunk_file, checks):
                    verbose_print(f'invalidate chunk file {chunk_file}')
                    return None
        self.config['files'] = d

    def compress(self, compressed_ext='zst'):
        if 'compressed' not in self.config:
            zfiles=[]
            for file in tqdm(self.chunk_files, desc='File compression'):
                filepath = safe_join_path(self.store_path, file)
                filepath = zstd_file(filepath, rm=True)
                zfiles.append(f'{file}.{compressed_ext}')
            self.config['compressed'] = compressed_ext
            self.chunk_files=zfiles

    def save(self, validation=True, compression='zst'):
        if len(self.chunks) > 0:
            self.save_chunk()
        self.config.update(dict(
            n_items = self.n_items,
            n_chunks=self.n_chunks,
            n_tokens = self.n_tokens,
            max_length = self.max_length,
            min_length = self.min_length,
            chunkseq=len(self.chunk_files),
            file_ext=self.file_ext,
        ))
        self.check_files(validation=validation)
        if compression:
            self.compress()
        with open(self.config_file, "w") as w:
            json.dump(self.config, w, indent=2)
        Metastore().update(self.store_path, self.prefix, self.config)
        verbose_print(f'トークン数: {self.n_tokens:,} 件数: {self.n_items:,}')

