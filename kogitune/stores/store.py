from typing import List, Union, Tuple
import random
import json
import numpy as np

from .tokenizers import *
from .files import *
from .chunks import *
import kogitune.adhocs as adhoc

## meta

DEFAULT_MAX_LENGTH = 4096
N_CHUNKS = 4096

class DatasetStore(object):
    def __init__(self, store_path, **kwargs):
        with adhoc.from_kwargs(**kwargs) as aargs:
            self.store_path = safe_dir(store_path)

            self.data_type = aargs['datatype|data_type|=text']
            self.block_size = aargs['max_length|block_size|=2048']
            self.split = aargs['split|=train']
            self.prefix = f"{self.data_type}{self.block_size}{self.split}"

            self.config_file = safe_join_path(self.store_path, f'{self.prefix}_config.json')
            self.chunk_files = []

            self.file_ext = aargs.get("file_ext", "npz")
            self.compressed = aargs.get("compressed", "zst")
            self.n_chunks = aargs.get("n_chunks", N_CHUNKS)
            self.max_length = 0
            self.mix_length = DEFAULT_MAX_LENGTH * 10
            self.chunks = []
            self.n_items = 0
            self.checked_files = {}
            self.load_config(append_mode=aargs['append_mode|append'])

    def load_config(self, append_mode=True):
        config = None
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config = json.load(f)
        if config is None or 'files' not in config:
            return
        chunk_files = list(config['files'].keys())
        if append_mode:
            if config.get('prefix') != self.prefix:
                raise ValueError('prefixが一致しない')
            self.data_type = config.get('datatype', self.data_type)
            self.split = config.get('split', self.split)
            self.n_chunks = config.get('n_chunks', self.n_chunks)
            self.file_ext = config.get('file_ext', self.file_ext)
            self.compressed = config.get('compressed', self.compressed)
            self.checked_files = config['files']

            last_chunk_file = chunk_files[-1]
            if self.compressed:
                last_chunk_file = uncompress_file(f'{last_chunk_file}.{self.compressed}', 
                                                  compressed=self.compressed, rm=True)
            chunks = load_chunk_file(self.store_path, last_chunk_file)
            if len(chunks) < self.n_chunks:
                self.chunks = chunks
                self.chunk_files = chunk_files[:-1]
                del self.check_files[last_chunk_file]
                os.remove(last_chunk_file)
            else:
                self.chunks = []
                self.chunk_files = chunk_files
            self.n_items = len(self.chunk_files) * self.n_chunks
            self.max_length = config.get('max_length', self.max_length)
            self.mix_length = config.get('min_length', self.min_length)
        else:
            for file in chunk_files:
                filepath = safe_join_path(self.store_path, file)
                if os.path.exists(filepath):
                    os.remove(filepath)
                if os.path.exists(f'{filepath}.{self.compressed}'):
                    os.remove(f'{filepath}.{self.compressed}')

    def shuffle_chunk(self, N):
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
            chunk_file = chunkseq_to_filename(len(self.chunk_files), self.prefix, self.file_ext)
            save_chunk_file(self.store_path, chunk_file, self.chunks)
            self.chunk_files.append(chunk_file)
            self.n_items += len(self.chunks)
            self.chunks = []

    def append(self, block: List[int]):
        if len(block) > 0 and not isinstance(block[0], int):
            for inner_block in block:
                self.append(inner_block)
            return

        self.chunks.append(np.array(block, dtype=np.int32))
        self.max_length = max(len(block), self.max_length)
        self.min_length = min(len(block), self.mix_length)
        if len(self.chunks) == self.n_chunks:
            self.save_chunk()

    def check_files(self, skip_validation=True):
        for chunk_file in adhoc.tqdm(self.chunk_files, desc='File validation'):
            if chunk_file not in self.checked_files:
                filepath = safe_join_path(self.store_path, chunk_file)
                checks = {
                    'filesize': get_filesize(filepath), 
                    'sha1': get_filesha1(filepath),
                }
                self.checked_files[chunk_file] = checks
            if not skip_validation:
                if not load_chunk_file('', filepath):
                    adhoc.print(f'unloaded and broken chunk file {chunk_file}')
                    return None
                if not check_chunk_file(self.store_path, chunk_file, checks):
                    adhoc.print(f'invalidate chunk file {chunk_file}')
                    return None

    def compress(self, compressed='zst'):
        if compressed:
            for file in adhoc.tqdm(self.chunk_files, desc='File compressed'):
                filepath = safe_join_path(self.store_path, file)
                compressed_file = f'{filepath}.{compressed}'
                if not os.path.exists(compressed_file):
                    compress_file(filepath, compressed=compressed, rm=True)

    def save(self, tokenizer, skip_validation=True, compressed='zst'):
        fraction = 0 # 端数
        if len(self.chunks) > 0:
            fraction = len(self.chunks)
            self.save_chunk()
        self.check_files(skip_validation=skip_validation)
        self.compress(compressed=compressed)
        n_files = len(self.checked_files)
        if fraction > 0:
            n_items = (n_files-1) * self.n_chunks + fraction
        else:
            n_items = n_files * self.n_chunks
        n_tokens = n_items * self.block_size
        config = dict(
#            source = source,
            datatype = self.data_type,
            split = self.split,
            prefix = self.prefix,
            store_path = self.store_path, 
            tokenizer_path = tokenizer.name_or_path,
            vocab_size = tokenizer.vocab_size,
            tokenizer_id =  tokenizer_id(tokenizer),
            n_chunks = self.n_chunks,
            fraction = fraction,
            n_items = n_items,
            n_tokens = n_tokens,
            block_size = self.block_size,
            max_length = self.max_length,
            min_length = self.min_length,
            file_ext = self.file_ext,
            compressed=compressed,
            files = self.checked_files,
        )
        with open(self.config_file, "w") as w:
            json.dump(config, w, indent=2)
        adhoc.print(f'チャンク: {n_files}, 件数: {n_items:,}, トークン数: {n_tokens:,}')

