from typing import List, Union, Tuple
import random
import json
import re
import numpy as np
import pandas as pd

import hashlib
from transformers import AutoTokenizer

from .tokenizers import *
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
            json.dump(self.config, w)
        self.update_metadata()
        verbose_print(f'トークン数: {self.n_tokens:,} 件数: {self.n_items:,}')

    def update_metadata(self):
        index_file = safe_join_path(self.store_path, f'index.json')
        metadata = read_metadata(index_file)
        split = metadata.get('prefixes', {})
        summary = {k:v for k,v in self.config.items() if isinstance(v, (int, float, bool, str))}
        split[self.prefix] = summary
        metadata['prefixes'] = split
        write_metadata(index_file, metadata)

def getint(args, keys, default):
    return get_dict_multi_keys(args, keys, default, format_fn=int)

empty_tokens = []

class DefaultSplitter(object):

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.split_args = args
        self.max_length = getint(args, 'max_length|block_size', DEFAULT_MAX_LENGTH)
        self.min_length = getint(args, 'min_length', self.max_length//4)
        self.ellipsis_token_id = find_ellipsis_token_id(tokenizer)
        # self.pad_token_id = getint(args, 'pad_token_id', tokenizer.pad_token_id)
        # self.eos_token_id = getint(args, 'eos_token_id', tokenizer.eos_token_id)
        # self.trancate_size=getint(args, 'trancate_size', 0)
        # self.sep = args.get('sep', None)
        self.stat_token_counts = []
        self.text_token_count = 0
        self.text_length_count = 0
        self.total_count = 0
        self.drop_count = 0
        self.token_count = 0
        self.trancate_count = 0
        self.overlap_count = 0
        self.padding_count = 0
    
    def about(self):
        return dict(name=self.__class__.__name__)

    def split(self, text:str, blocks: List[List[int]]):
        raise NotImplemented()

    def flush(self, blocks: List[List[int]]):
        pass

    def split_iter(self, iterator, update_fn=lambda x: None):
        blocks = []
        for text in iterator:
            self.total_count += 1
            self.split(text, blocks)
            update_fn(blocks)
            blocks = []
        self.flush(blocks)
        update_fn(blocks)

    def encode_and_count(self, text, eos=True):
        self.text_length_count += len(text)
        if eos:
            tokens = self.tokenizer.encode(text)
            self.text_token_count += len(tokens)-1 
        else:
            tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
            self.text_token_count += len(tokens)
        if len(self.stat_token_counts) < 10000:
            self.stat_token_counts.append(len(tokens))
        return tokens

    # def resize_token_counts(self, size):
    #     self.token_counts[-1] += size

    def report(self, logs: dict, verbose=True):
        logs['source_tokens'] = self.text_token_count
        logs['source_chars'] = self.text_length_count
        logs['tokens_per_char'] = self.text_token_count / self.text_length_count
        if verbose:
            verbose_print(f'文字数 {format_unit(self.text_length_count, scale=1000)} {self.text_length_count} 圧縮率（トークン/文字数） {self.text_token_count*100 / self.text_length_count:.2f}%')
            verbose_print(f'１件あたりのトークン長の統計情報 {len((self.stat_token_counts))}/{self.total_count}')
            print(pd.DataFrame({'tokens': self.stat_token_counts}).describe())
        logs['token_stats'] = _record_tokens(self.stat_token_counts)

        drop = self.drop_count / self.total_count
        logs['dropped'] = self.drop_count
        logs['n_tokens'] = self.token_count
        trancated = self.trancate_count / self.token_count
        logs['trancated'] = trancated
        padding = self.padding_count / (self.total_count*self.max_length)
        logs['padding'] = padding
        overlap = self.overlap_count / self.token_count
        logs['overlap'] = overlap
        if verbose:
            verbose_print(f'トークン数: {format_unit(self.token_count, scale=1000)} {self.token_count:,} ブロック長: 最大(max_length){self.max_length} 最小(min_length){self.min_length}')
            verbose_print(f'未使用(ドロップ): {drop*100:.2f}% {self.drop_count:,}/{self.total_count:,}')
            if not hasattr(self, 'trancate_size'):
                self.trancate_size = -1
            verbose_print(f'切り詰め(trancate={self.trancate_size}): {trancated*100:.2f}% ({self.trancate_count:,}/{self.token_count:,})')
            if hasattr(self, 'overlap_size'):
                verbose_print(f'オーバーラップ(overlap={self.overlap_size}): {overlap*100:.2f}% ({self.overlap_count:,}/{self.token_count:,})')
            verbose_print(f'想定パディング: {padding*100:.2f}% ({self.padding_count:,}/{self.total_count*self.max_length:,})')


class SimpleTextBlockSplitter(DefaultSplitter):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.extra_tokens=empty_tokens
        self.trancate_size = getint(args, 'trancate_size', self.max_length // 8)

    def split(self, text:str, blocks: List[List[int]]):
        tokens = self.encode_and_count(text)
        work_size = self.max_length
        tokens = self.extra_tokens + tokens
        for i in range(0, len(tokens) - work_size + 1, work_size):  
            segmented = tokens[i : i + work_size]
            blocks.append(segmented)
            self.token_count += work_size
        extra_size = len(tokens) % work_size
        if extra_size < self.trancate_size :
            self.trancate_count += extra_size            
            self.extra_tokens = empty_tokens
        else:
            self.extra_tokens = tokens[-extra_size:]

    def report(self, logs: dict = None, verbose=True):
        super().report(logs, verbose=verbose)
        if logs:
            logs['block_size'] = self.max_length

LINE_PATTERN = re.compile(r'\n([^\n])')

def add_section_for_line(text):
    return LINE_PATTERN.sub(r'\n<sectioN>\1', text)

DOC_PATTERN = re.compile(r'\n\n([^\n])')

def add_section_for_doc(text):
    return DOC_PATTERN.sub(r'\n\n<sectioN>\1', text)

PYTHON_PATTERN = re.compile(r'\n(def|    def|class|\nif|\ntry|\n#|\n[A-Za-z0-9_]+\s=) ')

def add_section_for_python(code):
    return PYTHON_PATTERN.sub(r'\n<sectioN>\1 ', code)

MARKDOWN_PATTERN = re.compile(r'\n(#|[^\n])')

def add_section_for_markdown(text):
    return MARKDOWN_PATTERN.sub(r'\n<sectioN>\1', text)

def find_add_section(section):
    if section == 'python' or section == 'py':
        return add_section_for_python
    if section == 'markdown' or section == 'md':
        return add_section_for_markdown
    if section == 'line':
        return add_section_for_line
    return add_section_for_doc

class OverlapTextBlockSplitter(DefaultSplitter):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.section = args.get('section', 'doc').lower()
        self.add_section = find_add_section(self.section)
        self.overlap_size = getint(args, 'overlap_size|overlap', self.max_length // 4)
        self.pad_id = find_token_id(tokenizer, '\n', '<nL>')
        self.padding_size = getint(args, 'padding_size|padding', self.max_length // 8)


    def split(self, text:str, blocks: List[List[int]]):
        text = self.add_section(text)
        text_blocks = text.split('<sectioN>')
        chunks = [self.encode_and_count(sec, eos=False) for sec in text_blocks]
        chunk_size = len(chunks)
        work_size = self.max_length
        i = 0
        while i < chunk_size:
            tokens = chunks[i]
            overlapped = 0
            i += 1
            while len(tokens) < work_size and i < chunk_size:
                tokens += chunks[i]
                overlapped = len(chunks[i])
                i += 1
            for j in range(0, len(tokens) - work_size + 1, work_size):  
                blocks.append(tokens[j : j + work_size])
                self.token_count += work_size
            # -----------------> overlapped
            # -----> ==========> reminder は引く必要がある
            reminder = len(tokens) % work_size  # 残り
            if 0 < (overlapped - reminder) < self.overlap_size:
                self.overlap_count += (overlapped - reminder)
                i -= 1
                continue
            # 切り捨てる
            self.trancate_count += reminder
            

    def report(self, logs: dict = None, verbose=True):
        super().report(logs, verbose=verbose)
        if logs:
            logs['block_size'] = self.max_length
            logs['section'] = self.section


class MultiTextBlockSplitter(DefaultSplitter):
    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.work_size = getint(args, 'work_size', 512)
        self.pad_size = getint(args, 'pad_size', 64)
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

class SimpleTextSplitter(DefaultSplitter):

    def __init__(self, tokenizer, args):
        super().__init__(tokenizer, args)
        self.output_sep_token_id = find_token_id(tokenizer, 
                                                 args.get('output_sep', '<outpuT>'), 
                                                 args.get('sep', '<seP>'), 
                                                 '<sep>', '<nL>', '\n', ' ')
        self.trancate_size = getint(args, 'trancate|trancate_size', max(self.min_length, self.max_length//4))

    def report(self, logs: dict = None, verbose=True):
        super().report(logs, verbose=verbose)
        if self.output_sep_token_id is not None:
            logs['output_sep_token_id'] = self.output_sep_token_id            

    def trancate_text(self, inputs: List[int], blocks: List[List[int]]):
        inputs_size = len(inputs)
        if inputs_size > self.max_length:
            if inputs_size - self.max_length > self.trancate_size:
                # 切り詰めが大きすぎる
                self.drop_count +=1
                return
            half_size = self.max_length // 2
            prefix = inputs[:half_size]
            suffix = inputs[-half_size:]
            if self.ellipsis_token_id:
                prefix[-1] = self.ellipsis_token_id
            inputs = prefix + suffix
            self.trancate_count += (inputs_size - len(inputs))
        else:
            self.padding_count += self.block_size - inputs_size
        self.token_count += len(inputs)
        blocks.append([0] + inputs)

    def trancate_pair(self, inputs: List[int], labels: List[int], blocks: List[List[int]]):
        inputs_len = len(inputs)
        labels_len = len(labels)
        total_len = inputs_len + labels_len
        if total_len < self.min_length and labels_len >= self.max_length:
            # ラベルの方が大きい場合は諦める
            self.drop_count +=1
            return
        if total_len > self.max_length:
            trimming_size = self.max_length - total_len
            if inputs_len - trimming_size < (self.max_length // 4):
                # 諦める
                self.drop_count +=1
                return
            inputs = inputs[:-trimming_size]
            self.trancate_count += trimming_size
            assert len(inputs) + labels_len == self.max_length
        index = len(inputs)
        self.padding_count += self.max_length - (len(inputs)+len(labels))
        self.token_count += self.max_length
        inputs[-1] = self.output_sep_token_id
        blocks.append([(index * CHUNK_MAGIC) + 1] + inputs + labels)

    def split(self, text:Union[str, Tuple[str,str]], blocks: List[List[int]]):
        if isinstance(text, tuple):
            inputs = self.encode_and_count(text[0])        
            labels = self.encode_and_count(text[1])
            self.trancate_pair(inputs, labels, blocks)
        else:
            inputs = self.encode_and_count(text)
            self.trancate_text(inputs, blocks)
            self.output_sep_token_id = None


def select_splitter(tokenizer, args: dict):
    splitter = None
    data_type = get_dict_multi_keys(args, 'data_type|training_type', 'text')
    if data_type == 'text':
        format = args.get('format', 'simple')
        args['min_length'] = args['max_length']
        if format=='overlap' or 'section' in args or 'overlap' in args:
            args['format'] = 'overlap'
            splitter = OverlapTextBlockSplitter(tokenizer, args)
        if splitter is None:
            if format != 'simple':
                verbose_print(f"format={format}は、サポートされていません。")
            splitter = SimpleTextBlockSplitter(tokenizer, args)
    else: # ファインチューニング用
        splitter = SimpleTextSplitter(tokenizer, args)
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

def append_valid_file(val_files: List[str], filename: str):
    if '_train' in filename:
        filename2 = filename.replace('_train', '_valid')
        if os.path.isfile(filename2):
            val_files.append(filename2)
            return
        filename2 = filename.replace('_train', '_dev')
        if os.path.isfile(filename2):
            val_files.append(filename2)
            return
        filename2 = filename.replace('_train', '_val')
        if os.path.isfile(filename2):
            val_files.append(filename2)
            return

def split_to_store(filenames: List[str], validation=True, args={}):
    if isinstance(filenames, str):
        filenames = filenames.split('|')

    tokenizer = get_dict_multi_keys(args, 'tokenizer|tokenizer_path', DEFAULT_TOKENIZER)
    if isinstance(tokenizer, str):
        tokenizer = load_tokenizer(tokenizer)
    
    store_path = get_dict_multi_keys(args, 'store_path|store_dir', None)
    if store_path is None:
        filebase = get_filebase(filenames[0])
        _, _, tokenizer_name = tokenizer.name_or_path.rpartition('/')
        store_path=f'{tokenizer_name}/{filebase}'
        args['store_path'] = store_path
        verbose_print(f'Saving To.. {store_path}')
    else:
        filebase = store_path.replace('/', '_')

    specified_data_type = get_dict_multi_keys(args, 'data_type|type', None)
    data_type = detect_datatype(filenames[0], args)
    if specified_data_type is not None and specified_data_type != data_type:
        verbose_print('警告: データ形式が違うようです')
    args['data_type']=data_type
    split = get_dict_multi_keys(args, 'split', 'train')
    data_size = getint(args, 'max_length|block_size', -1)
    prefix = f"{data_type}{data_size}{split}"
    random.seed(getint(args, 'random_seed', 42))

    store = DatasetStore(store_path, prefix, args)
    splitter = select_splitter(tokenizer, args)
    store.config.update(dict(
        source = filenames,
        data_type = data_type, split=split,
        tokenizer_path = str(tokenizer.name_or_path),
        tokenizer = record_tokenizer(tokenizer),
        splitter = splitter.about(),
    ))
    print(store.config)
    val_files = []
    for filename in filenames:
        filename, file_args = parse_url_args(filename)
        data_type = detect_datatype(filename, file_args)
        iterator = iterate_line(filename, N=getint(file_args, 'N|n', -1), args=file_args)
        splitter.split_iter(iterator=iterator, update_fn=store.extend)
        append_valid_file(val_files, filename)

    splitter.report(store.config, verbose=args.get('verbose', False))
    store.save(validation=validation)
    verbose_print({k: v for k, v in store.config.items() if not isinstance(v, dict)})

    if args.get('histogram', False):
        make_histogram(tokenizer, store_path, store.chunk_files, verbose=args.get('verbose', False))

    if len(val_files) > 0:
        verbose_print('split="valid"も作成します')
        args['split'] = 'valid'
        args['histogram'] = False
        split_to_store(val_files, validation=validation, args=args)

def make_local_store(filename:str, tokenizer, args:dict):
    if 'cache_dir' in args and 'store_path' not in args:
        filebase = get_filebase(filename)
        args['store_path'] = safe_join_path(args['cache_dir'], filebase)
    args['tokenizer'] = tokenizer
    split_to_store(filename, validation=args.get('validation', False), args=args)
    return str(os.path.abspath(args['store_path']))

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


