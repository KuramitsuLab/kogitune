from typing import List, Union
import os
import time
import random
from pathlib import Path

import json
import hashlib
import subprocess
from urllib.parse import urlparse, parse_qs


import numpy as np
import gzip
import pyzstd

from tqdm import tqdm

#import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from .commons import *

# ファイルシステム

def safe_dir(dir):
    if dir.endswith('/'):
        dir = dir[:-1]
    return dir

def safe_join_path(dir, file):
    if file is None: 
        return dir
    if dir.endswith('/'):
        dir = dir[:-1]
    if file.startswith('/'):
        file = file[1:]
    return f'{dir}/{file}'

def safe_new_file(filebase, ext, max=1000):
    filename=f'{filebase}.{ext}'
    if not os.path.exists(filename):
        return filename
    for i in range(1, max):
        filename=f'{filebase}_{i}.{ext}'
        if not os.path.exists(filename):
            break
    return filename

def get_filebase(filename):
    filebase = filename
    if '/' in filebase:
        _, _, filebase = filebase.rpartition('/')
    filebase, _, _ = filebase.partition('.')
    return filebase

def get_filename_by_pid(prefix='cache'):
    return f'{prefix}{os.getpid()}'


## file 

def zopen(filepath, mode='rt'):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode)
    elif filepath.endswith('.zst'):
        return pyzstd.open(filepath, mode)
    else:
        return open(filepath, mode)

import re
fileline_pattern = re.compile(r"L(\d{4,})\D")

def extract_linenum_from_filename(filepath):
    matched = fileline_pattern.search(filepath)
    if matched:
        return int(matched.group(1))
    return None

def rename_with_linenum(filepath: str, N: int, ext='json', rename=True):
    extracted = extract_linenum_from_filename(filepath)
    if extracted:
        newpath = filepath.replace(f'L{extracted}', f'L{N}')
    else:
        newpath = filepath.replace(f'.', f'_L{N}.', 1)
    if rename:
        if os.path.exists(newpath):
            os.remove(newpath)
        if os.path.exists(filepath):
            os.rename(filepath, newpath)
    return newpath

def get_linenum(filepath):
    ret = extract_linenum_from_filename(filepath)
    if ret is not None:
        return ret

    if filepath.endswith('.gz'):
        ret = subprocess.run(f"gzcat {filepath} | wc -l", shell=True, stdout=subprocess.PIPE , stderr=subprocess.PIPE ,encoding="utf-8")
    elif filepath.endswith('.zst'):
        ret = subprocess.run(f"zstd -dcf {filepath} | wc -l", shell=True, stdout=subprocess.PIPE , stderr=subprocess.PIPE ,encoding="utf-8")
    else:
        ret = subprocess.run(f"wc -l {filepath}", shell=True, stdout=subprocess.PIPE , stderr=subprocess.PIPE ,encoding="utf-8")
    try:
        return int(ret.stdout)
    except:
        pass

    with zopen(filepath) as f:
        c=0
        line = f.readline()
        while line:
            c+=1
            line = f.readline()
    return c

class DummyTqdm:
    def update(self, n=1):
        pass
    def close(self):
        pass

def collator_none(s):
    return s

def collator_strip(s):
    return s.strip()

def find_collator(s):
    func = globals().get(f'collator_{s}')
    if func is None:
        patterns = [s.replace('collator_', '') for s in globals() if s.startswith('collator_')]
        raise ValueError(f'collator_{s} is not found. Select pattern from {patterns}')
    return func

def read_multilines(filenames:Union[str,List[str]], bufsize=4096, N=-1, collator = 'strip', tqdm = None):
    if isinstance(filenames, str):
        filenames = filenames.split('|')
    for filename in filenames:
        if tqdm is not None:
            N = get_linenum(filename) if N==-1 else N
            pbar = tqdm(total=N, desc=filename)
        else:
            pbar = DummyTqdm()
        collator_fn = find_collator(collator)
        buffer=[]
        with zopen(filename) as f:
            line = f.readline()
            c=0
            while line:
                buffer.append(collator_fn(line))
                c+=1
                pbar.update()
                if len(buffer) == bufsize:
                    yield buffer
                    buffer=[]
                if N != -1 and c >= N:
                    break
                line = f.readline()
            yield buffer
        pbar.close()

def filelines(filename, N=-1):
    from tqdm import tqdm
    N = get_linenum(filename) if N==-1 else N
    print('@', tqdm, N, type(N), filename)
    with tqdm(total=N, desc=filename) as pbar:
        with zopen(filename) as f:
            line = f.readline()
            c=1
            while line and c <= N:
                pbar.update()
                yield line.strip()
                line = f.readline()
                c += 1


######## OLD?


def parse_strip(s):
    return s.strip().replace('<nL>', '\n')

def parse_jsonl(line):
    d = json.loads(line)
    if 'out' in d:
        return d['in'], d['out']
    return d['text']

def file_iterator(filename, N=None, args={}):
    if N == -1:
        N = get_linenum(filename)
    if N:
        from tqdm import tqdm
        pbar = tqdm(total=N, desc=filename)
    parse_fn = parse_strip
    if '.json' in filename:
        parse_fn = parse_jsonl
    c=0
    with zopen(filename) as f:
        line = f.readline()
        while line:
            c+=1
            if N: 
                pbar.update()
                if c > N: break
            yield parse_fn(line)
            line = f.readline()
    if N:
        pbar.close()

def parse_strip_nl(line):
    return line.strip().replace('<nL>', '\n')

def parse_static_text(line):
    d = json.loads(line)
    return d['text']

class parse_text:
    def __init__(self, key='text'):
        self.key = key
    def __call__(self, line):
        d = json.loads(line)
        return d[self.key]

class parse_seq2seq:
    def __init__(self, keyin, keyout):
        self.keyin = keyin
        self.keyout = keyout
    def __call__(self, line):
        d = json.loads(line)
        return d[self.keyin], d[self.keyout]


def detect_datatype(filename:str, args: dict):
    if '.json' in filename:
        with zopen(filename) as f:
            line = f.readline()
            d = json.loads(line)
            key = get_dict_multi_keys(args, 'column|columns|content', None)
            if key:
                keys = key.split(',')
                if len(keys) == 2 and keys[0] in d and keys[1] in d:
                    args['parse_fn'] = parse_seq2seq(keys[0], keys[1])
                    return 'seq2seq'
                if keys[0] in d:
                    args['parse_fn'] = parse_text(key[0])
                    return 'text'
            if 'in' in d and 'out' in d:
                args['parse_fn'] = parse_seq2seq('in', 'out')
                return 'seq2seq'
            if 'text' in d:
                args['parse_fn'] = parse_static_text
                return 'text'
            raise ValueError('🦊 どのデータ列を使うつもりなのかな？', d)
    args['data_type'] = 'text'
    args['parse_fn'] = parse_strip_nl
    return args['data_type']

def iterate_line(filename, N=None, args={}):
    if N == -1:
        N = get_linenum(filename)
    if N:
        from tqdm import tqdm
        pbar = tqdm(total=N, desc=filename)
    parse_fn = args['parse_fn']
    c=0
    with zopen(filename) as f:
        line = f.readline()
        while line:
            c+=1
            if N: 
                pbar.update()
                if c > N: break
            yield parse_fn(line)
            line = f.readline()
    if N:
        pbar.close()

def _makedirs(path):
    dir, _,  file = path.rpartition("/")
    if '.' in file: #拡張子が含まれる場合
        os.makedirs(dir,  exist_ok=True)
    elif not os.path.isfile(path):
        os.makedirs(path,  exist_ok=True)

def get_file_sha1(filepath: str):
    # ファイルをバイナリモードで読み込む
    with open(filepath, 'rb') as f:
        # ファイルの内容を読み込む
        content = f.read()
        # SHA-1ハッシュオブジェクトを作成
        sha1 = hashlib.sha1()
        # ファイルの内容をハッシュオブジェクトに追加
        sha1.update(content)
        # 16進数でハッシュ値を取得
        sha1_hexdigest = sha1.hexdigest()
    return sha1_hexdigest

def get_filesize(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return os.path.getsize(file_path)
    else:
        return -1

def touch(file_path):
    file = Path(file_path)
    file.touch(exist_ok=True)

def compress_file(filename, compression='zst', rm=False, sync=True):
    if filename.endswith(f'.{compression}'):
        return filename
    if os.path.exists(f'{filename}.{compression}'):
        return f'{filename}.{compression}'
    if compression == 'zst':
        if rm:
            cmd = f"zstd -fq --rm {filename}"
        else:
            cmd = f"zstd -fq {filename}"
        if not sync:
            cmd = f'{cmd} &'
        subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)
        return f'{filename}.zst'
    return filename

def uncompress_file(filename, compression='zst', rm=False, sync=True):
    if not filename.endswith(f'.{compression}'):
        filename2 = f'{filename}.{compression}'
        if os.path.exists(filename2):
            filename = filename2
        else:
            return filename
    if compression == 'zst':
        unzstd_filename = filename[:-4]
        if not os.path.exists(unzstd_filename):
            if rm:
                cmd = f"zstd -dfq --rm {filename}"
            else:
                cmd = f"zstd -dfq {filename}"
            subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)
        return unzstd_filename
    return filename

def zstd_file(filename, rm=False, sync=True):
    if not os.path.exists(f'{filename}.zst'):
        if rm:
            cmd = f"zstd -fq --rm {filename}"
        else:
            cmd = f"zstd -fq {filename}"
        if not sync:
            cmd = f'{cmd} &'
        subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)
    return f'{filename}.zst'

def unzstd_file(filename, rm=False, sync=True):
    if filename.endswith('.zst'):
        unzstd_filename = filename[:-4]
        if not os.path.exists(unzstd_filename):
            if rm:
                cmd = f"zstd -dfq --rm {filename}"
            else:
                cmd = f"zstd -dfq {filename}"
            result = subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)
        return unzstd_filename
    # else:
    #     if not os.path.exists(filename) and os.path.exists(f'{filename}.zst'):
    #         return unzstd_file(f'{filename}.zst', rm=rm, sync=sync)
    return filename

def wait_for_file(file_path, timeout=60):
    """
    指定されたファイルが存在するかを定期的にチェックし、
    タイムアウトまでにファイルが見つかった場合は True を返します。
    タイムアウトした場合は False を返します。
    """
    start_time = time.time()
    end_time = start_time + timeout
    while time.time() < end_time:
        if get_filesize(file_path) > 0:
            verbose_print(f'{time.time()-start_time:.2f}秒, 待ちました')
            return True  # ファイルが見つかった
        time.sleep(0.5)  # 1秒待つ
    return False  # タイムアウト

def resolve_file(url_base, file_path, cache_dir, compressed=None, sync=True, verbose=True):
    remote_file = safe_join_path(url_base, file_path)
    # if remote_file.startswith('/'):
    #     # ローカルなファイルパスの場合
    #     return remote_file
    cached_file = safe_join_path(cache_dir, file_path)
    # ディレクトリを作っておく
    os.makedirs(cached_file.rpartition("/")[0], exist_ok=True)
    cached_file_size = get_filesize(cached_file)
    if cached_file_size > 0:
        return cached_file

    # コマンド
    temp_file = cached_file.replace('npz', 'tmp')
    if compressed:
        temp_file2 = f'{temp_file}.{compressed}'
        if remote_file.startswith('https://') or remote_file.startswith('http://'):
            cmd = f"wget -qO {temp_file2} {remote_file}.{compressed}"
        else:
            cmd = f'cp {remote_file}.{compressed} {temp_file2}'
        cmd = f"{cmd} && zstd -dfq --rm {temp_file2}"
    else:
        if remote_file.startswith('https://') or remote_file.startswith('http://'):
            cmd = f"wget -qO {temp_file} {remote_file}"
        else:
            cmd = f'cp {remote_file} {temp_file}'
    if temp_file != cached_file:
        cmd = f"{cmd} && mv {temp_file} {cached_file}"
    
    if sync:
        if cached_file_size == 0:
            #verbose_print('ダウンロード中 最大30秒待ちます.', remote_file)
            if wait_for_file(cached_file, 30):
                return cached_file
        touch(cached_file)
        subprocess.call(cmd, shell=True)
        cached_file_size = get_filesize(cached_file)
        if cached_file_size == 0:
            if verbose:
                verbose_print(f'ダウンロード失敗 file={cached_file} {format_unit(cached_file_size, scale=1024)}B', cmd)
            os.remove(cached_file)
        return cached_file

    if get_filesize(cached_file) == -1:
        touch(cached_file)
        #verbose_print('プレフェッチ', remote_file, cmd)
        subprocess.call(f"{cmd} &", shell=True, stderr=subprocess.DEVNULL)
    return None


# chunk file 

def chunkseq_to_filename(chunkseq:int, prefix:str, file_ext:str):
    dir = f"{(chunkseq//100):04d}"
    return safe_join_path(dir, f"{prefix}_{(chunkseq%100):02d}.{file_ext}")

def save_chunk_file(base_dir:str, chunk_file:str, chunks:List[np.ndarray]):
    filepath = safe_join_path(base_dir, chunk_file)
    _makedirs(filepath)
    if filepath.endswith('.npz'):
        np.savez(filepath, *chunks)

def load_chunk_file(base_dir:str, chunk_file:str=None, subblocks=1):
    if base_dir=='':
        filepath = chunk_file
    else:
        filepath = safe_join_path(base_dir, chunk_file)
    try:
        # filepath = unzstd_file(filepath)
        npz = np.load(filepath, allow_pickle=True)
        chunks = [npz[n] for n in npz.files]
        if subblocks > 1:
            newchunks=[]
            for chunk in chunks:
                splits = np.array_split(chunk, subblocks)
                newchunks.extend(splits)
            return newchunks
        return chunks
    except BaseException as e:
        verbose_print(f'チャンクファイルの破損 {filepath}: 原因 {e}')
        return None

def check_chunk_file(base_dir:str, chunk_file:str, checks: dict):
    filepath = safe_join_path(base_dir, chunk_file)
    if 'filesize' in checks:
        if get_filesize(filepath) != checks['filesize']:
            return False
    if 'sha1' in checks:
        if get_file_sha1(filepath) != checks['sha1']:
            return False
    return True

def make_chunk_filelist(base_dir:str, chunk_files:List[str]):
    d = {}
    for chunk_file in tqdm(chunk_files, desc='File validation.'):
        if not load_chunk_file(base_dir, chunk_file):
            return None
        filepath = safe_join_path(base_dir, chunk_file)
        checks = {'filesize': get_filesize(filepath), 'sha1': get_file_sha1(filepath)}
        if not check_chunk_file(base_dir, chunk_file, checks):
            verbose_print(f'broken chunk file {chunk_file}')
            return None
        d[chunk_file] = checks
    return d

def shuffle_chunk_files(store_path: str, files:List[str], random_seed=42):
    for k in range(4):
        random.shuffle(files)
        for i in tqdm(range(0, len(files)-1, 2), desc=f'turn {k}'):
            chunks = load_chunk_file(store_path, files[i])
            chunks2 = load_chunk_file(store_path, files[i+1])
            length = len(chunks)
            merged_chunks = chunks+chunks2
            random.shuffle(merged_chunks)
            save_chunk_file(store_path, files[i], merged_chunks[:length])
            save_chunk_file(store_path, files[i+1], merged_chunks[length:])

# parse urls

def parse_url_list(url_list=[]):
    if isinstance(url_list, str):
        if os.path.exists(url_list):
            with open(url_list) as f:
                return [url.strip() for url in f.readlines() if url.strip() != '' and not url.startswith('#')]
        return url_list.split('|')
    return url_list

def _convert_to_number(value):
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

def parse_url_args(url, args={}):
    parsed_url = urlparse(url)
    param_args = parse_qs(parsed_url.query)
    param_args = {k: _convert_to_number(v[0]) for k, v in param_args.items()}
    param_args['url_scheme'] = parsed_url.scheme
    param_args['url_host'] = parsed_url.netloc
    param_args['url_path'] = parsed_url.path
    if parsed_url.username:
        param_args['url_userame'] = parsed_url.username
        param_args['url_password'] = parsed_url.password
    if len(parsed_url.scheme):
        if parsed_url.port:
            param_args['url_port'] = parsed_url.port
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}:{parsed_url.port}{parsed_url.path}"
        else:
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    else:
        base_url = f"{parsed_url.path}"
        base_dir = os.path.abspath(base_url)
        if os.path.isdir(base_dir):
            base_url = base_dir
    args = args.copy()
    args.update(param_args)
    return safe_dir(base_url), args

def basename_from_url(url, ext='', prefix=''):
    if isinstance(url, (list, tuple)):
        url = url[0]
    _, _args = parse_url_args(url, {})
    base = _args['url_path']
    if '/' in base:
        _, _, base = base.rpartition('/')
    if ext:
        return f'{prefix}{base}'
    return base

'''
def read_metadata(index_file_or_url, cache_dir=None):
    if cache_dir is not None:
        index_file = resolve_file(index_file_or_url, 'kogitune.json', cache_dir)
    else:
        index_file = index_file_or_url
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            metadata = json.load(f)
            return metadata
    return {}

def write_metadata(index_file, metadata):
    with open(index_file, "w") as f:
        json.dump(metadata, f, indent=2)


def find_better_prefix(metadata, args: dict):
    if 'prefix' in args:
        return args['prefix']
    prefixes = metadata.get('prefixes', None)
    if prefixes is None: # older version
        return 'pretrain' if args.get('data_type', 'text') else 'train'
    req_max_length = args['max_length']
    selected_prefix = None
    selected_max_length = 0
    for prefix, config in prefixes.items():
        if config['data_type'] == args['data_type']:
            max_length = config.get('max_length', -1)
            if req_max_length <= max_length and max_length > selected_max_length:
                selected_max_length = max_length
                selected_prefix = prefix
    if not selected_prefix:
        for prefix, config in prefixes.items():
            if config['data_type'] == args['data_type']:
                max_length = config.get('max_length', -1)
                print(req_max_length, max_length, selected_max_length)
                if max_length >= selected_max_length:
                    selected_max_length = max_length
                    selected_prefix = prefix
        if selected_max_length > 0:
            verbose_print(f"ブロック長が小さすぎます。max_length={selected_max_length}が適切です。")
    return selected_prefix
            
def find_valid_prefix(metadata, train_prefix):
    prefixes = metadata.get('prefixes', {})
    if train_prefix.replace('train', 'valid') in prefixes:
        return train_prefix.replace('train', 'valid')
    if train_prefix.replace('train', 'dev') in prefixes:
        return train_prefix.replace('train', 'dev')
    return None

'''         


