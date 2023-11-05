import os
import time
import random
from pathlib import Path

import json
import hashlib
import subprocess
from urllib.parse import urlparse, parse_qs

from typing import List

# from collections import deque
# from filelock import FileLock

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

def get_filebase(filename):
    filebase = filename
    if '/' in filebase:
        _, _, filebase = filebase.rpartition('/')
    filebase, _, _ = filebase.partition('.')
    return filebase

def get_filename_by_pid(prefix='cache'):
    return f'{prefix}{os.getpid()}'

## file 

def zopen(filepath):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt')
    elif filepath.endswith('.zst'):
        return pyzstd.open(filepath, 'rt')
    else:
        return open(filepath, 'r')

def get_filelines(filepath):
    with zopen(filepath) as f:
        c=0
        line = f.readline()
        while line:
            c+=1
            line = f.readline()
    return c

def parse_strip(s):
    return s.strip().replace('<nL>', '\n')

def parse_jsonl(line):
    d = json.loads(line)
    if 'out' in d:
        return d['in'], d['out']
    return d['text']

def file_iterator(filename, N=None, args={}):
    if N == -1:
        N = get_filelines(filename)
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

def iterate_line(filename, N=None, args={}):
    if N == -1:
        N = get_filelines(filename)
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
            verbose_print(f'{time.time()-start_time} 秒, 待ちました')
            return True  # ファイルが見つかった
        time.sleep(0.5)  # 1秒待つ
    return False  # タイムアウト


def zstd_file(filename, rm=False, sync=True):
    if not os.path.exists(f'{filename}.zst'):
        if rm:
            cmd = f"zstd -q --rm {filename}"
        else:
            cmd = f"zstd -q {filename}"
        if not sync:
            cmd = f'{cmd} &'
        subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)
    return f'{filename}.zst'


def unzstd_file(filename, rm=False, sync=True):
    if filename.endswith('.zst'):
        unzstd_filename = filename[:-4]
        if not not os.path.exists(unzstd_filename):
            if rm:
                cmd = f"zstd -dq --rm {filename}"
            else:
                cmd = f"zstd -dq {filename}"
            if not sync:
                cmd = f'{cmd} &'
            subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)
            return unzstd_filename
    else:
        if not os.path.exists(filename) and os.path.exists(f'{filename}.zst'):
            return unzstd_file(f'{filename}.zst', rm=rm, sync=sync)
    return filename

def resolve_file(url_base, file_path, cache_dir, sync=True, verbose=True):
    remote_file = safe_join_path(url_base, file_path)
    if remote_file.startswith('/'):
        # ローカルなファイルパスの場合
        return remote_file
    cached_file = safe_join_path(cache_dir, file_path)
    # ディレクトリを作っておく
    os.makedirs(cached_file.rpartition("/")[0], exist_ok=True)
    cached_file_size = get_filesize(cached_file)
    #print('@', cached_file_size, cached_file)
    if cached_file_size > 0:
        return cached_file

    # コマンド
    if remote_file.startswith('file:'):
        remote_file = os.path.abspath(remote_file[5:]) # file: をとる
        cmd = f'cp {remote_file} {cached_file}'
    else:
        cmd = f"wget -qO {cached_file}.tmp {remote_file} && mv {cached_file}.tmp {cached_file}"

    if sync:
        if cached_file_size == 0:
            verbose_print('ダウンロード中 最大30秒待ちます.', remote_file)
            if wait_for_file(cached_file, 30):
                return cached_file
        touch(cached_file)
        subprocess.call(cmd, shell=True)
        cached_file_size = get_filesize(cached_file)
        if cached_file_size == 0:
            if verbose:
                verbose_print(f'ダウンロード失敗 file={cached_file} {cached_file_size} bytes', cmd)
            os.remove(cached_file)
        else:
            verbose_print(f'Downloaded {get_filesize(cached_file):,} bytes:', cmd)
        return cached_file

    if get_filesize(cached_file) == -1:
        touch(cached_file)
        verbose_print('プレフェッチ', remote_file)
        subprocess.call(f"{cmd} &", shell=True, stderr=subprocess.DEVNULL)
        if remote_file.endswith('.zst'):
            unzstd_file(cached_file, sync=False)
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
    # if filepath.endswith('.npz.zst'):
    #     filepath = filepath[:-4]
    #     np.savez(filepath, *chunks)
    #     zstd_file(filepath, rm=True, sync=False)
    # else:
    #     assert filepath.endswith('.npz')
    #     # np.savez_compressed(filepath, *chunks)

def load_chunk_file(base_dir:str, chunk_file:str=None, subblocks=1):
    if base_dir=='':
        filepath = chunk_file
    else:
        filepath = safe_join_path(base_dir, chunk_file)
    try:
        filepath = unzstd_file(filepath)
        npz = np.load(filepath)
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
    if len(parsed_url.scheme):
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    else:
        base_url = f"{parsed_url.path}"
    args = args.copy()
    args.update(param_args)
    return base_url, args


def read_metadata(index_file_or_url, cache_dir=None):
    if cache_dir is not None:
        index_file = resolve_file(index_file_or_url, 'index.json', cache_dir)
    else:
        index_file = index_file_or_url
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            metadata = json.load(f)
            return metadata
    return {}

def write_metadata(index_file, metadata):
    with open(index_file, "w") as f:
        json.dump(metadata, f)

def find_better_prefix(metadata, args: dict):
    if 'prefix' in args:
        return args['prefix']
    prefixes = metadata.get('prefixes', None)
    if prefixes is None: # older version
        return 'pretrain' if args.get('data_type', 'text') else 'train'
    args_max_length = args['max_length']
    selected_prefix = None
    selected_max_length = 0
    for prefix, config in prefixes.items():
        if config['data_type'] == args['data_type']:
            max_length = config.get('max_length', DEFAULT_MAX_LENGTH)
            if max_length <= args_max_length and max_length > selected_max_length:
                selected_max_length = max_length
                selected_prefix = prefix
    return selected_prefix
            
def find_valid_prefix(metadata, train_prefix):
    prefixes = metadata.get('prefixes', {})
    if train_prefix.replace('train', 'valid') in prefixes:
        return train_prefix.replace('train', 'valid')
    if train_prefix.replace('train', 'dev') in prefixes:
        return train_prefix.replace('train', 'dev')
    return None
            


