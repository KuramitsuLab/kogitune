import os
import time
import random
from pathlib import Path

import json
import hashlib
import subprocess
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

_ID = 0

def random_name():
    global _ID
    _ID+= 1
    return f'Cache{_ID-1}'

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

## file 

def zopen(filepath):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt')
    elif filepath.endswith('.zstd'):
        return pyzstd.open(filepath, 'rt')
    else:
        return open(filepath, 'r')

def get_filelines(filepath):
    with zopen(filepath) as f:
        line = f.readline()
        c=1
        while line:
            line = f.readline()
            c+=1
    return c

def parse_strip(s):
    return s.strip().replace('<nL>', '\n')

def parse_jsonl(line):
    d = json.loads(line)
    if 'out' in d:
        return f"{d['in']}<outpuT>{d['out']}"
    return d['text']

def file_iterator(filename, N=None, kwargs={}):
    if N == -1:
        N = get_filelines(filename)-1
    if N:
        from tqdm import tqdm
        pbar = tqdm(total=N, desc=filename)
    parse_fn = parse_strip
    if '.json' in filename:
        parse_fn = parse_jsonl
    c=0
    with zopen(filename) as f:
        line = f.readline()
        while line is not None:
            c+=1
            if N: 
                pbar.update()
                if c > N: break
            yield parse_fn(line)
            line = f.readline()
    if N:
        pbar.close()



def _remove_heading_nL(s):
    while s.startswith('<nL>'):
        s = s[4:]
    return s

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
        time.sleep(1)  # 1秒待つ
    return False  # タイムアウト

def resolve_file(url_base, file_path, cache_dir, sync=True):
    remote_file = safe_join_path(url_base, file_path)
    if remote_file.startswith('/'):
        # ローカルなファイルパスの場合
        return remote_file
    cache_file = safe_join_path(cache_dir, file_path)
    # ディレクトリを作っておく
    os.makedirs(cache_file.rpartition("/")[0], exist_ok=True)
    cache_file_size = get_filesize(cache_file)
    #print('@', cache_file_size, cache_file)
    if cache_file_size > 0:
        return cache_file

    # ダウンロードコマンド
    if remote_file.startswith('file:'):
        remote_file = os.path.abspath(remote_file[5:]) # file: をとる
        cmd = f'cp {remote_file} {cache_file}'
    else:
        cmd = f"wget -qO {cache_file}.tmp {remote_file} && mv {cache_file}.tmp {cache_file}"

    if sync:
        if cache_file_size == 0:
            verbose_print('ダウンロード中 最大30秒待ちます.', remote_file)
            if wait_for_file(cache_file, 30):
                return cache_file
        touch(cache_file)
        subprocess.call(cmd, shell=True)
        verbose_print(f'Downloaded {get_filesize(cache_file):,} bytes:', cmd)
        return cache_file

    if get_filesize(cache_file) == -1:
        touch(cache_file)
        verbose_print('プレフェッチ', remote_file)
        subprocess.call(f"{cmd} &", shell=True, stderr=subprocess.DEVNULL)
    # else:
    #     verbose_print('既にダウンロード中..', remote_file)
    return None


# chunk file 

def chunkseq_to_filename(chunkseq:int, prefix:str, file_ext:str):
    dir = f"{(chunkseq//100):04d}"
    return f"{dir}/{prefix}{(chunkseq%100):02d}.{file_ext}"

def save_chunk_file(base_dir:str, chunk_file:str, chunks:List[np.ndarray]):
    filepath = safe_join_path(base_dir, chunk_file)
    _makedirs(filepath)
    if filepath.endswith('.npz'):
        np.savez_compressed(filepath, *chunks)
    # return {'filesize': get_filesize(filepath), 
    #         'sha1': get_file_sha1(filepath)}

def load_chunk_file(base_dir:str, chunk_file:str=None):
    filepath = safe_join_path(base_dir, chunk_file)
    try:
        #if filepath.endswith('.npz'):
        npz = np.load(filepath)
        return [npz[n] for n in npz.files]
    except BaseException as e:
        verbose_print(f'broken chunk file {chunk_file}: {e}')
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
    for chunk_file in tqdm(chunk_files, desc='file validation..'):
        if not load_chunk_file(base_dir, chunk_file):
            return None
        filepath = safe_join_path(base_dir, chunk_file)
        checks = {'filesize': get_filesize(filepath), 'sha1': get_file_sha1(filepath)}
        if not check_chunk_file(base_dir, chunk_file, checks):
            verbose_print(f'broken chunk file {chunk_file}')
            return None
        d[chunk_file] = checks
    return d

def shuffle_chunk_files(base_dir:str, chunk_file:str, chunk_file2:str):
    assert chunk_file != chunk_file2
    chunks = load_chunk_file(base_dir, chunk_file)
    chunks2 = load_chunk_file(base_dir, chunk_file2)
    length = len(chunks)
    merged_chunks = chunks+chunks2
    random.shuffle(merged_chunks)
    save_chunk_file(base_dir, chunk_file, merged_chunks[:length])
    save_chunk_file(base_dir, chunk_file, merged_chunks[length:])
