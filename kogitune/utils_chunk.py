from typing import List, Union
import os
import time
import random
from pathlib import Path

import hashlib
import subprocess

import numpy as np

from .adhoc_args import verbose_print

from .utils_file import safe_join_path
from kogitune.configurable_tqdm import configurable_tqdm

N_CHUNKS = 4096

def safe_makedirs(path):
    dir, _,  file = path.rpartition("/")
    if '.' in file: #拡張子が含まれる場合
        os.makedirs(dir,  exist_ok=True)
    elif not os.path.isfile(path):
        os.makedirs(path,  exist_ok=True)

def chunkseq_to_filename(chunkseq:int, prefix:str, file_ext:str):
    dir = f"{(chunkseq//100):04d}"
    return safe_join_path(dir, f"{prefix}_{(chunkseq%100):02d}.{file_ext}")

def save_chunk_file(base_dir:str, chunk_file:str, chunks:List[np.ndarray]):
    filepath = safe_join_path(base_dir, chunk_file)
    safe_makedirs(filepath)
    if filepath.endswith('.npz'):
        np.savez(filepath, *chunks)

def load_chunk_file(base_dir:str, chunk_file:str=None, subblocks=1):
    if base_dir=='':
        filepath = chunk_file
    else:
        filepath = safe_join_path(base_dir, chunk_file)
    try:
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

def get_filesha1(filepath: str):
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

def check_chunk_file(base_dir:str, chunk_file:str, checks: dict):
    filepath = safe_join_path(base_dir, chunk_file)
    if 'filesize' in checks:
        if get_filesize(filepath) != checks['filesize']:
            return False
    if 'sha1' in checks:
        if get_filesha1(filepath) != checks['sha1']:
            return False
    return True

def make_chunk_filelist(base_dir:str, chunk_files:List[str]):
    d = {}
    for chunk_file in configurable_tqdm(chunk_files, desc='File validation.'):
        if not load_chunk_file(base_dir, chunk_file):
            return None
        filepath = safe_join_path(base_dir, chunk_file)
        checks = {'filesize': get_filesize(filepath), 'sha1': get_filesha1(filepath)}
        if not check_chunk_file(base_dir, chunk_file, checks):
            verbose_print(f'broken chunk file {chunk_file}')
            return None
        d[chunk_file] = checks
    return d

def shuffle_chunk_files(store_path: str, files:List[str], random_seed=42):
    for k in range(4):
        random.shuffle(files)
        for i in configurable_tqdm(range(0, len(files)-1, 2), desc=f'turn {k}'):
            chunks = load_chunk_file(store_path, files[i])
            chunks2 = load_chunk_file(store_path, files[i+1])
            length = len(chunks)
            merged_chunks = chunks+chunks2
            random.shuffle(merged_chunks)
            save_chunk_file(store_path, files[i], merged_chunks[:length])
            save_chunk_file(store_path, files[i+1], merged_chunks[length:])

def check_command_installed(command="zstd", verbose=False):
    try:
        # 'command --version' コマンドを実行してみる
        result = subprocess.run([command, "--version"], capture_output=True, text=True, check=True)
        # コマンドの実行に成功した場合、command はインストールされている
        return True
    except subprocess.CalledProcessError as e:
        # コマンドの実行に失敗した場合、command はインストールされていないか、別の問題がある
        if verbose:
            verbose_print(f'{command}コマンドが使えないよ！ 詳細: {e.stderr}')
        return False
    except FileNotFoundError:
        if verbose:
            verbose_print(f'{command}コマンドがないよ！')
        return False

check_command_installed('wget', verbose=True)
check_command_installed('zstd', verbose=True)

def compress_file(filename, compressed='zst', rm=False, sync=True):
    if filename.endswith(f'.{compressed}'):
        return filename
    if os.path.exists(f'{filename}.{compressed}'):
        return f'{filename}.{compressed}'
    if compressed == 'zst':
        if rm:
            cmd = f"zstd -fq --rm {filename}"
        else:
            cmd = f"zstd -fq {filename}"
        if not sync:
            cmd = f'{cmd} &'
        subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)
        return f'{filename}.zst'
    return filename

def uncompress_file(filename, compressed='zst', rm=False, sync=True):
    if not filename.endswith(f'.{compressed}'):
        filename2 = f'{filename}.{compressed}'
        if os.path.exists(filename2):
            filename = filename2
        else:
            return filename
    if compressed == 'zst':
        unzstd_filename = filename[:-4]
        if not os.path.exists(unzstd_filename):
            if rm:
                cmd = f"zstd -dfq --rm {filename}"
            else:
                cmd = f"zstd -dfq {filename}"
            subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL)
        return unzstd_filename
    return filename

"""OLD?
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
"""

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
            verbose_print(f'{time.time()-start_time:.2f}秒, 待ちました')
            return True  # ファイルが見つかった
        time.sleep(0.5)  # 1秒待つ
    return False  # タイムアウト

def resolve_file(url_base, file_path, cache_dir, compressed=None, sync=True, verbose=True):
    remote_file = safe_join_path(url_base, file_path)
    cached_file = safe_join_path(cache_dir, file_path)
    # ディレクトリを作っておく
    os.makedirs(cached_file.rpartition("/")[0], exist_ok=True)
    cached_file_size = get_filesize(cached_file)
    if cached_file_size > 0:
        return cached_file

    # コマンド
    if compressed == 'zst':
        temp_file = cached_file.replace('npz', 'tmp')
        if remote_file.startswith('https://') or remote_file.startswith('http://'):
            cmd = f"wget -qO {temp_file}.zst {remote_file}.zst && zstd -dfq --rm {temp_file}.zst && mv {temp_file} {cached_file}"
        else:
            cmd = f"zstd -dfq {remote_file}.zst && mv {remote_file} {cached_file}"
    else:
        temp_file = cached_file.replace('npz', 'tmp')
        if remote_file.startswith('https://') or remote_file.startswith('http://'):
            cmd = f"wget -qO {temp_file} {remote_file} && mv {temp_file} {cached_file}"
        else:
            cmd = f"cp {remote_file} {cached_file}"
    
    if sync:
        if cached_file_size == 0:
            #verbose_print('ダウンロード中 最大30秒待ちます.', remote_file)
            if wait_for_file(cached_file, 30):
                return cached_file
        touch(cached_file)
        result=subprocess.call(cmd, shell=True)
        cached_file_size = get_filesize(cached_file)
        if cached_file_size == 0:
            if verbose:
                verbose_print(f'ダウンロード失敗 file={cached_file} by', cmd)
            os.remove(cached_file)
        return cached_file

    if get_filesize(cached_file) == -1:
        touch(cached_file)
        #verbose_print('プレフェッチ', remote_file, cmd)
        subprocess.call(f"{cmd} &", shell=True, stderr=subprocess.DEVNULL)
    return None
