from typing import List, Union
import os
import time
import re
import random
from pathlib import Path

import json
import hashlib
import subprocess
from urllib.parse import urlparse, parse_qs

import numpy as np
import gzip
import pyzstd

from utils_tqdm import configure_progress_bar

# パス

"""
def parse_url_list(url_list=[]):
    if isinstance(url_list, str):
        if os.path.exists(url_list):
            with open(url_list) as f:
                return [url.strip() for url in f.readlines() if url.strip() != '' and not url.startswith('#')]
        return url_list.split('|')
    return url_list

def _convert_to_number(value):
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
"""

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

'''
def safe_new_file(filebase, ext, max=1000):
    filename=f'{filebase}.{ext}'
    if not os.path.exists(filename):
        return filename
    for i in range(1, max):
        filename=f'{filebase}_{i}.{ext}'
        if not os.path.exists(filename):
            break
    return filename
'''

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
    if filepath.endswith('.zst'):
        return pyzstd.open(filepath, mode)
    elif filepath.endswith('.gz'):
        return gzip.open(filepath, mode)
    else:
        return open(filepath, mode)


## linenum

fileline_pattern = re.compile(r"L(\d{3,})\D")

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


# readline

class _JSONTemplate(object):
    def __init__(self, template='{text}'):
        self.template = template
    def __call__(self, s) -> str:
        return self.template.format(**json.loads(s))

def _collator_none(s):
    return s

def _collator_strip(s):
    return s.strip()

def _collator_json(s):
    return json.loads(s)['text']

def _find_reader_fn(reader_name):
    func = globals().get(f'_reader_{reader_name}')
    if func is None:
        patterns = [s.replace('_reader_', '') for s in globals() if s.startswith('_reader_')]
        raise ValueError(f'_reader_{reader_name} is not found. Select pattern from {patterns}')
    return func

def configure_line_reader(**kwargs):
    from .adhocargs import AdhocArguments
    with AdhocArguments.from_main(**kwargs) as aargs:
        template = aargs['json_template']
        if template:
            return _JSONTemplate(template)
        reader_name = aargs['line_reader|=strip']
        return _find_reader_fn(reader_name)

def filelines(filenames:Union[str,List[str]], N=-1, json_template=None, line_reader = 'strip'):
    reader_fn = configure_line_reader(json_template=json_template, line_reader=line_reader)
    if isinstance(filenames, str):
        filenames = filenames.split('|')
    for i, filename in enumerate(filenames):
        N = get_linenum(filename) if N==-1 else N
        pbar = configure_progress_bar(total=N, desc=f'{filename}[{i+1}/{len(filenames)}]')
        with zopen(filename) as f:
            line = f.readline()
            c=0
            while line:
                line = reader_fn(line)
                c+=1
                pbar.update()
                yield line
                if N != -1 and c >= N:
                    break
                line = f.readline()
            yield line
        pbar.close()

def read_multilines(filenames:Union[str,List[str]], bufsize=4096, N=-1, json_template=None, line_reader = 'strip', tqdm = None):
    reader_fn = configure_line_reader(json_template=json_template, line_reader=line_reader)
    if isinstance(filenames, str):
        filenames = filenames.split('|')
    for i, filename in enumerate(filenames):
        N = get_linenum(filename) if N==-1 else N
        pbar = configure_progress_bar(total=N, desc=f'{filename}[{i+1}/{len(filenames)}]')
        buffer=[]
        with zopen(filename) as f:
            line = f.readline()
            c=0
            while line:
                buffer.append(reader_fn(line))
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


######## OLD?

"""
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
"""




