from typing import List, Union
import os
import re

import json
import subprocess

import gzip
import pyzstd

import kogitune.adhocs as adhoc

# ファイルシステム

def list_filenames(filenames) -> List[str]:
    if isinstance(filenames, str):
        filenames = filenames.split('|')
    return filenames

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

def basename(path:str, skip_dot=False):
    if '?' in path:
        path, _, _ = path.partition('?')
    if '/' in path:
        _, _, path = path.rpartition('/')
    if '\\' in path:
        _, _, path = path.rpartition('\\')
    if not skip_dot and '.' in path:
        path, _, _ = path.partition('.')
    return path

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
        self.test_mode = True
    def __call__(self, s) -> str:
        if self.test_mode:
            self.test_mode = False
            try:
                sample = json.loads(s)
                return self.template.format(**sample)
            except KeyError as e:
                adhoc.warn(key_error=e, template=self.template, sample=sample)
        return self.template.format(**json.loads(s))

def reader_strip(s):
    return s.strip()

def reader_json(s):
    return json.loads(s)['text']

def reader_jsonl(s):
    return json.loads(s)['text']

def _find_reader_fn(reader_name):
    func = globals().get(f'reader_{reader_name}')
    if func is None:
        patterns = [s.replace('reader_', '') for s in globals() if s.startswith('reader_')]
        adhoc.warn(unknown_line_reader=reader_name, 
                     expected=patterns, default_line_reader='strip')
        return reader_strip
    return func

def adhoc_line_reader(**kwargs):
    with adhoc.from_kwargs(**kwargs) as aargs:
        template = aargs['json_template']
        if template:
            return _JSONTemplate(template)
        reader_name = aargs['line_reader|=strip']
        return _find_reader_fn(reader_name)

def filelines(filenames:Union[str,List[str]], N=-1, json_template=None, line_reader = 'strip'):
    reader_fn = adhoc_line_reader(json_template=json_template, line_reader=line_reader)
    if isinstance(filenames, str):
        filenames = filenames.split('|')
    for i, filename in enumerate(filenames):
        N = get_linenum(filename) if N==-1 else N
        pbar = adhoc.progress_bar(total=N, desc=f'{filename}[{i+1}/{len(filenames)}]')
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

def read_multilines(filenames:Union[str,List[str]], bufsize=4096, N=-1, json_template=None, line_reader = 'strip'):
    if isinstance(filenames, str):
        filenames = filenames.split('|')
    if filenames[0].endswith('.jsonl'):
        line_reader='jsonl'
    reader_fn = adhoc_line_reader(json_template=json_template, line_reader=line_reader)
    for i, filename in enumerate(filenames):
        N = get_linenum(filename) if N==-1 else N
        pbar = adhoc.progress_bar(total=N, desc=f'{filename}[{i+1}/{len(filenames)}]')
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

