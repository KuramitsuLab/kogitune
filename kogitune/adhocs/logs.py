import sys
import os
import json
import time

from .stacks import get_section, aargs_print
from .formats import format_unit

## ログ用のセクション


# ログ本体

_MAX_LIMIT = 1000000
_MAIN_LOG = {}

def _check_logdata(data):
    if isinstance(data, (int, float, str, bool)) or data is None:
        return data
    elif isinstance(data, dict):
        d={}
        for key, value in data.items():
            if isinstance(value, list) and len(value) == 1:
                d[key] = _check_logdata(value[0])
            else:
                d[key] = _check_logdata(value)
        return d
    elif isinstance(data, (list, tuple)):
        return [_check_logdata(x) for x in data]
    else:
        return f'{data}' # stringfy

def get_log(section=None):
    return _check_logdata(_MAIN_LOG)

def save_log(output_path):
    global _MAIN_LOG
    log = _check_logdata(_MAIN_LOG)
    if '.json' in output_path:
        output_filename = output_path.replace('.json', '_log.json')
    else:
        output_filename = os.path.join(output_path, '_log.json')
    with open(output_filename, 'w') as w:
        json.dump(log, fp=w, ensure_ascii=False, indent=2)
    _MAIN_LOG = {}

def log(section:str, key:str, data, 
        max_limit=_MAX_LIMIT, verbose=False, message=None):
    global _MAIN_LOG
    if section is None:
        section = get_section()
    if section not in _MAIN_LOG:
        _MAIN_LOG[section] = {}
    log = _MAIN_LOG[section]
    if key in log:
        values = log[key]
        if len(values) < max_limit:
            values.append(data)
    else:
        log[key] = [data]
    if message:
        print(message, data)
    elif verbose:
        print('確認してね！', data)


def get_identifier(method):
    name = str(method)
    if 'method ' in name and ' of' in name:
        _, _, name = name.partition('method ')
        name, _, _ = name.partition(' of')
        return name
    if "class '" in name and "'>" in name:
        _, _, name = name.partition("class '")
        name, _, _ = name.partition("'>")
        if '.' in name:
            _,_,name = name.partition('.')
        return name
    if 'function ' in name and ' at' in name:
        _, _, name = name.partition('function ')
        name, _, _ = name.partition(' at')
        return name
    
    return name

def log_args(function_or_method, version:str, path:str, args:dict):
    name = get_identifier(function_or_method)
    d = dict(
        name = name,
        version = version, 
        path = path,
        args = args,
    )
    log('arguments', name, d)

def setlog(_section, **kwargs):
    if _section is None:
        _section = get_section()
    for key, value in kwargs.items():
        log(_section, key, value)

# def _stringfy_kwargs(_message=None, **kwargs):
#     ss = []
#     if _message:
#         ss.append(_message)
#     for key, value in kwargs.items():
#         ss.append(f'{key}={value}')
#     return ' '.join(ss)   

def notice(_message: str, **kwargs):
    section = get_section()
    for key, value in kwargs.items():
        log(section, key, value)
    aargs_print(_message, verbose=section, **kwargs)

def warn(**kwargs):
    import re
    exit_at_end = False
    ss=[]
    for key, value in kwargs.items():
        if key.startswith('unknown_'):
            key = key[8:]
            ss.append(f'知らない値を設定しちゃったね. {key}={repr(value)}')
            exit_at_end = True
        elif key.startswith('unset_key'):
            ss.append(f'{value}が設定してないよ')
            exit_at_end = True
        elif key.startswith('key_error'):
            try:
                matches = re.findall(r"'(.*?)'",  f'{value}')
                key = matches[0]
                ss.append(f'テンプレートに{key}がないよ')
            except:
                ss.append(f'テンプレートにキーがないよ')
            exit_at_end = True
        elif key.startswith('expected'):
            ss.append(f'\nどれか選んでね. {value}')
        elif key.startswith('default_'):
            key = key[8:]
            ss.append(f'とりあえず、{key}={repr(value)}としておくよ')
            exit_at_end = False
        else:
            ss.append(f'{key}={repr(value)}')
    _message = ' '.join(ss)   
    aargs_print(_message)
    if exit_at_end:
        aargs_print('実行が続けられないので停止するよ', color='red')
        sys.exit(1)


_TIME = {}

def start_time(key: str):
    global _TIME
    _TIME[key] = time.time()

def end_time(key: str, message=None, total=None):
    if key not in _TIME:
        return
    elapsed_time = time.time() - _TIME[key]
    logdata = dict(key = key, 
                    elapsed_time=round(elapsed_time,3))
    if total is None and total != 0:
        logdata['throughput'] = round(elapsed_time/total,3)
    if message:
        notice(message, **logdata)
    else:
        notice(f'実行時間[{key}] {format_unit(elapsed_time, scale=60)}', **logdata)
    #setlog('time', section=get_section(), **logdata)
