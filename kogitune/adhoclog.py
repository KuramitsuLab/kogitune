import os
import json

from .adhocargs import verbose_print as print

_MAX_LIMIT = 1000000
_MAIN_LOG = {}

## ログセクション

_SECTION = []

def open_section(section: str):
    _SECTION.append(section)

def get_section():
    return _SECTION[:-1] if len(_SECTION) > 0 else 'main'

def close_section():
    section = _SECTION.pop()

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

def get_log():
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

def log(section:str, key:str, data, max_limit=_MAX_LIMIT, verbose=False, message=None):
    global _MAIN_LOG
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
 
def setlog(section:str, **kwargs):
    for key, value in kwargs.items():
        log(section, key, value)

def _stringfy_kwargs(message=None, **kwargs):
    ss = []
    if message:
        ss.append(message)
    for key, value in kwargs.items():
        ss.append(f'{key}={value}')
    return ' '.join(ss)   

def notice(message: str, **kwargs):
    print(_stringfy_kwargs(message, **kwargs))
    for key, value in kwargs.items():
        log(get_section(), key, value)


def fatal(message, **kwargs):
    import sys
    log('error', 'fatal', _stringfy_kwargs(message, **kwargs), verbose=True)
    print('続けて実行できないので停止するよ')
    sys.exit(1)

def perror(message, **kwargs):
    log('error', 'error', _stringfy_kwargs(message, **kwargs), verbose=True)

def warn(message, **kwargs):
    log('error', 'warn', _stringfy_kwargs(message, **kwargs), verbose=True)

