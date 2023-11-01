from typing import Optional
import os

# def safe_dir(dir):
#     if dir.endswith('/'):
#         dir = dir[:-1]
#     return dir

# def safe_join(dir, file):
#     if dir.endswith('/'):
#         dir = dir[:-1]
#     if file.startswith('/'):
#         file = file[1:]
#     return f'{dir}/{file}'

def get_dict_multi_keys(d: dict, key:str, default=None, format_fn=lambda x: x):
    keys = key.split('|')
    for key in keys:
        try:
            if key in d:
                return format_fn(d[key])
        except:
            pass
    return default

def get_environ(key:str, default_value=None, param_specified=None)->str:
    if isinstance(param_specified, str):
        return param_specified
    return get_dict_multi_keys(os.environ, key, default_value)


def getint_environ(key:str, default_value=0, param_specified=None)->int:
    if isinstance(param_specified, int):
        return param_specified
    return get_dict_multi_keys(os.environ, key, default_value, format_fn=int)

DEFAULT_TOKENIZER = get_environ('KG_TOKENIZER_PATH|TOKENIZER_PATH', 'kkuramitsu/kawagoe')
DEFAULT_BLOCK_SIZE = getint_environ('KG_BLOCK_SIZE', 2048)

'''
def getint(kwargs:dict, 
           key:str, default_value=0, 
           param_specified:Optional[int]=None) -> int:
    if isinstance(param_specified, int):
        return param_specified
    try:
        return int(kwargs.get(key, default_value))
    except:
        return default_value

def getfloat(kwargs:dict, 
             key:str, default_value=0.0,
             param_specified:Optional[float]=None) -> float:
    if param_specified:
        return float(param_specified)
    try:
        return float(kwargs.get(key, default_value))
    except:
        return default_value

def getint_from_environ(key:str, 
                        default_value:Optional[int]=None, 
                        param_specified:Optional[int]=None)->int:
    return getint(os.environ, key, 
                  default_value=default_value, 
                  param_specified=param_specified)
'''

DEFAULT_MAX_LENGTH = 4096
N_CHUNKS = 4096
CHUNK_MAGIC = 8

def format_unit(num: int, scale=1000)->str:
    """
    大きな数をSI単位系に変換して返す
    """
    if scale == 1024:
        if num < scale:
            return str(num)
        elif num < scale**2:
            return f"{num / scale:.1f}K"
        elif num < scale**3:
            return f"{num / scale**2:.1f}M"
        elif num < scale**4:
            return f"{num / scale**3:.1f}G"
        elif num < scale**5:
            return f"{num / scale**4:.1f}T"
        elif num < scale**6:
            return f"{num / scale**5:.1f}P"
        else:
            return f"{num / scale**6:.1f}Exa"
    elif scale == 60:
        if num < 1.0:
            return f"{num * 1000:.1f}ms"
        if num < scale:
            return f"{num:.1f}sec"
        elif num < scale**2:
            return f"{num / scale:.1f}min"
        elif num < (scale**2)*24:
            return f"{num /(scale**2):.1f}h"
        else:
            num2 = num % (scale**2)*24
            return f"{num//(scale**2)*24}d {num2/(scale**2):.1f}h"
    else:
        if num < 1_000:
            return str(num)
        elif num < 1_000_000:
            return f"{num / 1_000:.1f}K"
        elif num < 1_000_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num < 1_000_000_000_000:
            return f"{num / 1_000_000_000:.1f}B"
        else:
            return f"{num / 1_000_000_000_000:.1f}T"


def verbose_print(*args, **kwargs):
    """
    PaperTown 用のデバッグプリント
    """
    print('🦊', *args, **kwargs)

def verbose_error(*args, **kwargs):
    """
    PaperTownのエラープリント
    """
    print('💣', *args, **kwargs)
