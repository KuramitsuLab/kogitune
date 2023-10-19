from typing import Optional
import os

def safe_dir(dir):
    if dir.endswith('/'):
        dir = dir[:-1]
    return dir

def safe_join(dir, file):
    if dir.endswith('/'):
        dir = dir[:-1]
    if file.startswith('/'):
        file = file[1:]
    return f'{dir}/{file}'

def getint(kwargs:dict, 
           key:str, default_value=0, 
           specified:Optional[int]=None) -> int:
    if isinstance(specified, int):
        return specified
    try:
        return int(kwargs.get(key, default_value))
    except:
        return default_value

def getfloat(kwargs:dict, 
             key:str, default_value=0.0,
             specified:Optional[float]=None) -> float:
    if specified:
        return float(specified)
    try:
        return float(kwargs.get(key, default_value))
    except:
        return default_value

def getint_from_environ(key, given=None, default=None):
    return getint(os.environ, key, 
                  default_value=default, 
                  specified=given)

DEFAULT_TOKENIZER = os.environ.get('KG_TOKENIZER_PATH', 'kkuramitsu/kawagoe')
DEFAULT_CACHE_DIR = safe_dir(os.environ.get('KG_CACHE_DIR', '.'))

DEFAULT_BLOCK_SIZE = getint_from_environ('KG_BLOCK_SIZE', 2048)
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
    print('🏙', *args, **kwargs)

def verbose_error(*args, **kwargs):
    """
    PaperTownのエラープリント
    """
    print('🌆', *args, **kwargs)
