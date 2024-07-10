import os
import sys
import time

try:
    from termcolor import colored
except:
    def colored(text, color):
        return text

### 言語

USE_JA = None

def use_ja():
    global USE_JA
    if USE_JA is None:
        USE_JA = 'ja' in os.environ.get('LANG','')
    return USE_JA

### ログ

LOGFILE_STACK = []

class LogFile(object):
    def __init__(self, filepath: str, mode='w'):
        self.filepath = filepath
        self.file = open(filepath, mode=mode)

    def __enter__(self):
        global LOGFILE_STACK
        self.file.__enter__()
        LOGFILE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global LOGFILE_STACK
        if exc_type is not None:
            pass
        LOGFILE_STACK.pop()
        return self.file.__exit__(exc_type, exc_value, traceback)

    def print(self, *args, **kwargs):
        print(*args, file=self.file, **kwargs)

def open_log_file(path: str, filename: str, mode='w'):
    os.makedirs(path, exist_ok=True)
    return LogFile(os.path.join(path, filename), mode=mode)

## フォーマット

def format_unit(num: int, scale=1000)->str:
    """
    大きな数をSI単位系に変換して返す
    """
    if scale == 1024:
        if num < scale:
            return str(num)
        elif num < scale**2:
            return f"{num / scale:.2f}K"
        elif num < scale**3:
            return f"{num / scale**2:.2f}M"
        elif num < scale**4:
            return f"{num / scale**3:.2f}G"
        elif num < scale**5:
            return f"{num / scale**4:.2f}T"
        elif num < scale**6:
            return f"{num / scale**5:.2f}P"
        else:
            return f"{num / scale**6:.2f}Exa"
    elif scale == 60:
        if num < 1.0:
            return f"{num * 1000:.3f}ms"
        day = num // (3600*24)
        num = num % (3600*24)
        hour = num // 3600
        num = num % 3600
        min = num // 60
        sec = num % 60
        if day > 0:
            return f"{day}d{hour}h{min}m{sec:.0f}s"
        elif hour > 0:
            return f"{hour}h{min}m{sec:.0f}s"
        elif min > 0:
            return f"{min}m{sec:.0f}s"
        return f"{sec:.3f}s"
    else:
        if num < 1_000:
            return str(num)
        elif num < 1_000_000:
            return f"{num / 1_000:.2f}K"
        elif num < 1_000_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num < 1_000_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        else:
            return f"{num / 1_000_000_000_000:.2f}T"

### プリント

WATCH_COUNT = {}

def aargs_print(*args, **kwargs):
    face = kwargs.pop('face', '🦊')
    if 'watch' in kwargs:
        watch_key = kwargs.pop('watch')
        c = WATCH_COUNT.get(watch_key, 5)
        if c == 0:
            return
        WATCH_COUNT[watch_key] = c - 1
        face='🔍'
#    once = kwargs.pop('once', None)
    color = kwargs.pop('color', None)
    sep = kwargs.pop('sep', ' ')
    end = kwargs.pop('end', os.linesep)
    text = sep.join(str(a) for a in args)
    if color:
        text = colored(text, color)
    print(f'{face}{text}', end=end)
    if len(LOGFILE_STACK) > 0:
        LOGFILE_STACK[-1].print(f'{face}{text}', end=end)

def list_kwargs(**kwargs):
    ss = []
    for key, value in kwargs.items():
        ss.append(f'{key}={value}')
    return ss   

def notice(*args, **kwargs):
    aargs_print(*args, *list_kwargs(**kwargs))
    sys.stdout.flush()

def warn(*args, **kwargs):
    aargs_print(colored('FIXME' 'red'), *args, *list_kwargs(**kwargs))

SAVED_LIST = []

def saved(filepath:str, desc:str, rename_from=None):
    global SAVED_LIST
    if rename_from and os.path.exists(rename_from):
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(rename_from, filepath)
    SAVED_LIST.append((filepath, desc))

def report_saved_files():
    global SAVED_LIST
    if len(SAVED_LIST) == 0:
        return
    width = max(len(filepath) for filepath, _ in SAVED_LIST) + 8
    for filepath, desc in SAVED_LIST:
        print(colored(filepath.ljust(width), 'blue'), desc)
    SAVED_LIST = []

class start_timer(object):
    """
    タイマー
    """
    def __init__(self):
        pass

    def __enter__(self):
        self.start_time = time.time()
        return self

    def notice(self, *args, iteration='total', **kwargs):
        elapsed_time = time.time() - self.start_time
        total = kwargs.get(iteration, None)
        if total is not None and total > 0:
            kwargs = dict(elapsed_time=format_unit(elapsed_time, scale=60),
                elapsed_second=round(elapsed_time,3), 
                throughput=round(elapsed_time/total,3), 
                iteration=total, **kwargs)
        else:
            kwargs = dict(
                elapsed_time=format_unit(elapsed_time, scale=60),
                elapsed_second=round(elapsed_time,3), **kwargs)
        notice(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        pass
