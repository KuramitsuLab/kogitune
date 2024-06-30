import os

try:
    from termcolor import colored
except:
    def colored(text, color):
        return text

USE_JA = None

def use_ja():
    global USE_JA
    if USE_JA is None:
        USE_JA = 'ja' in os.environ.get('LANG','')
    return USE_JA

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
    once = kwargs.pop('once', None)
    color = kwargs.pop('color', None)
    sep=kwargs.pop('sep', ' ')
    end=kwargs.pop('end', os.linesep)
    # flush = kwargs.pop('flush_buf', False)
    # if once:
    #     value = f'{args[0]}' if once == True else once
    #     if value in _PRINT_ONCE:
    #         return
    #     _PRINT_ONCE.add(value)
    text = sep.join(str(a) for a in args)
    if color:
        text = colored(text, color)
    print(f'{face}{text}', end=end)
    # if flush:
    #     print(flush_buf())
