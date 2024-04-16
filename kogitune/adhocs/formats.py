import os

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
            return f"{num / 1_000:.2}K"
        elif num < 1_000_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num < 1_000_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        else:
            return f"{num / 1_000_000_000_000:.2f}T"

QUOTE = '"""'

def _repr(value, indent=0): # 再帰的な繰り返し
    if isinstance(value, str):
        if value.count('\n') < 2:
            return repr(value)
        lines = []
        if value.startswith('\n'):
            lines.append(QUOTE)
        else:
            lines.append(f"{QUOTE}\\")
            value = f'{os.linesep}{value}'
        spc = (' '*(indent+2))
        lines.append(value.replace('\n', '\n' + spc))
        lines.append(os.linesep + spc + QUOTE)
        return ''.join(lines)
    elif isinstance(value, float):
        return round(value, 3)
    elif isinstance(value, dict) or hasattr(value, 'items'):
        lines=[]
        lines.append('{')
        indent_space=' ' * (indent+2)
        for k, v in value.items():
            lines.append(f'{indent_space}"{k}": {_repr(v, indent+2)},')
        lines.append((' ' * indent) + '}')
        return os.linesep.join(lines)
    elif isinstance(value, (list, tuple)):
        s = f'{value}'
        if len(s) < 80:
            return s
        lines=[]
        lines.append('[')
        indent_space=' ' * (indent+2)
        for i, v in enumerate(value):
            lines.append(f'{indent_space}{_repr(v, indent+2)},')
        lines.append((' ' * indent) + ']')
        return os.linesep.join(lines)        
    elif isinstance(value, bool):
        return str(value).lower()
    elif value == None:
        return 'null'
    else:
        return f'{value}'

def _stringfy_kwargs(_message=None, **kwargs):
    ss = []
    if _message:
        ss.append(_message)
    for key, value in kwargs.items():
        ss.append(f'{key}={value}')
    return ' '.join(ss)   

def format_print_args(*args, **kwargs):
    sep = kwargs.pop('sep', ' ')
    kwargs.pop('end', '\n')
    if len(kwargs) > 0:
        args = args + (kwargs,)
    return sep.join(_repr(v) for v in args)

_BUF = []
 
def p_buf(*args, **kwargs):
    global _BUF
    _BUF.append(format_print_args(*args, **kwargs)+kwargs.get('end', os.linesep))
    if kwargs.get('flush_buf', False):
        return flush_buf()

def flush_buf():
    s = ''.join(_BUF)
    _BUF =[]
    return s
