import re
import inspect
from .dicts import find_simkey
# from .OLDlogs import log_args

def get_version(class_or_function):
    module_name = class_or_function.__module__
    if '.' in module_name:
        module_name, _, _ = module_name.partition('.')
    # モジュールをインポートする
    module = __import__(module_name)
    # モジュールのバージョン情報を取得
    version = getattr(module, '__version__', '(unknown version)')
    return f'{module_name}: {version}'

def typename(annotation):
    name = str(annotation)
    if name.startswith('<class'):
        return annotation.__name__
    return name.replace('typing.', '')

def classname(function_or_method):
    text = str(function_or_method)
    # 正規表現を用いてクラス名を取り出す
    pattern = r"<class '([^']+)'"
    match = re.search(pattern, text)
    if match:
        class_name = match.group(1).split('.')[-1]  # フルパスからクラス名のみを取り出す
        return class_name
    else:
        return None

def has_VAR_KEYWORD(function_or_method):
    signature = inspect.signature(function_or_method)
    for _, param in signature.parameters.items():
        if param.kind == param.VAR_KEYWORD:
            return True
    return False

def get_parameters(function_or_method, default_only=True):
    signature = inspect.signature(function_or_method)
    parameters = {}
    for name, param in signature.parameters.items():
        if default_only and param.default != param.empty:
            parameters[name] = param.default
        else:
            d = {'kind': str(param.kind)}
            if param.annotation != param.empty:
                d['type'] = typename(param.annotation)
            if param.default != param.empty:
                d['default'] = param.default
            parameters[name]=d
    return parameters

def extract_kwargs(function_or_method, kwargs: dict, excludes=[], use_simkey=True):
    params = get_parameters(function_or_method)
    has_var_keyword = has_VAR_KEYWORD(function_or_method)
    #print('@', params)
    new_kwargs={}
    for key in list(kwargs.keys()):
        if key in excludes:
            continue
        if key in params or has_var_keyword:
            new_kwargs[key] = kwargs[key]
            continue
        simkey = find_simkey(params, key, max_distance=(len(key)/4)+1)
        if use_simkey and simkey:
            print('@typo', key, simkey)
            new_kwargs[key] = kwargs[simkey]
    #print('@', new_kwargs)
    return new_kwargs


def check_kwargs(kwargs: dict, function_or_method, path=None,
                 excludes=[]):
    params = get_parameters(function_or_method)
    dropped=[]
    for key in list(kwargs.keys()):
        if key in excludes:
            kwargs.pop(key)
            continue
        if key in params:
            continue
        simkey = find_simkey(params, key, max_distance=(len(key)/4)+1)
        if simkey:
            print('@typo', key, simkey)
            kwargs[simkey] = kwargs.pop(key)
        else:
            dropped.append(key)
    if len(dropped) > 0 and not has_VAR_KEYWORD(function_or_method):
        print('@drop {function_or_method}', dropped)
        for key in dropped:
            del kwargs[key]
    # if path:
    #     log_args(function_or_method, 
    #              get_version(function_or_method),
    #              path, kwargs)
    