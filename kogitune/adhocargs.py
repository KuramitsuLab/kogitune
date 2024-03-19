from typing import List, Optional
import os
import sys
import json
import re
import inspect
from urllib.parse import urlparse, parse_qs

def parse_argument_value(key:str, value:str):
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    if key.startswith('_list') and value.endsith('.txt'):
        return load_text_list(value)
    return value

def parse_path_arguments(url_or_filepath: str, include_urlinfo=False, global_args={}):
    """
    pathから引数を読み込む
    """
    if url_or_filepath.startswith('{') and url_or_filepath.startswith('}'):
        ## JSON形式であれば、最初のパラメータはパス名、残りは引数として解釈する。
        args = json.loads(url_or_filepath)
        first_key = list(args.keys())[0]
        return args.pop(first_key), global_args | args

    parsed_url = urlparse(url_or_filepath)
    options = parse_qs(parsed_url.query)
    args = {k: parse_argument_value(k, v[0]) for k, v in options.items()}
    if len(parsed_url.scheme):
        if parsed_url.port:
            url = f"{parsed_url.scheme}://{parsed_url.netloc}:{parsed_url.port}{parsed_url.path}"
        else:
            url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    else:
        url = f"{parsed_url.path}"
    if include_urlinfo:
        args['url_scheme'] = parsed_url.scheme
        args['url_host'] = parsed_url.netloc
        if parsed_url.username:
            args['userame'] = parsed_url.username
            args['password'] = parsed_url.password
        if parsed_url.port:
            args['port'] = parsed_url.port
        args['path'] = parsed_url.path
    return url, global_args | args

_key_pattern = re.compile(r'^[A-Za-z0-9\.\-_]+\=')

def _parse_key_value(key, next_value, args):
    if _key_pattern.match(key):
        if key.startswith('--'):
            key = key[2:]
        key, _, value = key.partition('=')
        return key, parse_argument_value(key, value)     
    elif key.startswith('--'):
        key = key[2:]
        if next_value.startswith('--'):
            if key.startswith('enable_') or key.startswith('enable-'):
                return key[7:], True
            elif key.startswith('disable_') or key.startswith('disable-'):
                return key[8:], False
            return key, True
        else:
            args['_'] = next_value
            return key, parse_argument_value(key, next_value)
    else:
        if args.get('_') != key:
            files = args.get('files', [])
            files.append(key)
            args['files'] = files
    return key, None

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
            return f"{num * 1000:.2f}ms"
        day = num // (3600*24)
        num = num % (3600*24)
        hour = num // 3600
        num = num % 3600
        min = num // 60
        sec = num % 60
        if day > 0:
            return f"{day}d {hour}h {min}m {sec}s"
        elif hour > 0:
            return f"{hour}h {min}m {sec}s"
        elif min > 0:
            return f"{min}m {sec}s"
        return f"{sec}s"
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

## ファイル

def get_basename_from_filepath(filepath:str)->str:
    """
    ファイルパスからベースの名前を取り出す
    """
    filebase = filepath
    if '/' in filebase:
        _, _, filebase = filebase.rpartition('/')
    if '\\' in filebase:
        _, _, filebase = filebase.rpartition('\\')
    if '_L' in filebase:
        left, _, right = filebase.rpartition('_L')
        if right[0].isdigit():
            filebase = left
    filebase, _, _ = filebase.partition('.')
    return filebase

## コンフィグファイル

def load_yaml(config_file):
    import yaml
    loaded_data = {}
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        for section, settings in config.items():
            if isinstance(settings, dict):
                for key, value in settings.items():
                    loaded_data[key] = value
        return loaded_data

def load_json(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)

def load_config(config_file):
    if config_file.endswith('.json'):
        return load_json(config_file)
    if config_file.endswith('.yaml'):
        return load_yaml(config_file)
    return {}

def load_text_list(list_file):
    with open(list_file) as f:
        return [line.strip() for line in f.readlines() if line.strip() != '' and not line.startswith('#')]

# main adhoc arguments

main_aargs = None

def main_adhoc_arguments():
    global main_aargs
    if main_aargs is None:
        main_aargs = AdhocArguments({})
    return main_aargs

def verbose_print(*args, **kwargs):
    aargs = main_adhoc_arguments()
    aargs.verbose_print(*args, **kwargs)



# main


class AdhocArguments(object):
    """
    アドホックな引数パラメータ
    """
    def __init__(self, 
                 args:dict, parent=None, expand_config='config', use_environ=True, 
                 face='🦊'):
        self._args = {}
        self._used_keys = set()
        self._use_environ = use_environ
        self.face = face
        self.parent = parent
        if parent:
            self._use_environ = False
            self.face = parent.face
        for key, value in args.items():
            if key == expand_config:
                self.load_config(value)
            else:
                self._args[key] = value

    def __repr__(self):
        if self.parent:
            return f'{self._args}+{self.parent}'
        return repr(self._args)

    def used(self, key):
        self._used_keys.add(key)
        if self.parent:
            self.parent.used(key)

    def get(self, keys, default_value=None):
        keys = keys.split('|')
        default_key = keys[0]
        for key in keys:
            if key in self._args:
                self.used(key)
                return self._args[key]
            if key.startswith('='):
                return parse_argument_value(default_key, key[1:])
            if key.startswith('!'):
                if key.startswith('!!'):
                    return self.raise_error(default_key, key[2:])
                return self.warn_unset_key(default_key, parse_argument_value(default_key, key[1:]))
            if self.parent and key in self.parent:
                return self.parent[key]
            if self._use_environ:
                    environ_key = key.upper()
                    if environ_key in os.environ:
                        value = parse_argument_value(key, os.environ[environ_key])
                        self._args[key] = value
                        self.used(key)
                        return value
        return default_value

    def copy_to(self, keys: str, newargs: dict):
        keys = keys.split('|')
        default_key = keys[0]
        for key in keys:
            if key in self:
                newargs[default_key] = self[key]
                return

    def __getitem__(self, key):
        return self.get(key, None)

    def __setitem__(self, key, value):
        self._args[key] = value
        # setattr(self, key, value)
        self.used(key)

    def __contains__(self, key):
        return key in self._args or (self.parent and key in self.parent) or (self._use_environ and key.upper() in os.environ)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            for key, value in self._args.items():
                if key not in self._used_keys:
                    raise TypeError(f'{key} is an unused keyword')

    @classmethod
    def from_main(cls, **kwargs):
        if 'aargs' in kwargs and isinstance(kwargs['aargs'], AdhocArguments):
            aargs = kwargs.pop('aargs')
        else:
            aargs = main_adhoc_arguments()
        return AdhocArguments(kwargs, parent=aargs)

    def from_kwargs(self, **kwargs):
        return AdhocArguments(kwargs, parent=self)

    def as_kwargs(self, **kwargs):
        kwargs = kwargs.copy()
        aargs = self
        while aargs is not None:
            for key, value in aargs._args.items():
                if key not in kwargs:
                    kwargs[key] = value
            aargs = aargs.parent
        return kwargs


    def update(self, otherdict:dict, overwrite=True, used=True):
        for key, value in otherdict.items():
            if overwrite or key not in self._args:
                self._args[key] = value
                if used:
                    self.used(key)

    def load_config(self, config_file, merge=True, overwrite=True):
        loaded_data = load_config(config_file)
        if merge:
            self.update(loaded_data, overwrite=overwrite)
        return loaded_data

    def subset(self, keys='', prefix=None):
        subargs = {}
        keys = set(keys.split('|'))
        for key, value in self._args.items():
            if key in keys:
                self._used_keys.add(key)
                subargs[key] = value
            elif prefix and key.startswith(prefix):
                self._used_keys.add(key)
                key = key[len(prefix):]
                if key.startswith('_'):
                    key = key[1:]
                subargs[key] = value
        return subargs

    def get_subargs(self, keys, exclude=None):
        subargs = {}
        for key in keys.split('|'):
            if key.endswith('*'):
                prefix = key[:-1]
                for key, value in self._args.items():
                    if key.startswith(prefix):
                        self._used_keys.add(key)
                        key = key[len(prefix):]
                        subargs[key] = value
            elif key in self:
                subargs[key] = self[key]
        if exclude is not None:
            for key in exclude.split('|'):
                if key in subargs:
                    del subargs[key]
        return subargs

    def find_options(self, prefix: str, namespace: dict = None):
        if namespace is None:
            # 呼び出し元のフレームを取得
            caller_frame = inspect.stack()[1].frame
            # 呼び出し元のグローバル変数の名前空間を取得
            namespace = caller_frame.f_globals
        return [s.replace(f'{prefix}_', '') for s in globals() if s.startswith(f'{prefix}_')]

    def find_function(self, option:str, prefix: str, namespace: dict = None):
        if namespace is None:
            # 呼び出し元のフレームを取得
            caller_frame = inspect.stack()[1].frame
            # 呼び出し元のグローバル変数の名前空間を取得
            namespace = caller_frame.f_globals
        func = namespace.get(f'{prefix}_{option}')
        if func is None:
            patterns = [s.replace(f'{prefix}_', '') for s in globals() if s.startswith(f'{prefix}_')]
            raise ValueError(f'{prefix}_{option} is not found. Select pattern from {patterns}')
        return func

    def check_unused(self):
        show_notion = True
        for key, value in self._args.items():
            if key not in self._used_keys:
                if show_notion:
                    self.print(f'未使用のパラメータ一覧//List of unused parameters')
                    show_notion = False
                print(f'{key}: {repr(value)}')
        if not show_notion:
            self.print(f'スペルミスがないか確認して//Check if typos exist.')

    def save_as_json(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory) and directory != '':
            os.makedirs(directory)

        with open(file_path, 'w', encoding='utf-8') as w:
            print(json.dumps(self._args, ensure_ascii=False, indent=4), file=w)

    def raise_uninstalled_module(self, module_name):
        self.print(f'{module_name}がインストールされていません//Uninstalled {module_name}')
        print(f'pip3 install -U {module_name}')
        sys.exit(1)

    def raise_error(self, key, desc):
        if desc:
            raise ValueError(desc)
        raise TypeError(f'{key}の設定を忘れてます')

    def warn_unset_key(self, key, value):
        self.print(f'{key} を忘れずに。とりあえず {key}={value} で続けます //Please set {key}')
        return value

    def raise_unset_key(self, key, desc_ja=None, desc_en=None):
        desc_ja = f' ({desc_ja})' if desc_ja else ''
        desc_en = f' ({desc_en})' if desc_en else ''
        self.print(f'{key}{desc_ja}を設定してください//Please set {key}{desc_en}')
        sys.exit(1)


    def print(self, *args, **kwargs):
        print(self.face, *args, **kwargs)

    def verbose_print(self, *args, **kwargs):
        print(self.face, *args, **kwargs)

    @classmethod
    def to_adhoc(cls, aargs: dict=None, args=None, **kwargs):
        aargs = aargs or args
        if not isinstance(aargs, AdhocArguments):
            if aargs is None:
                aargs = AdhocArguments({})
            elif isinstance(aargs, dict):
                aargs = AdhocArguments(aargs)
        # args = {k:v for k,v in vars(hparams).items() if v is not None}
        aargs.update(dict(kwargs))
        return aargs


def adhoc_parse_arguments(subcommands:Optional[List[str]]=None,
                          requires:Optional[List[str]]=None,
                          use_environ=True, expand_config=None)->AdhocArguments:
    global main_aargs
    if subcommands is not None:
        if isinstance(subcommands,str):
            subcommands=subcommands.split('|')
        if len(sys.argv) == 1 or sys.argv[1] not in subcommands:
            print(f'{sys.argv[0]} requires subcommands: {subcommands}')
            sys.exit(0)
        argv = sys.argv[2:]
        args={'_': '', 'subcommand': sys.argv[1]}
    else:
        argv = sys.argv[1:]
        args={'_': ''}

    for arg, next_value in zip(argv, argv[1:] + ['--']):
        key, value = _parse_key_value(arg, next_value, args)
        if value is not None:
            args[key.replace('-', '_')] = value
    del args['_']

    aargs = AdhocArguments(args, 
                          parent=None,
                          expand_config=expand_config, 
                          use_environ=use_environ)

    if requires:
        if isinstance(requires, str):
            requires = requires.split('|')
        lost_found = False
        for key in requires:
            if key not in aargs:
                aargs.print(f'Option {key} is required.')
                lost_found = True
        if lost_found:
            sys.exit(1)
    main_aargs = aargs
    return aargs

###
###





if __name__ == '__main__':
    aargs = adhoc_parse_arguments()
    with AdhocArguments.from_main(a=False,b=1) as aargs:
        print(aargs)
        print(aargs['a'])