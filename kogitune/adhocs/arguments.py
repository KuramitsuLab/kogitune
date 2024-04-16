from typing import List, Optional
import os
import sys
import json
import re
import inspect

try:
    from termcolor import colored
except:
    def colored(text, color):
        return text

from .stacks import (
    push_section, pop_section, get_section, 
    get_main_aargs, get_stack_aargs,
)

from .dicts import (
    find_simkey, parse_key_value, 
    pop_from_dict, load_config,
    copy_dict_keys, move_dict_keys,
    check_verbose_in_dict, 
)

from .formats import (
    format_print_args, flush_buf
)

from .logs import log

class Default(object):
    """
    デフォルト値であることを示すラッパークラス
    """
    def __init__(self, value):
        self.value = value
    

_PRINT_ONCE = set()

# main

class AdhocArguments(object):
    """
    アドホックな引数パラメータ
    """
    def __init__(self, 
                 args:dict, parent=None, 
                 caller='main', section=None, 
                 expand_config='config', 
                 use_environ=True, 
                 face='🦊'):
        self.local_args = {}
        self.used_keys = set()
        self.searched_keys = set()
        self.use_environ = use_environ
        self.face = face
        self.parent = parent
        self.caller = caller
        if parent:
            self.use_environ = False
            self.face = parent.face
        for key, value in args.items():
            if isinstance(value, Default):
                continue
            if key == expand_config:
                self.load_config(value)
            else:
                self.local_args[key] = value
        self.stack_backups = push_section(section or caller, self)

    def __repr__(self):
        return repr(self.as_dict())

    def used(self, key):
        self.used_keys.add(key)
        if self.parent:
            self.parent.used(key)

    def get(self, keys, default_value=None):
        if not isinstance(keys, list):
            keys = keys.split('|')
        default_key = keys[0]
        self.searched_keys.add(default_key)
        text = None
        for key in keys:
            if key.startswith('=') or key.startswith('!'):
                text = key
                continue
            if key in self.local_args:
                self.used(key)
                return self.local_args[key]
        if self.parent:
            return self.parent.get(keys, default_value)
        keys = [key for key in keys if not key.startswith('=') and not key.startswith('!')]
        # 小文字の環境変数は常に参照する
        for key in keys:
            if key in os.environ:
                return self.get_found(key, os.environ[key])
        if self.use_environ:
            for key in keys:
                ukey = key.upper()
                if ukey in os.environ:
                    return self.get_found(key, os.environ[ukey])
        for key in keys:
            simkey = find_simkey(self.local_args, key)
            if simkey is not None:
                self.print(f'タイポ？ {simkey}は{key}だよね？違ったらごめんね', color='red', once=True)
                return self.local_args[simkey]
        if text is not None:
            if text.startswith('='):
                return parse_key_value(default_key, text[1:])
            elif text.startswith('!!'):
                return self.raise_key_error(default_key, text[2:])
            elif text.startswith('!'):
                default_value = parse_key_value(default_key, text[1:])
                return self.warn_key_error(default_key, default_value)
        return default_value

    def get_found(self, key: str, string_value: str):
        value = parse_key_value(key, string_value)
        self.local_args[key] = value
        self.used(key)
        return value

    def raise_key_error(self, key, desc):
        if desc:
            raise ValueError(desc)
        raise TypeError(f'{key}の設定を忘れてます')

    def warn_key_error(self, key, value):
        self.print(f'{key} を忘れずに。とりあえず {key}={value} で続けます')
        return value

    def copy_to(self, newargs: dict, *keys_list):
        copy_dict_keys(self, newargs, *keys_list)

    def __getitem__(self, key):
        return self.get(key, None)

    def __setitem__(self, key, value):
        self.local_args[key] = value
        # setattr(self, key, value)
        self.used(key)

    def __contains__(self, key):
        return key in self.local_args or (self.parent and key in self.parent) or (self.use_environ and key.upper() in os.environ)

    def as_dict(self):
        dic = {}
        if self.parent:
            dic.update(self.parent.items())
        else:
            dic.update({k: parse_key_value(k,v) for k,v in os.environ.items() if k.islower()})
        dic.update(self.local_args)
        return dic

    def keys(self):
        return self.as_dict().keys()

    def values(self):
        return self.as_dict().values()

    def items(self):
        return self.as_dict().items()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.check_unused()
        pop_section(self.stack_backups)

    def check_unused(self):
        if self == get_main_aargs():
            show_notion = True
            for key, value in self.local_args.items():
                if key not in self.used_keys:
                    if show_notion:
                        self.print(f'未使用のコマンド引数があるよ//List of unused parameters')
                        show_notion = False
                    print(f'{key}: {repr(value)}')
            if not show_notion:
                self.print(f'スペルミスがないか確認して//Check if typos exist.')
        else:
            for key, value in self.local_args.items():
                if key not in self.used_keys:
                    print('@used_keys', self.used_keys)
                    raise TypeError(f'{key} is an unused keyword at {self.caller}')

    def update(self, otherdict:dict, overwrite=True, used=True):
        for key, value in otherdict.items():
            if overwrite or key not in self.local_args:
                self.local_args[key] = value
                if used:
                    self.used(key)

    def load_config(self, config_file, merge=True, overwrite=True):
        loaded_data = load_config(config_file)
        if merge:
            self.update(loaded_data, overwrite=overwrite)
        return loaded_data

    # def subset(self, keys='', prefix=None):
    #     subargs = {}
    #     keys = set(keys.split('|'))
    #     for key, value in self.local_args.items():
    #         if key in keys:
    #             self.used_keys.add(key)
    #             subargs[key] = value
    #         elif prefix and key.startswith(prefix):
    #             self.used_keys.add(key)
    #             key = key[len(prefix):]
    #             if key.startswith('_'):
    #                 key = key[1:]
    #             subargs[key] = value
    #     return subargs

    # def get_subargs(self, keys, exclude=None):
    #     subargs = {}
    #     for key in keys.split('|'):
    #         if key.endswith('*'):
    #             prefix = key[:-1]
    #             for key, value in self.local_args.items():
    #                 if key.startswith(prefix):
    #                     self.used_keys.add(key)
    #                     key = key[len(prefix):]
    #                     subargs[key] = value
    #         elif key in self:
    #             subargs[key] = self[key]
    #     if exclude is not None:
    #         for key in exclude.split('|'):
    #             if key in subargs:
    #                 del subargs[key]
    #     return subargs

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

    def save_as_json(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory) and directory != '':
            os.makedirs(directory)

        with open(file_path, 'w', encoding='utf-8') as w:
            print(json.dumps(self.as_dict(), ensure_ascii=False, indent=4), file=w)

    # def raise_uninstalled_module(self, module_name):
    #     self.print(f'{module_name}がインストールされていません//Uninstalled {module_name}')
    #     print(f'pip3 install -U {module_name}')
    #     sys.exit(1)

    # def raise_error(self, key, desc):
    #     if desc:
    #         raise ValueError(desc)
    #     raise TypeError(f'{key}の設定を忘れてます')

    # def warn_unset_key(self, key, value):
    #     self.print(f'{key} を忘れずに。とりあえず {key}={value} で続けます //Please set {key}')
    #     return value

    # def raise_unset_key(self, key, desc_ja=None, desc_en=None):
    #     desc_ja = f' ({desc_ja})' if desc_ja else ''
    #     desc_en = f' ({desc_en})' if desc_en else ''
    #     self.print(f'{key}{desc_ja}を設定してください//Please set {key}{desc_en}')
    #     sys.exit(1)

    def print(self, *args, **kwargs):
        if 'verbose' in kwargs:
            target = kwargs.pop('verbose')
            if not check_verbose_in_dict(self, target, default_verbose=True):
                return
        face = kwargs.pop('face', self.face)
        once = kwargs.pop('once', None)
        color = kwargs.pop('color', None)
        flush = kwargs.pop('flush_buf', False)
        if once:
            value = f'{args[0]}' if once == True else once
            if value in _PRINT_ONCE:
                return
            _PRINT_ONCE.add(value)
        text = format_print_args(*args, **kwargs)
        if color:
            text = colored(text, color)
        print(f'{face}{text}', end=kwargs.pop('end', os.linesep))
        if flush:
            print(flush_buf())

# main adhoc arguments

def from_kwargs(open_section=None, **kwargs):
    # aargs = pop_from_dict('aargs', kwargs)
    # if not isinstance(aargs, AdhocArguments):
    aargs = get_stack_aargs()
    caller_frame = inspect.stack()[1].function
    return AdhocArguments(kwargs, parent=aargs, 
                          caller=caller_frame, section=open_section or get_section())

### parse_argument

_key_pattern = re.compile(r'^[A-Za-z0-9\.\-_]+\=')

def _parse_key_value(key, next_value, args):
    if _key_pattern.match(key):
        if key.startswith('--'):
            key = key[2:]
        key, _, value = key.partition('=')
        return key, parse_key_value(key, value)     
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
            return key, parse_key_value(key, next_value)
    else:
        if args.get('_') != key:
            files = args.get('files', [])
            files.append(key)
            args['files'] = files
    return key, None

def parse_main_args(subcommands:Optional[List[str]]=None,
                        use_environ=True, expand_config='config')->AdhocArguments:
    if subcommands is not None:
        assert isinstance(subcommands, list)
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
    log('main', 'argv', argv)
    return aargs

###
###

if __name__ == '__main__':
    aargs = parse_main_args()
    with from_main(a=False,b=1) as aargs:
        print(aargs)
        print(aargs['a'])