from typing import List, Optional
import os
import sys
import re
import inspect
import importlib

from .dicts import (
    aargs_print,
    parse_key_value, load_config, 
    ChainMap, use_os_environ,
)

# main

class AdhocArguments(ChainMap):
    """
    アドホックな引数パラメータ
    """
    def __init__(self, args:dict, parent=None, caller='main'):
        super().__init__(args, parent) 
        self.caller = caller

    def __enter__(self):
        push_stack_aargs(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.check_unused()
        pop_stack_aargs()

    def check_unused(self):
        unused_keys = self.unused_keys()
        if len(unused_keys) > 0:
            aargs_print(f'未使用の引数があるよ//List of unused arguments')
            for key in unused_keys:
                value = self[key]
                print(f'{key}: {value}')
            print(f'引数にスペルミスがないか確認して//Check if typos exist.')
        # else:
        #     for key, value in self.local_args.items():
        #         if key not in self.used_keys:
        #             print('@used_keys', self.used_keys)
        #             raise TypeError(f'{key} is an unused keyword at {self.caller}')

    def from_kwargs(self, **kwargs):
        return AdhocArguments(kwargs, self)


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

def parse_argv(argv: List[str], expand_config='config'):
    # argv = sys.argv[1:]
    args={'_': ''}
    for arg, next_value in zip(argv, argv[1:] + ['--']):
        key, value = _parse_key_value(arg, next_value, args)
        if value is not None:
            key = key.replace('-', '_')
            if key == expand_config:
                loaded_data = load_config(value)
                args.update(loaded_data)
            else:
                args[key.replace('-', '_')] = value
    del args['_']
    return args

def parse_main_args(use_subcmd=False, use_environ=True, expand_config='config'):
    env = use_os_environ() if use_environ else None    
    if use_subcmd:
        if len(sys.argv) == 1:
            print(f'{sys.argv[0]} requires subcommands')
            sys.exit(0)
        args = parse_argv(sys.argv[2:], expand_config=expand_config)
        args['subcommdn'] = sys.argv[1]
    else:
        args = parse_argv(sys.argv[1:], expand_config=expand_config)
    return AdhocArguments(args, env, caller='main')

import importlib

def load_symbol(module_path, symbol):
    module = importlib.import_module(module_path)
    return getattr(module, symbol)

def load_class(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls

def load_subcommand(subcommand, **kwargs):
    fname = f'{subcommand}_cli'
    if '.' in fname:
        cls = load_class(fname)
    else:
        cls = load_symbol('kogitune.cli', fname)
    cls(**kwargs)


def launch_subcommand(module_path='kogitune.cli'):
    with parse_main_args(use_subcmd=True) as aargs:
        subcmd = aargs['subcommand']
        fname = f'{subcmd}_cli'
        if '.' in fname:
            cls = load_class(fname)
        else:
            cls = load_symbol(module_path, fname)
        cls()

###
AARGS_STACKS = []
AARGS_ENV = None

def get_stack_aargs():
    global AARGS_STACKS, AARGS_ENV
    if len(AARGS_STACKS) == 0:
        if AARGS_ENV is None:
            AARGS_ENV = AdhocArguments({}, use_os_environ())
        return AARGS_ENV
    return AARGS_STACKS[-1]

def push_stack_aargs(aargs):
    global AARGS_STACKS
    AARGS_STACKS.append(aargs)

def pop_stack_aargs():
    global AARGS_STACKS
    AARGS_STACKS.pop()

def from_kwargs(**kwargs) -> AdhocArguments:
    if 'aargs' in kwargs:
        # aargs をパラメータに渡すのは廃止
        raise ValueError('FIXME: aargs is unncessary')
    aargs = get_stack_aargs()
    caller_frame = inspect.stack()[1].function
    return AdhocArguments(kwargs, parent=aargs, caller=caller_frame)

def aargs_from(args: dict = None, **kwargs) -> AdhocArguments:
    if args is None:
        args = kwargs
    else:
        args = args | kwargs
    aargs = get_stack_aargs()
    caller_frame = inspect.stack()[1].function
    return AdhocArguments(args, parent=aargs, caller=caller_frame)

