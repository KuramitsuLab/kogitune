import os
import sys
import json
import re

def parse_argument_value(value):
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
    return value

_key_pattern = re.compile(r'^[A-Za-z0-9\.\-_]+\=')

def _parse_key_value(key, next_value, args):
    if _key_pattern.match(key):
        if key.startswith('--'):
            key = key[2:]
        key, _, value = key.partition('=')
        return key, parse_argument_value(value)     
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
            return key, parse_argument_value(next_value)
    else:
        if args.get('_') != key:
            files = args.get('files', [])
            files.append(key)
            args['files'] = files
    return key, None

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

class AdhocArguments(object):
    """
    アドホックな引数パラメータ
    """
    def __init__(self, args:dict, expand_config=None, default_args:dict=None, use_environ=True):
        self._args = {}
        self._used_keys = set()
        self._use_environ = use_environ
        for key, value in args.items():
            if key == expand_config:
                self.load_config(value)
            else:
                self._args[key] = value
        if default_args:
            lost_found = False
            for key, value in default_args.items():
                if key in args:
                    continue
                if use_environ:
                    environ_key = key.upper()
                    if environ_key in os.environ:
                        value = parse_argument_value(os.environ[environ_key])
                if isinstance(value, tuple) and len(value)==1:
                    print(f'Option {key} is required. {value[0]}')
                    lost_found = True
                else:
                    self._used_keys.add(key)
                    args[key] = value
            if lost_found:
                sys.exit(1)

    def __repr__(self):
        return repr(self._args)

    def __getitem__(self, key):
        keys = key.split('|')
        for key in keys:
            if key in self._args:
                self._used_keys.add(key)
                return self._args[key]
            if key.startswith('='):
                return parse_argument_value(key[1:])
            if key.startswith('!'):
                raise ValueError(f'{key[0]} is unset.')
            if self._use_environ:
                environ_key = key.upper()
                if environ_key in os.environ:
                    value = parse_argument_value(os.environ[environ_key])
                    self._used_keys.add(key)
                    self._args[key] = value
                    return value
        return None

    def __setitem__(self, key, value):
        self._args[key] = value
        setattr(self, key, value)
        self._used_keys.add(key)

    def __contains__(self, key):
        return key in self._args or key.upper() in os.environ

    def update(self, otherdict:dict, overwrite=True):
        for key, value in otherdict.items():
            if overwrite or key not in self._args:
                self._args[key] = value

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

    def utils_check(self):
        show_notion = True
        for key, value in self._args.items():
            if key not in self._used_keys:
                if show_notion:
                    self.utils_print(f'未使用のパラメータ一覧//List of unused parameters')
                    show_notion = False
                print(f'{key}: {repr(value)}')
        if not show_notion:
            self.utils_print(f'スペルミスがないか確認してください//Check if typos exist.')

    def save_as_json(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory) and directory != '':
            os.makedirs(directory)

        with open(file_path, 'w', encoding='utf-8') as w:
            print(json.dumps(self._args, ensure_ascii=False, indent=4), file=w)

    def raise_uninstalled_module(self, module_name):
        self.utils_print(f'{module_name}がインストールされていません//Uninstalled {module_name}')
        print(f'pip3 install -U {module_name}')
        sys.exit(1)

    def raise_unset_key(self, key, desc_ja=None, desc_en=None):
        desc_ja = f' ({desc_ja})' if desc_ja else ''
        desc_en = f' ({desc_en})' if desc_en else ''
        self.utils_print(f'{key}{desc_ja}を設定してください//Please set {key}{desc_en}')
        sys.exit(1)

    def utils_print(self, *args, **kwargs):
        print("🐥", *args, **kwargs)

    def verbose_print(self, *args, **kwargs):
        print("🐓", *args, **kwargs)

def adhoc_argument_parser(subcommands=None,
                          default_args:dict=None, 
                          use_environ=True, expand_config=None):
    if subcommands is not None:
        if isinstance(subcommands,str):
            subcommands=subcommands.split('|')
        if len(sys.argv) == 1:
            print(f'{sys.argv[0]} requires subcommands: {subcommands}')
            sys.exit(0)
        if sys.argv[1] not in subcommands:
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
    return AdhocArguments(args, 
                          expand_config=expand_config, 
                          default_args=default_args, 
                          use_environ=use_environ)

if __name__ == '__main__':
    args = adhoc_argument_parser()
    print(args)