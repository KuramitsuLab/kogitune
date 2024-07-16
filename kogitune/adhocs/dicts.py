from typing import List, Union, Any
import os
import json
from urllib.parse import urlparse, parse_qs

try:
    from Levenshtein import distance as edit_distance
except:
    ## 編集距離が使えない時の簡便な距離
    def edit_distance(text, text2):
        a = set(list(text))
        b = set(list(text2))
        return len(a.difference(b))+len(b.difference(a))

from .prints import aargs_print, use_ja

def find_simkey(dic, given_key, max_distance=1):
    key_map = {}
    for key in dic.keys():
        if key not in key_map:
            key_map[key] = edit_distance(key, given_key)
    keys = sorted([(dis, k) for k, dis in key_map.items() if dis <= max_distance])
    if len(keys) > 0:
        aargs_print(keys, verbose='simkey')
        return keys[0][1]
    return None

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

def load_config(config_file, default_value={}):
    if config_file.endswith('.json'):
        return load_json(config_file)
    if config_file.endswith('.yaml'):
        return load_yaml(config_file)
    return default_value

def load_list(list_file, convert_fn=str):
    with open(list_file) as f:
        return [convert_fn(line.strip()) for line in f.readlines() if line.strip() != '' and not line.startswith('#')]

def parse_key_value(key:str, value:Union[str,int]):
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    lvalue = value.lower()
    if lvalue == 'true':
        return True
    if lvalue == 'false':
        return False
    if key.startswith('_list') and value.endsith('.txt'):
        return load_list(value)
    if key.startswith('_config') and value.endsith('_args'):
        return load_config(value, default_value=value)
    if key.startswith('_comma') and value.endsith('_camma'):
        return value.split(',')
    return value

def list_keys(keys: Union[List[str],str]):
    if isinstance(keys, list):
        return keys
    return keys.split('|')

def normal_key(key):
    if key.endswith('_comma') or key.endswith('_camma'):
        return key[:-6]
    return key

def get_key_value(dic: dict, keys:str, default_value=None, use_simkey = 1):
    keys = list_keys(keys)
    default_key = normal_key(keys[0])
    for key in keys:
        if key.startswith('='):
            return default_key, parse_key_value(default_key, key[1:])
        if key.startswith('!!'):
            raise KeyError(f"'{default_key}' is not found in {list(dic.keys())}.")
        if key.startswith('!'):
            value = parse_key_value(key[0], key[1:])
            if use_ja():
                aargs_print(f"`{default_key}`が設定されてないよ. デフォルト確認してね！ {default_key}={repr(value)}.")
            else:
                aargs_print(f"FIXME: `{default_key}` is missing. Confirm default value{default_key}={repr(value)}.")
            return default_key, value
        if key in dic:
            return default_key, dic.get(key)
    if use_simkey > 0:
        for key in keys:
            simkey = find_simkey(dic, key, max_distance=use_simkey)
            if simkey in dic:
                return default_key, dic.get(simkey)
    return default_key, default_value


def copy_dict_keys(src_args:dict, dist_args: dict, *keys_list):
    for keys in keys_list:
        keys = list_keys(keys)
        default_key = keys[0]
        for key in keys:
            if key in src_args:
                dist_args[default_key] = src_args[key]
                break

def filter_as_json(data: dict):
    if isinstance(data, (int, float, str, bool)) or data is None:
        return data
    elif isinstance(data, dict):
        d={}
        for key, value in data.items():
            try:
                d[key] = filter_as_json(value)
            except ValueError as e:
                pass
        return d
    elif isinstance(data, (list, tuple)):
        return [filter_as_json(x) for x in data]
    else:
        raise ValueError()

def dump_as_json(data: dict):
    return json.dumps(filter_as_json(data), indent=2)


def parse_path_args(path: str, parent_args=None, include_urlinfo=False):
    """
    pathから引数を読み込む
    """
    if path.startswith('{') and path.startswith('}'):
        ## JSON形式であれば、最初のパラメータはパス名、残りは引数として解釈する。
        args = json.loads(path)
        first_key = list(args.keys())[0]
        if parent_args is None:
            return args.pop(first_key), args
        return args.pop(first_key), ChainMap(args, parent_args)

    parsed_url = urlparse(path)
    options = parse_qs(parsed_url.query)
    args = {k: parse_key_value(k, v[0]) for k, v in options.items()}
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
    if parent_args is None:
        return url, args
    return url, ChainMap(args, parent_args)

def use_os_environ():
    env = {}
    for key in os.environ.keys():
        if key.islower():
            env[key] = parse_key_value(key, os.environ[key])
    return env

class ChainMap(object):
    def __init__(self, dic: dict, parent:dict = None):
        self.parent = parent
        self.local_dic = {} if dic is None else dic
        self.used_keys = []

    def __repr__(self):
        if self.parent is None:
            return repr(self.local_dic)
        return f'{self.local_dic} {self.parent}'
    
    def __contains__(self, key):
        if key in self.local_dic:
            return True
        if self.parent is None:
            return False
        return key in self.parent

    def get(self, key, default_value=None):
        if key in self.local_dic:
            self.use_key(key)
            return self.local_dic[key]
        if self.parent is None:
            return default_value
        return self.parent.get(key, default_value)

    def pop(self, key, default_value=None):
        """
        popはしない
        """
        return self.get(key, default_value)

    def keys(self):
        keys = list(self.local_dic.keys())
        if self.parent is not None:
            for key in self.parent.keys():
                if key not in keys:
                    keys.append(key)
        return keys

    def use_key(self, key):
        self.used_keys.append(key)
        if hasattr(self.parent, 'use_key'):
            self.parent.use_key(key)

    def unused_keys(self):
        unused_keys = []
        for key in self.local_dic.keys():
            if key not in self.used_keys:
                unused_keys.append(key)
        return unused_keys

    def __getitem__(self, key):
        if '|' in key:
            _, value = get_key_value(self, key)
            return value
        if key in self.local_dic:
            self.use_key(key)
            return self.local_dic[key]
        if self.parent is None:
            return None
        return self.parent.get(key, None)

    def __setitem__(self, key, value):
        self.local_dic[key] = value
        self.use_key(key)

    def record(self, *keys, field=None, dic=None):
        """
        """
        for key in keys:
            key, value = get_key_value(self, key)
            if isinstance(dic, dict):
                if not key.startswith("_"):
                    dic[key] = value
            if field is not None:
                if key.startswith("_"):
                    setattr(field, key[1:], value)
                else:
                    setattr(field, key, value)
