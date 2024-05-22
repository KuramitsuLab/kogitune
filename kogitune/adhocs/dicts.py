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

def check_verbose_in_dict(dic: dict, verbose_target:str, default_verbose=True):
    verbose = dic.get('verbose', default_verbose)
    if isinstance(verbose, bool):
        return verbose
    verbose = str(verbose)
    return verbose_target in verbose


def find_simkey(dic, given_key, max_distance=1):
    key_map = {}
    for key in dic.keys():
        if key not in key_map:
            key_map[key] = edit_distance(key, given_key)
    keys = sorted([(dis, k) for k, dis in key_map.items() if dis <= max_distance])
    if len(keys) > 0:
        #print(keys)
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

def load_text_list(list_file):
    with open(list_file) as f:
        return [line.strip() for line in f.readlines() if line.strip() != '' and not line.startswith('#')]

def parse_key_value(key:str, value:str):
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
        return load_text_list(value)
    if key.startswith('_config') and value.endsith('_args'):
        return load_config(value, default_value=value)
    return value

def list_keys(keys: Union[List[str],str]):
    if isinstance(keys, list):
        return keys
    return keys.split('|')

def pop_from_dict(keys: Union[List[str],str], 
        dic:dict, default_value=None,
        use_environ=True, use_simkey=0):
    keys = list_keys(keys)
    default_key = keys[0]
    for key in keys:
        if key in dic:
            return dic.pop(key)
        if key in os.environ:
            # 環境変数は見るが取り除かない
            return parse_key_value(key, os.environ[key])
        ukey = key.upper()
        if use_environ and ukey in os.environ:
            # 環境変数は見るが取り除かない
            return parse_key_value(ukey, os.environ[ukey])
    if use_simkey > 0:
        simkey = find_simkey(default_key, dic, max_distance=use_simkey)
        return dic.pop(simkey)
    return default_value


def get_from_dict(keys: Union[List[str],str], 
                  dic:dict, 
                  default_value=None,
                  use_environ=True, use_simkey=1):
    keys = list_keys(keys)
    default_key = keys[0]
    for key in keys:
        if key in dic:
            return dic[key]
        if key.startswith('='):
            return parse_key_value(default_key, key[1:])
        if key.startswith('!!'):
            raise KeyError(f"'{default_key}' is not found in {list(dic.keys())}.")
        if key.startswith('!'):
            value = parse_key_value(default_key, key[1:])
            print(f"FIXME: `{default_key}` is missing. Confirm {default_key}={value}.")
            return value
        if key in os.environ:
            return parse_key_value(key, os.environ[key])
        ukey = key.upper()
        if use_environ and ukey in os.environ:
            # 環境変数は見るが取り除かない
            return parse_key_value(key, os.environ[ukey])
    if use_simkey > 0:
        simkey = find_simkey(default_key, dic, max_distance=use_simkey)
        return dic[simkey]
    return default_value

def copy_dict_keys(src_args:dict, dist_args: dict, *keys_list):
    for keys in keys_list:
        keys = list_keys(keys)
        default_key = keys[0]
        for key in keys:
            if key in src_args:
                dist_args[default_key] = src_args[key]
                break

def move_dict_keys(src_args:dict, dist_args: dict, *keys_list):
    for keys in keys_list:
        keys = list_keys(keys)
        default_key = keys[0]
        for key in keys:
            if key in src_args:
                if default_key in dist_args:
                    del src_args[key]
                else:
                    dist_args[default_key] = src_args.pop(key)

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




def parse_path_args(url_or_filepath: str, include_urlinfo=False, global_args={}):
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
    return url, global_args | args
