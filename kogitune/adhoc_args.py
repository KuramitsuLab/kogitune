from kogitune.adhocargs import AdhocArguments, adhoc_parse_arguments, parse_path_arguments, verbose_print
from kogitune.configurable_tqdm import configurable_progress_bar, configurable_tqdm
from kogitune.configurable_tokenizer import configurable_tokenizer

import json

MAX_LIMIT = 1000000
MAIN_LOG = {}

def check_logdata(data):
    if isinstance(data, (int, float, str, bool)) or data is None:
        return data
    elif isinstance(data, dict):
        d={}
        for key, value in data.items():
            if isinstance(value, list) and len(value) == 1:
                d[key] = check_logdata(value[0])
            else:
                d[key] = check_logdata(value)
        return d
    elif isinstance(data, (list, tuple)):
        return [check_logdata(x) for x in data]
    else:
        return f'{data}' # stringfy

def adhoc_log(section:str, key:str, data, max_limit=MAX_LIMIT):
    global MAIN_LOG
    if section not in MAIN_LOG:
        MAIN_LOG[section] = {}
    log = MAIN_LOG[section]
    if key in log:
        values = log[key]
        if len(values) < max_limit:
            values.append(data)
    else:
        log[key] = [data]

import os

def save_adhoc_log(output_path):
    global MAIN_LOG
    log = check_logdata(MAIN_LOG)
    if '.json' in output_path:
        output_filename = output_path.replace('.json', '_log.json')
    else:
        output_filename = os.path.join(output_path, '_log.json')
    with open(output_filename, 'w') as w:
        json.dump(log, fp=w, ensure_ascii=False, indent=2)
    MAIN_LOG = {}