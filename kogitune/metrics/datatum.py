from typing import List
import os
import json

from .commons import *

def load_testdata(dataset_path: str, aargs):
    datalist = []
    n = 10
    for i in range(1, n + 1):
        datalist.append({
            "task_id": f"test_{i}",
            "prompt": f"test_prompt_{i}",
            "canonical_solution": f"test_solution_{i}",
            "test": f"test_test_{i}",
            "entry_point": f"test_entry_{i}"
        })
    return 'humaneval-dummy', datalist

def load_jsonl(dataset_path:str, aargs):
    datalist = []
    try:
        with zopen(dataset_path, 'r') as f:
            datalist = [json.loads(line.strip()) for line in f]
    except FileNotFoundError as e:
        raise e
    return basename(dataset_path), datalist

def load_hfdataset(dataset_path:str, aargs):
    import datasets
    dataset_args = aargs['dataset_args|dataset_config']
    if dataset_args is None:
        dataset_path, dataset_args = parse_path_arguments(dataset_path)
    dataset = datasets.load_dataset(dataset_path, **dataset_args)
    if isinstance(dataset, datasets.dataset_dict.DatasetDict):
        if 'test' in dataset:
            adhoc.warn(f'splitの指定がないから、split=testで進めるよ。')
            dataset = dataset['test']
        else:
            adhoc.warn(f'splitの指定が必要だよ')
            print(dataset)
    datalist = [{k: v for k, v in item.items()} for item in dataset]
    dataset_name=basename(dataset_path)
    if 'name' in dataset_args:
        name = dataset_args['name']
        dataset_name = f'{dataset_name}_{name}'
    return dataset_name, datalist

def transform_data(datalist: List[dict], aargs = None, transform: str=None):
    """
    transform='key1=key2|key2'
    """
    if transform is not None:
        keys = transform
    elif aargs is not None:
        keys = aargs['transform_keys|remove_keys|transform']
        if keys is None:
            return
    keys = keys.replace(r'\n', '\n').split('|')
    transforms=[]
    for key in keys:
        key, _, format = key.partition('=')
        transforms.append((key, format))
    if len(transforms) > 0:
        for data in datalist:
            for key, format in transforms:
                if format == '':
                    del data[key]
                elif '{' in format:
                    data[key] = format.format(**data)
                else:
                    data[key] = data[format]

def filter_datatag(name):
    if name.startswith('openai_'):
        name = name[len('openai_'):]
    return name

def load_data(aargs: adhoc.Arguments):
    dataset_path = aargs['dataset_path|dataset|!!datasetの設定がないよ']
    if '.json' in dataset_path:
        dataset_name, datalist = load_jsonl(dataset_path, aargs)
    elif dataset_path.startswith('dummy:'):
        dataset_name, datalist = load_testdata(dataset_path[3:], aargs)
    elif dataset_path.startswith('hf:'):
        dataset_name, datalist = load_hfdataset(dataset_path[3:], aargs)
    else:
        dataset_name, datalist = load_hfdataset(dataset_path, aargs)

    transform_data(datalist, aargs)

    if 'datatag' not in aargs:
        aargs['datatag'] = filter_datatag(dataset_name)

    dumpdata = json.dumps(datalist[0], indent=4, ensure_ascii=False)
    adhoc.print(f'データセット({dataset_path})を確認しておいてね\n  features: {list(datalist[0].keys())}\n  num_rows: {len(datalist)}\n{dumpdata}', once=True)
    return datalist

def _guess_uniquekey(datalist: List[dict]):
    for key in datalist[0].keys():
        if 'id' in key.lower():
            return key
    return None

def prepare_result(result_file:str, datalist:List[dict], aargs):
    if result_file:
        if os.path.exists(result_file):
            with zopen(result_file, 'rt') as f:
                result_list = [json.loads(line) for line in f]
            if datalist is None or len(result_list) == len(datalist):
                adhoc.warn(f'ファイル {result_file}に追記するよ')
                return result_list
            adhoc.warn(f'ファイル {result_file}の一致しないから上書きするよ')
        else:
            if datalist is None:
                raise FileNotFoundError(result_file)
            adhoc.print(f'新しく保存するよ.. {result_file}')

    unique_key = aargs['unique_key|unique_id']
    if unique_key is None:
        unique_key = _guess_uniquekey(datalist)
    elif unique_key not in datalist[0]:
        adhoc.warn(f'unique_key{unique_key}がデータセットにないよ')
        unique_key = None
    if unique_key:
        return [{unique_key: data[unique_key]} for data in datalist]
    else:
        return [{'chain_id': f'index/{n}'} for n in range(len(datalist))]

def needs_model_inference(result_list, n):
    for record in result_list:
        if 'outputs' not in record or n > len(record['outputs']):
            return True
    return False

def save_result(result_file:str, result_list:List[dict]):
    directory = os.path.dirname(result_file)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)

    with open(result_file, 'w', encoding='utf-8') as w:
        for result in result_list:
            assert(isinstance(result, dict))
            print(json.dumps(result, ensure_ascii=False), file=w)

def save_score(score_file, result:dict):
    directory = os.path.dirname(score_file)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)

    with open(score_file, 'a', encoding='utf-8') as w:
        print(json.dumps(result, ensure_ascii=False), file=w)
