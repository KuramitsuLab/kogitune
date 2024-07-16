from typing import List, Union
import os
import json
import getpass
from datetime import datetime

from .commons import *
from ..datasets import load_template
from .models import load_model
from .evaluators import evaluate_metric

def load_testdata(testdata: str, aargs):
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

def load_jsonl(datapath:str, aargs):
    datalist = []
    try:
        with zopen(datapath, 'r') as f:
            datalist = [json.loads(line.strip()) for line in f]
    except FileNotFoundError as e:
        raise e
    return basename(datapath), datalist

def load_hfdataset(datapath:str, aargs):
    import datasets
    dataset_args = aargs['dataset_config|dataset_args']
    if dataset_args is None:
        datapath, dataset_args = adhoc.parse_path_args(datapath)
    dataset = datasets.load_dataset(datapath, **dataset_args)
    if isinstance(dataset, datasets.dataset_dict.DatasetDict):
        if 'test' in dataset:
            adhoc.notice("splitの指定がないから、split='test'で進めるよ。")
            dataset = dataset['test']
        else:
            adhoc.warn(f'splitの指定が必要だよ')
            print(dataset)
    datalist = [{k: v for k, v in item.items()} for item in dataset]
    dataname = basename(datapath)
    if 'name' in dataset_args:
        name = dataset_args['name']
        dataname = f'{dataname}_{name}'
    return dataname, datalist

def filter_datatag(name):
    if name.startswith('openai_'):
        name = name[len('openai_'):]
    return name

def load_testdata_from(aargs):
    testdata = aargs['testdata|testdata|dataset|!!']
    if '.json' in testdata:
        dataname, datalist = load_jsonl(testdata, aargs)
    elif testdata.startswith('dummy:'):
        dataname, datalist = load_testdata(testdata[3:], aargs)
    elif testdata.startswith('hf:'):
        dataname, datalist = load_hfdataset(testdata[3:], aargs)
    else:
        dataname, datalist = load_hfdataset(testdata, aargs)

    datatag = aargs['datatag'] if 'datatag' in aargs else filter_datatag(dataname)

    transform = aargs['transform_keys|transform']
    if transform is not None:
        adhoc.transform_keys(datalist, transform)

    dumpdata = json.dumps(datalist[0], indent=4, ensure_ascii=False)
    adhoc.print(f'テストデータ({testdata})を確認しておいてね\n  features: {list(datalist[0].keys())}\n  num_rows: {len(datalist)}\n{dumpdata}', once=True)
    return datatag, datalist


def load_sample_list(sample_file:str):
    with zopen(sample_file, 'rt') as f:
        sample_list = [json.loads(line) for line in f]
        return sample_list

def save_sample_list(sample_file:str, sample_list :List[dict]):
    directory = os.path.dirname(sample_file)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)

    with open(sample_file, 'w', encoding='utf-8') as w:
        for result in sample_list:
            assert(isinstance(result, dict))
            print(json.dumps(result, ensure_ascii=False), file=w)

def _guess_uniquekey(datalist: List[dict]):
    for key in datalist[0].keys():
        if 'id' in key.lower():
            return key
    return None

def prepare_sample_list(sample_file: str, datalist:List[dict], aargs):
    if os.path.exists(sample_file):
        sample_list = load_sample_list(sample_file)
        if datalist is None or len(sample_list) == len(datalist):
            adhoc.notice(f'既存ファイル {sample_file}に追記するよ')
            return sample_list
    # 新しく sample_list を作る
    unique_key = aargs['unique_key|unique_id']
    if unique_key is None:
        unique_key = _guess_uniquekey(datalist)
    elif unique_key not in datalist[0]:
        adhoc.notice(f'unique_key{unique_key}がデータセットにないよ')
        unique_key = None

    if unique_key:
        sample_list = [{unique_key: data[unique_key]} for data in datalist]
    else:
        sample_list = [{'chain_id': f'index/{n}'} for n in range(len(datalist))]

    chain_key = aargs['chain_key|merge_key|chain|merge']
    if chain_key is not None and chain_key in datalist[0]:
        for source, sample in zip(datalist, sample_list):
            sample[chain_key] = source[chain_key]

    save_sample_list(sample_file, sample_list)
    return sample_list

def needs_generation(sample_list, n):
    for record in sample_list:
        if 'outputs' not in record or n > len(record['outputs']):
            return True
    return False

def modeltag_from(aargs):
    model_path = aargs['model_path|!!']
    if ':' in model_path:
        _, _, model_path = model_path.partition(':')
    model_path = basename(model_path, skip_dot=True)
    if 'checkpoint' in model_path and model_path.endswith('000'):
        model_path = f'{model_path[:-3]}k'
    modeltag = aargs['model_tag|modeltag']
    if modeltag is None:
        if model_path.startswith('checkpoint-'):
            adhoc.notice('modeltagを設定した方がいいよ！')
        return model_path
    else:
        if model_path.startswith('checkpoint-'):
            checkpoint = model_path.replace('checkpoint-', '')
            modeltag = f'{modeltag}_cp{checkpoint}'
    return modeltag

def generate_from(aargs):
    datatag, datalist = load_testdata_from(aargs)
    template = load_template(sample=datalist[0], aargs=aargs)

    modeltag = modeltag_from(aargs)
    sample_file = f'{datatag}__{modeltag}.jsonl'

    sample_list = prepare_sample_list(sample_file, datalist, aargs)
    if 'input' not in sample_list[0]:
        template.load_to_sample_list(datalist, sample_list)

    n = aargs['num_return_sequences|n|N|=1']
    if not needs_generation(sample_list, n):
        save_sample_list(sample_file, sample_list)      
        return sample_file

    def saving():
        save_sample_list(sample_file, sample_list)      

    test_run = aargs[f'test_run|head|={len(sample_list)}']
    if test_run < len(sample_list):
        adhoc.print(f'とりあえず、先頭のhead={test_run}件のみ試してみます')    

    model = load_model(aargs=aargs)
    model.configure(template, datalist)
    adhoc.notice('生成をはじめます', model=model, n=n, generator_args=model.generator_args)
    try:
        with adhoc.start_timer() as timer:
            sample_list = [sample for sample in sample_list[:test_run] if 'outputs' not in sample]
            model.generate_streaming(sample_list, 
                                        n=n, 
                                        saving_func=saving, 
                                        batch_size=aargs['eval_batch_size|batch_size|=2'])
            timer.notice('お疲れ様！！ 生成終わりました', total=len(sample_list))
    finally:
        save_sample_list(sample_file, sample_list)
    return sample_file

def get_metric_list(aargs):
    metric_list = aargs['metric_list|metric']
    return adhoc.list_values(metric_list)

def eval_from(datatag, modeltag, aargs):
    metric_list = get_metric_list(aargs)
    sample_file = f'{datatag}__{modeltag}.jsonl'
    sample_list = load_sample_list(sample_file)
    for metric_path in metric_list:
        result = evaluate_metric(sample_list, metric_path, force_eval=aargs['force_eval|=False'])
        if result:
            save_sample_list(sample_file, sample_list)
            result['model'] = modeltag
            result['data'] = datatag
            print(result)
            result['datetime'] = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
            result['contact'] = getpass.getuser()
            score_file = aargs[f'output_file|={datatag}_score.jsonl']
            save_score(score_file, result)

def save_score(score_file, result:dict):
    directory = os.path.dirname(score_file)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)

    with open(score_file, 'a', encoding='utf-8') as w:
        print(json.dumps(result, ensure_ascii=False), file=w)
