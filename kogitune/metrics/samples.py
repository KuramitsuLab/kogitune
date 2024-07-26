from typing import List, Union
import os
import numpy as np
import json
import getpass
from datetime import datetime

from .commons import *
from ..datasets import load_template
from .models import load_model_from
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
    if 'dataset_name' in aargs:
        # dataset_name test_list で設定されるの優先する 
        dataset_args['name'] = aargs['dataset_name']
    if 'trust_remote_code' in dataset_args:
        dataset_args['trust_remote_code'] = True
    try:
        dataset = datasets.load_dataset(datapath, **dataset_args)
    except ValueError as e:
        adhoc.print('データセットのパラメータが変だよ', e)
        raise e
    if isinstance(dataset, datasets.dataset_dict.DatasetDict):
        if 'test' in dataset:
            adhoc.notice("splitの指定がないから、split='test'で進めるよ。")
            dataset = dataset['test']
        else:
            print(dataset)
            raise ValueError(f'splitの指定が必要だよ')
        
    datalist = [{k: v for k, v in item.items()} for item in dataset]
    dataname = basename(datapath)
    if 'name' in dataset_args:
        name = dataset_args['name']
        dataname = f'{dataname}_{name}'
    split = dataset_args.get('split', 'test')
    if split != 'test': ## split も追加する
        dataname = f'{dataname}_{split}'
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
    adhoc.print(f'テストデータ({testdata})を確認しておいてね\n  features: {list(datalist[0].keys())}\n  num_rows: {len(datalist)}\n{dumpdata}', once=testdata)
    return datatag, datalist

## sample_file

def sample_file_name(datatag, modeltag):
    return f'{modeltag}_x_{datatag}.jsonl'

def parse_tags(sample_file:str):
    base_name = basename(sample_file, skip_dot=True).replace('.jsonl', '')
    if '_x_' not in base_name:
        return base_name, ''
    modeltag, _, datatag = base_name.partition('_x_')
    return datatag, modeltag

## 

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
        if len(sample_list) == len(datalist) and aargs['overwrite|=False'] != True:
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

    chain_keys = adhoc.list_values(aargs['chain_keys|chain_key|chain'])
    for chain_key in chain_keys:
        if chain_key not in datalist[0]:
            adhoc.notice(f'FIXME: No such chain_key: {chain_key}')
            continue
        for source, sample in zip(datalist, sample_list):
            sample[chain_key] = source[chain_key]

    add_dict = aargs['additional_dataset|add_dict']
    if add_dict is not None:
        sample_list = [sample | add_dict for sample in sample_list]

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
    eval_type = aargs['eval_type|=generation']
    datatag, datalist = load_testdata_from(aargs)
    template = load_template(sample=datalist[0], aargs=aargs)
    modeltag = modeltag_from(aargs)
    sample_file = sample_file_name(datatag, modeltag)
    sample_list = prepare_sample_list(sample_file, datalist, aargs)
    result_key = template.load_sample(eval_type, datalist, sample_list)

    n = aargs['num_return_sequences|n|N|=1']

    test_run = aargs[f'test_run|head|={len(sample_list)}']

    model = load_model_from(aargs)
    test_list = [sample for sample in sample_list if result_key not in sample]
    if len(test_list) == 0:
        return sample_file
    if eval_type != 'loss' and eval_type != 'choice':
        model.configure(template, datalist, aargs)
    adhoc.notice('生成をはじめます', model=model, eval_type=eval_type, n=n, generator_args=model.generator_args)
    if test_run < len(test_list):
        adhoc.print(f'とりあえず、先頭のhead={test_run}件のみ、試してみます')
        test_list = test_list[:test_run]
    try:
        with adhoc.start_timer() as timer:
            model.predict_sample(test_list, 
                                 eval_type=eval_type, 
                                 n=n, 
                                 batch_size=aargs['eval_batch_size|batch_size|=2'])
            timer.notice('お疲れ様！！ 生成終わりました', total=len(test_list))
    finally:
        save_sample_list(sample_file, sample_list)
    return sample_file

def backup_sample(sample, key):
    if key in sample:
        for i in range(10):
            newkey = f'{key}.{i}'
            if newkey not in sample:
                break
        sample[newkey] = sample[key]

def selfcheck_from(aargs):
    check_aargs = adhoc.ChainMap(
        {'do_sample': False, 'num_return_sequences': 1, 'n': 1},
        parent = aargs,
    )
    adhoc.notice('SelfCheck用の参照を作成します', check_aargs=check_aargs)
    sample_file = generate_from(check_aargs)
    sample_list = load_sample_list(sample_file)
    for sample in sample_list:
        if 'output' in sample:  # head を指定した場合、output が足りなくなる
            backup_sample(sample, 'reference')
            sample['reference'] = sample['output']
            del sample['output']
    save_sample_list(sample_file, sample_list)
    n = aargs['num_return_sequences|n|N|=1']
    if n == 1:
        aargs['num_return_sequences'] = 8
    if 'temperature' not in aargs:
        aargs['temerature'] = 0.8
    if 'do_sample' not in aargs:
        aargs['do_sample'] = True

def back_translation(sample_file, instruction):
    sample_list = load_sample_list(sample_file)
    for sample in sample_list:
        if 'output' in sample:  # head を指定した場合、output が足りなくなる
            backup_sample(sample, 'input')
            sample['input'] = instruction + '\n\n' + sample['output']
            del sample['output']
    save_sample_list(sample_file, sample_list)


def get_metric_list(aargs):
    metric_list = aargs['metric_list|metric']
    return adhoc.list_values(metric_list)

def eval_from(datatag, modeltag, aargs):
    metric_list = get_metric_list(aargs)
    sample_file = sample_file_name(datatag, modeltag)
    sample_list = load_sample_list(sample_file)
    adhoc.notice('評価を始めます', metric_list, sample_file)
    for metric_path in metric_list:
        result = evaluate_metric(sample_list, metric_path, force_eval=aargs['force_eval|=False'])
        if result:
            save_sample_list(sample_file, sample_list)
            result['model'] = modeltag
            result['data'] = datatag
            adhoc.notice(result)
            result['datetime'] = datetime.now().isoformat()
            result['contact'] = getpass.getuser()
            score_file = aargs[f'score_file|output_file|={datatag}_score.jsonl']
            save_score(score_file, result)
            adhoc.saved(score_file, "スコアの記録")
            metric_name = result['metric']
            testname = datatag if metric_name == 'exact_match' else f'{datatag}/{metric_name}'
            filepath = aargs['leadersboard|=leadersboard.csv']
            update_leadersboard(modeltag, testname, result['mean'], filepath=filepath)
            adhoc.saved(filepath, "リーダーズボード//Leaders Board")

def save_score(score_file, result:dict):
    directory = os.path.dirname(score_file)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)

    with open(score_file, 'a', encoding='utf-8') as w:
        print(json.dumps(result, ensure_ascii=False), file=w)

def update_leadersboard(model, key, score, filepath='leadersboard.csv'):
    import pandas as pd
    found = False
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        json_list = df.to_dict(orient='records')
        for d in json_list:
            if d['model'] == model:
                d[key] = score
                scores = [v for k, v in d.items()]
                d['score'] = np.array(scores[2:]).mean()
                found = True
    else:
        json_list = []
    if found == False:
        json_list.append({
            'model': model, 
            'score': score,
            key: score,
        })
    df = pd.DataFrame(json_list).sort_values(by='score', ascending=False)
    df.to_csv(filepath, index=False)
    adhoc.print(df.transpose(), face='')

def import_output_cli(**kwargs):
    with adhoc.aargs_from(**kwargs) as aargs:
        from_file = aargs['from_file|!!']
        output_key = aargs['output_key|!!']
        result_list = load_sample_list(from_file)
        for file in adhoc.list_values(aargs['files']):
            source_list = load_sample_list(file)
            imported_list = []
            for sample, source in zip(result_list, source_list):
                sample = sample.copy()
                sample['output'] = source[output_key]
                imported_list.append(sample)
            file = file.replace('.jsonl', '') + '_imported.jsonl'
            save_sample_list(file, imported_list)