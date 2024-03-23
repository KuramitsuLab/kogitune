from typing import List
import time
import getpass
from datetime import datetime
from zoneinfo import ZoneInfo

from .local_utils import *

from .datatum import load_data, prepare_result, needs_model_inference, save_score, save_result
from .templates import load_template

from .models import load_model, get_modeltag
from .evaluators import evaluate_metric

def sec(result_file):
    return result_file.replace('.jsonl', '')

def chain_eval_with_aargs(aargs):
    datalist = load_data(aargs)
    template = load_template(datalist, aargs)

    datatag = aargs['datatag']
    modeltag = get_modeltag(aargs)
    result_file = aargs['result_file|output_file']
    if result_file is None:
        result_file = f'{datatag}_{modeltag}.jsonl'

    result_list = prepare_result(datalist, result_file, aargs)
    n = aargs['num_return_sequences|n|N|=1']
    test_run = aargs['test_run|head']

    if needs_model_inference(result_list, n):
        model = load_model(aargs=aargs)
        adhoc.notice(sec(result_file), '推論をはじめます', model=model, n=n)

        if isinstance(test_run,int):
            adhoc.print(f'とりあえず、先頭の{test_run}件のみ試してみます')
            result_list = result_list[:test_run]
        
        elapsed_time = 0
        save_items = aargs[f'save_items|={len(result_list)//2}']
        for i, record in enumerate(configurable_tqdm(result_list, desc=f'{model}')):
            source = datalist[i]
            if 'input' not in record:
                record['input'] = template.create_prompt(source)
            if 'reference' not in record:
                record['reference'] = template.create_reference(source)
            if 'outputs' not in record:
                start_time = time.time()
                record['outputs'] = model.generate_list(record['input'], n=n)
                record['time'] = (time.time() - start_time) / n
            else:
                remaining_n = n - len(record['outputs'])
                if remaining_n > 0:
                    start_time = time.time()
                    record['outputs'].append(model.generate_list(record['input'], n=remaining_n))
                    record['time'] = (time.time() - start_time) / remaining_n
            if 'output' not in record:
                record['output'] = record['outputs'][0]
            if 'time' in record:
                elapsed_time += record['time']
            if i % save_items == save_items-1:
                save_result(result_file, result_list)
        adhoc.notice(sec(result_file), 'お疲れ様！！ 推論終わりました', 
                     total_time=elapsed_time, throughtput=elapsed_time/(len(datalist)*n))
        save_result(result_file, result_list)
    
    ## モデル評価を行います。
    metric_list = aargs['metric_list|metrics|metric']
    if metric_list is None:
        return
    if isinstance(metric_list, str):
        metric_list = metric_list.split('|')
    score_file = aargs['score_file|score_output_file|=score.jsonl']
    for metric_path in metric_list:
        result = evaluate_metric(result_list, metric_path)
        if result:
            save_result(result_file, result_list)
            print(result)
            result['modeltag'] = modeltag
            result['datatag'] = datatag
            result['datetime'] = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
            result['contact'] = getpass.getuser()
            save_score(score_file, result)

def chain_eval(**kwargs):
    with AdhocArguments.from_main(**kwargs) as aargs:
        models = aargs['files|model_list']
        if models:
            for model_path in models:
                aargs['model_path'] = model_path
                try:
                    chain_eval_with_aargs(aargs)
                except BaseException as e:
                    adhoc.warn(f'{model_path}の評価に失敗したよ: {e}')
        else:
            chain_eval_with_aargs(aargs)