from typing import List
import time
import getpass
from datetime import datetime
from zoneinfo import ZoneInfo
from filelock import FileLock

from .local_utils import *

from .datatum import load_data, prepare_result, needs_model_inference, save_score, save_result
from .templates import load_template

from .models import load_model, get_modeltag
from .evaluators import evaluate_metric

def sec(result_file):
    return result_file.replace('.jsonl', '')

def generate_with_args(aargs):
    datalist = load_data(aargs)
    template = load_template(datalist, aargs)

    datatag = aargs['datatag']
    modeltag = get_modeltag(aargs)
    result_file = aargs['result_file|output_file']
    if result_file is None:
        result_file = f'{datatag}__{modeltag}.jsonl'

    result_list = prepare_result(result_file, datalist, aargs)
    n = aargs['num_return_sequences|n|N|=1']
    test_run = aargs[f'test_run|head|={len(result_list)}']

    if needs_model_inference(result_list, n):
        adhoc.open_section('generation')
        model = load_model(aargs=aargs)
        model.configure(template, datalist)

        adhoc.notice('生成をはじめます', model=model, n=n, generator_args=model.generator_args)
        if test_run < len(result_list):
            adhoc.print(f'とりあえず、先頭の{test_run}件のみ試してみます')
        
        elapsed_time = 0
        save_items = aargs[f'save_items|={len(result_list)//4}']
        for i, record in enumerate(configurable_tqdm(result_list[:test_run], total=test_run, desc=f'{model}')):
            if i >= test_run:
                break
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
        adhoc.notice('お疲れ様！！ 生成終わりました', 
                     total_time=round(elapsed_time,3),
                     throughtput=round(elapsed_time/(test_run*n),3))
        save_result(result_file, result_list)
        adhoc.close_section()
    return result_file, result_list

def get_metric_list(aargs):
    metric_list = aargs['metric_list|metrics|metric']
    if metric_list is None:
        return None
    if isinstance(metric_list, str):
        metric_list = metric_list.split('|')
    return metric_list

def eval_with_args(result_file, result_list, metric_list, aargs):
    adhoc.open_section('eval')
    for metric_path in metric_list:
        result = evaluate_metric(result_list, metric_path, force_eval=aargs['force_eval|=False'])
        if result:
            save_result(result_file, result_list)
            print(result)
            if '__' in result_file:
                datatag, _, modeltag = result_file.replace('.jsonl', '').partition('__')
                result['modeltag'] = modeltag
                result['datatag'] = datatag
                result['datetime'] = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
                result['contact'] = getpass.getuser()
                score_file = aargs['score_file|score_output_file|=score.jsonl']
                save_score(score_file, result)
    adhoc.close_section()

def check_eval_only(aargs):
    files = aargs['files']
    metric_list = get_metric_list(aargs)
    if files is None or metric_list is None:
        return None, metric_list 
    for file in files:
        if not file.endswith('.jsonl') or '__' not in file:
            return None, metric_list
    return files, metric_list

def chain_eval(**kwargs):
    import traceback
    with AdhocArguments.from_main(import_to_main=True, **kwargs) as aargs:
        result_files, metric_list = check_eval_only(aargs)
        if result_files:
            metric_list = get_metric_list(aargs)
            for i, result_file in enumerate(result_files):
                try:
                    result_list = prepare_result(result_file, None, aargs)
                    eval_with_args(result_file, result_list, metric_list, aargs)
                except BaseException as e:
                    adhoc.warn(f'{result_file}の評価に失敗したよ: {e}')
                    traceback.print_exception(e)
            return
        models = aargs['files|model_list']
        if models:
            for i, model_path in enumerate(models, start=1):
                try:
                    aargs['model_path'] = model_path
                    result_file, result_list = generate_with_args(aargs)
                    if metric_list:
                        eval_with_args(result_file, result_list, metric_list, aargs)
                except BaseException as e:
                    adhoc.warn(f'{model_path}の生成・評価に失敗したよ: {e}')
                    traceback.print_exception(e)
        else:
            result_file, result_list = generate_with_args(aargs)
            if metric_list:
                eval_with_args(result_file, result_list, metric_list, aargs)
