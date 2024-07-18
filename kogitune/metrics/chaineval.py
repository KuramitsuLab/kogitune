from typing import List
# import getpass
# from datetime import datetime
# from zoneinfo import ZoneInfo
# from filelock import FileLock

from .commons import *

from .samples import parse_tags, generate_from, eval_from

def test_list(aargs):
    model_list = adhoc.list_values(aargs['model_list|model_path|model'])
    data_list = aargs['dataset_name_list']
    for model_path in model_list:
        args = dict(model_path=model_path)
        if data_list is not None:
            for name in adhoc.list_values(data_list):
                aargs['dataset_name'] = name
                yield args
        else:
            yield args    

def chain_eval(aargs):
    sample_files = aargs['files'] or []
    for local_args in test_list(aargs):
        local_aargs = adhoc.ChainMap(local_args, aargs)
        sample_file = generate_from(local_aargs)
        if sample_file not in sample_files:
            sample_files.append(sample_file)
    for sample_file in sample_files:
        datatag, modeltag = parse_tags(sample_file)
        eval_from(datatag, modeltag, aargs)


def chain_eval_cli(**kwargs):
    with adhoc.aargs_from(**kwargs) as aargs:
        chain_eval(aargs)

def eval_loss_cli(**kwargs):
    kwargs = kwargs | dict(eval_type='loss', metric='perplexity')
    with adhoc.aargs_from(**kwargs) as aargs:
        chain_eval(aargs)

def eval_choice_cli(**kwargs):
    kwargs = kwargs | dict(eval_type='choice', metric='exact_match')
    with adhoc.aargs_from(**kwargs) as aargs:
        chain_eval(aargs)



