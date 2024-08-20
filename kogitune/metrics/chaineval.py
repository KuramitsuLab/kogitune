from typing import List

from .commons import *

from .samples import parse_tags, generate_from, selfcheck_from, back_translation, eval_from

def test_list(aargs):
    model_list = adhoc.list_values(aargs['model_list|model_path|model'])
    data_list = aargs['dataset_name|dataset_names|dataset_name_list']
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
        eval_from(sample_file, aargs)

def back_eval(aargs):
    sample_files = aargs['files'] or []
    instruction = aargs['instruction|inst|prompt|!!']
    back_instruction = aargs['back|reverse|!!']
    for local_args in test_list(aargs):
        local_aargs = adhoc.ChainMap(local_args, aargs)
        local_aargs['add_dict'] = {'instruction': instruction}
        local_aargs['eval_type'] = 'back'
        sample_file = generate_from(local_aargs)
        back_translation(sample_file, back_instruction)
        sample_file = generate_from(local_aargs)
        if sample_file not in sample_files:
            sample_files.append(sample_file)
    for sample_file in sample_files:
        eval_from(sample_file, aargs)

def chain_eval_cli(**kwargs):
    with adhoc.aargs_from(**kwargs) as aargs:
        chain_eval(aargs)

def selfcheck_cli(**kwargs):
    kwargs = dict(max_new_tokens=128) | kwargs | dict(selfcheck=True)
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


