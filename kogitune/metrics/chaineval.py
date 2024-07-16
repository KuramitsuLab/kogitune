from typing import List
# import getpass
# from datetime import datetime
# from zoneinfo import ZoneInfo
# from filelock import FileLock

from .commons import *

from .samples import generate_from, eval_from

def chain_eval_cli(**kwargs):
    import traceback
    with adhoc.aargs_from(**kwargs) as aargs:
        sample_files = aargs['files'] or []
        model_list = adhoc.list_values(aargs['model_list|model_path|model'])
        for model_path in model_list:
            local_aargs = adhoc.ChainMap({
                'model_path': model_path,
                'tokenizer_path': model_path,
            }, aargs)
            sample_file = generate_from(local_aargs)
            if sample_file not in sample_files:
                sample_files.append(sample_file)
        for sample_file in sample_files:
            if '__' not in sample_file:
                continue
            datatag, _, modeltag = sample_file.partition('__')
            eval_from(datatag, modeltag, aargs)
