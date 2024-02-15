from typing import List
from ..adhocargs import AdhocArguments, get_basename_from_filepath
from kogitune.filters import *

def find_eval_function(name: str, aargs: AdhocArguments):
    ns = globals()
    if name in ns:
        return ns[name](aargs=aargs)

def maxmin(files: List[str], aargs=None):
    from .documents import JSONTemplateConvertor
    aargs = AdhocArguments.to_adhoc(aargs)
    name = aargs['eval|score']
    eval_func = find_eval_function(name, aargs)
    N = aargs['N|n|=1000'] # 大きさの調整
    filter = ComposeFilter(
        JSONTemplateConvertor(aargs['json_template|={text}']),
        MaxMinFilter(eval_func, window_size=N, record_name=name),
    )
    output_path = aargs['output_path']
    if output_path is None:
        basename = get_basename_from_filepath(files[0])
        output_path = f'{basename}_maxmin.jsonl'
        aargs.print(f'{output_path}に評価値も保存します。')
    filter.from_jsonl(files, N=N, output_path=output_path, num_workers=1)
