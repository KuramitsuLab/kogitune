from typing import List
from ..adhoc_args import AdhocArguments, parse_path_arguments, get_basename_from_filepath
from kogitune.filters import *

def find_eval_function(name: str, aargs: AdhocArguments):
    ns = globals()
    if name in ns:
        return ns[name](aargs=aargs)

function_map = {
    'zlib': ZLibCompression
}

def score_function(path):
    path, args = parse_path_arguments(path)
    if path in function_map:
        func = function_map[path]
    else:
        ns = globals()
        if path in ns:
            func = ns[path]
    return func(**args)

def maxmin(files: List[str], aargs=None):
    from .documents import JSONTemplateConvertor
    aargs = AdhocArguments.to_adhoc(aargs)
    name = aargs['score|eval']
    eval_func = score_function(name)
    N = aargs['N|n|=1000'] # 大きさの調整
    filter = ComposeFilter(
        JSONTemplateConvertor(aargs['json_template|={text}']),
        MaxMinFilter(eval_func, window_size=N, record_name=name),
    )
    output_path = aargs['output_file|output|!maxmin.jsonl']
    filter.from_jsonl(files, N=N, output_path=output_path, num_workers=1)
