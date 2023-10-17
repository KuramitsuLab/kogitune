import os
import argparse
#from papertown import DatasetStore, DataComposer, load_tokenizer
#from tqdm import tqdm

from .commons import *
from .splitters import split_to_store

def _tobool(s):
    return s.lower() == 'true' or s == '1'

def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _parse_args(args):
    args_list = args.split(',')
    args_dict = {}
    for arg in args_list:
        key, value = arg.split('=')
        if value.isdigit():
            args_dict[key] = int(value)
        elif _is_float(value):
            args_dict[key] = float(value)
        elif value.lower() == 'true' or value.lower() == 'false':
            args_dict[key] = _tobool(value)
        else:
            args_dict[key] = value
    return args_dict

def setup_store():
    parser = argparse.ArgumentParser(description="papertown_store")
    parser.add_argument("files", type=str, nargs="+", help="files")
    parser.add_argument("--desc", type=str, default=None)
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--store_path", default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--type", type=str, default='')
    parser.add_argument("--format", default="simple")
    parser.add_argument("--split", default="train")
    parser.add_argument("--split_args", type=_parse_args, default=None)
    parser.add_argument("--sep", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--N", "-N", type=int, default=-1)
    parser.add_argument("--verbose", type=_tobool, default=True)
    parser.add_argument("--histogram", type=_tobool, default=False)
    parser.add_argument("--num_works", type=int, default=0)
    hparams = parser.parse_args()  # hparams になる
    return hparams

def main_store():
    hparams = setup_store()
    split_to_store(
        hparams.files[0],
        N=hparams.N,
        desc=hparams.desc,
        tokenizer_path=hparams.tokenizer_path,
        training_type=hparams.type,
        format=hparams.format,
        split=hparams.split,
        split_args=hparams.split_args or {},
        block_size=hparams.block_size, 
        store_path=None, 
        verbose=True, 
        histogram=hparams.histogram
    )


def main_update():
    import os
    os.system('pip3 uninstall -y papertown')
    os.system('pip3 install -U git+https://github.com/kuramitsulab/papertown.git')
