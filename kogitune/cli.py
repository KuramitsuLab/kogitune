import os
import argparse

from kogitune.commons import *
from kogitune.splitters import split_to_store
from kogitune.composers import DataComposer

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
    parser.add_argument("--shuffle", type=_tobool, default=True)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--verbose", type=_tobool, default=True)
    parser.add_argument("--histogram", type=_tobool, default=False)
    parser.add_argument("--num_works", type=int, default=0)
    
    hparams = parser.parse_args()  # hparams になる
    return hparams

def main_store(hparams=None):
    if hparams is None:
        hparams = setup_store()
    split_to_store(
        hparams.files,
        N=hparams.N,
        desc=hparams.desc,
        tokenizer_path=hparams.tokenizer_path,
        training_type=hparams.type,
        format=hparams.format,
        split=hparams.split,
        split_args=hparams.split_args or {},
        block_size=hparams.block_size, 
        shuffle=hparams.shuffle,
        random_seed=hparams.random_seed,
        store_path=hparams.store_path,
        validation=True,
        verbose=hparams.verbose, 
        histogram=hparams.histogram
    )

def main_dump(hparams):
    with DataComposer(hparams.urls, 
                      training_type=hparams.training_type,
                      max_length=hparams.max_length, 
                      test_run=hparams.test_run) as dc:
        tokenizer = dc.prepare_tokenizer()
        for i in range(len(dc)):
            data = dc[i]
            print(f'--({i})--')
            text = tokenizer.decode(data)
            print(text)

def main_update(args):
    os.system('pip3 uninstall -y kogitune')
    os.system('pip3 install -U git+https://github.com/kuramitsulab/kogitune.git')


def main():
    # メインのパーサーを作成
    parser = argparse.ArgumentParser(description='kogitune 🦊')

    # サブコマンドのパーサーを作成
    subparsers = parser.add_subparsers(title='subcommands', 
                                       description='valid subcommands', 
                                       help='additional help')
    # 'foo' サブコマンド
    store_parser = subparsers.add_parser('store', help='foo help')
    store_parser = argparse.ArgumentParser(description="papertown_store")
    store_parser.add_argument("files", type=str, nargs="+", help="files")
    store_parser.add_argument("--desc", type=str, default=None)
    store_parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    store_parser.add_argument("--store_path", default=None)
    store_parser.add_argument("--block_size", type=int, default=None)
    store_parser.add_argument("--type", type=str, default='')
    store_parser.add_argument("--format", default="simple")
    store_parser.add_argument("--split", default="train")
    store_parser.add_argument("--split_args", type=_parse_args, default=None)
    store_parser.add_argument("--sep", type=str, default=None)
    store_parser.add_argument("--output", type=str, default=None)
    store_parser.add_argument("--N", "-N", type=int, default=-1)
    store_parser.add_argument("--shuffle", type=_tobool, default=True)
    store_parser.add_argument("--random_seed", type=int, default=42)
    store_parser.add_argument("--verbose", type=_tobool, default=True)
    store_parser.add_argument("--histogram", type=_tobool, default=False)
    store_parser.add_argument("--num_works", type=int, default=0)
    store_parser.set_defaults(func=main_store)

    # 'dump' サブコマンド
    dump_parser = subparsers.add_parser('dump', help='dump help')
    dump_parser.add_argument("urls", type=str, nargs="+", help="urls")
    dump_parser.add_argument("--max_length", type=int, default=512)
    dump_parser.add_argument("--training_type", type=str, default='')
    dump_parser.add_argument("--test_run", type=int, default=10)
    dump_parser.set_defaults(func=main_dump)


    # 'update' サブコマンド
    update_parser = subparsers.add_parser('update', help='bar help')
    update_parser.set_defaults(func=main_update)

    # 引数をパースして対応する関数を実行
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
