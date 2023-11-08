import os
import argparse

from kogitune.commons import *
from .splitters import split_to_store
from .composers import DataComposer
from .file_utils import parse_url_args, safe_new_file

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
    # parser.add_argument("--histogram", type=_tobool, default=False)
    parser.add_argument("--num_works", type=int, default=0)
    
    hparams = parser.parse_args()  # hparams になる
    return hparams


def main_head(hparams):
    with DataComposer(hparams.urls, 
                      data_type=hparams.data_type,
                      max_length=hparams.max_length, 
                      test_run=hparams.test_run, prefetch=0) as dc:
        tokenizer = dc.prepare_tokenizer()
        for i in range(len(dc)):
            example = dc[i]
            if 'input_ids' in example:
                print(f'inputs[{i}]:', tokenizer.decode(example['input_ids']))
            if 'labels' in example:
                print(f'labels[{i}]:', tokenizer.decode(example['labels']))
            print('---')

def setup_head(parser):
    parser.add_argument("urls", type=str, nargs="+", help="urls")
    parser.add_argument("--data_type", type=str, choices=['text', 'seq2seq'], required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--test_run", type=int, default=10)
    parser.set_defaults(func=main_head)


FREEZE='''
from datasets import load_from_disk
ds = load_from_disk("{}")
'''

def main_freeze(hparams):
    from tqdm import tqdm
    from datasets import Dataset
    input_ids = []
    attention_mask = []
    labels=[]
    with DataComposer(hparams.urls, 
                      data_type=hparams.data_type,
                      max_length=hparams.max_length,
                      prefetch=0) as dc:
        for i in tqdm(range(len(dc))):
            example=dc[i]
            input_ids.append(example['input_ids'])
            if 'attention_mask' in example:
                attention_mask.append(example['attention_mask'])
            if 'labels' in example:
                labels.append(example['labels'])
    if len(labels) > 0:
        ds_dict = { "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels }
    elif len(attention_mask) > 0:
        ds_dict = { "input_ids": input_ids, "attention_mask": attention_mask}
    else:
        ds_dict = { "input_ids": input_ids}
    ds = Dataset.from_dict(ds_dict).with_format("torch")
    print(ds)
    ds.save_to_disk(hparams.output_path)
    print(FREEZE.format(hparams.output_path))

def setup_freeze(parser):
    parser.add_argument("urls", type=str, nargs="+", help="urls")
    parser.add_argument("--data_type", type=str, choices=['text', 'seq2seq'], required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_path", type=str, default='local_dataset')
    parser.set_defaults(func=main_freeze)

def main_histogram(hparams):
    import pandas as pd
    from tqdm import tqdm
    with DataComposer(hparams.urls, 
                      data_type=hparams.data_type,
                      max_length=hparams.max_length,
                      prefetch=0) as dc:
        tokenizer = dc.get_tokenizer()
        token_ids = list(range(0, tokenizer.vocab_size))
        vocabs = tokenizer.convert_ids_to_tokens(token_ids)
        counts = [0] * tokenizer.vocab_size
        # csv_file = f'{store_path.replace("/", "_")}.csv'

        for i in tqdm(range(len(dc)), desc='counting tokens'):
            example=dc[i]
            for token_id in example['input_ids']:
                counts[token_id] += 1
            if 'labels' in example:
                for token_id in example['labels']:
                    counts[token_id] += 1
        df = pd.DataFrame({'tokens': vocabs, 'counts': counts})
        print(df['counts'].describe())
        if hparams.output_file is None:
            _, args = parse_url_args(hparams.urls[0], {})
            _, _, path = args['url_path'].rpartition('/')
            hparams.output_file = safe_new_file(f'histogram_{path}', 'csv')
            verbose_print(f"字句の出現頻度を'{hparams.output_file}'に保存しました。")
        df.to_csv(hparams.output_file)

def setup_histogram(parser):
    parser.add_argument("urls", type=str, nargs="+", help="urls")
    parser.add_argument("--data_type", type=str, choices=['text', 'seq2seq'], required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_file", type=str, default=None)
    parser.set_defaults(func=main_histogram)



def main_update(args):
    os.system('pip3 uninstall -y kogitune')
    os.system('pip3 install -U git+https://github.com/kuramitsulab/kogitune.git')

def main_store(hparams=None):
    if hparams is None:
        hparams = setup_store()
    args = {k:v for k,v in vars(hparams).items() if v is not None}
    split_to_store(hparams.files, validation=True, args=args)

def setup_store(parser):
    parser.add_argument("files", type=str, nargs="+", help="files")
    parser.add_argument("--desc", type=str, default=None)
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--store_path", default=None)
    parser.add_argument("--max_length", type=int, required=True)
    parser.add_argument("--min_length", type=int, default=None)
    parser.add_argument("--data_type", type=str, choices=['text', 'seq2seq'])
    parser.add_argument("--format", default="simple")
    parser.add_argument("--split", default="train")
    parser.add_argument("--section", type=str, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--padding", type=int, default=None)
    # parser.add_argument("--split_args", type=_parse_args, default=None)
    parser.add_argument("--N", "-N", type=int, default=-1)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--verbose", type=_tobool, default=True)
    parser.add_argument("--histogram", type=_tobool, default=False)
    parser.add_argument("--num_works", type=int, default=0)
    parser.set_defaults(func=main_store)

def main():
    # メインのパーサーを作成
    parser = argparse.ArgumentParser(description='kogitune 🦊')

    # サブコマンドのパーサーを作成
    subparsers = parser.add_subparsers(title='subcommands', 
                                       description='valid subcommands', 
                                       help='additional help')

    # 'store' サブコマンド
    setup_store(subparsers.add_parser('store', help='store'))

    # 'freeze' サブコマンド
    setup_freeze(subparsers.add_parser('freeze', help='freeze'))

    # 'freeze' サブコマンド
    setup_histogram(subparsers.add_parser('histogram', help='histogram'))

    # 'head' サブコマンド
    setup_head(subparsers.add_parser('head', help='dump'))

    # 'update' サブコマンド
    update_parser = subparsers.add_parser('update', help='update')
    update_parser.set_defaults(func=main_update)

    # 引数をパースして対応する関数を実行
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
