import os

from kogitune.commons import *
from .adhocargs import adhoc_parse_arguments
from .trainers.old_composers import DataComposer
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

def main_store(args=None):
    from .stores import split_to_store

    split_to_store(args.files, validation=True, args=args)

def main_head(args):
    with DataComposer(args.urls, 
                      data_type=args.data_type,
                      max_length=args.max_length, 
                      test_run=args.test_run, prefetch=0) as dc:
        tokenizer = dc.prepare_tokenizer()
        for i in range(len(dc)):
            example = dc[i]
            if 'input_ids' in example:
                print(f'inputs[{i}]:', tokenizer.decode(example['input_ids']))
            if 'labels' in example:
                print(f'labels[{i}]:', tokenizer.decode(example['labels']))
            print('---')


FREEZE='''
from datasets import load_from_disk
ds = load_from_disk("{}")
'''

def main_freeze(args):
    from tqdm import tqdm
    from datasets import Dataset
    input_ids = []
    attention_mask = []
    labels=[]
    with DataComposer(args.urls, 
                      data_type=args.data_type,
                      max_length=args.max_length,
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
    ds.save_to_disk(args.output_path)
    print(FREEZE.format(args.output_path))

def main_histogram(args):
    import pandas as pd
    from tqdm import tqdm
    with DataComposer(args.urls, 
                      data_type=args.data_type,
                      max_length=args.max_length,
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
        if args.output_file is None:
            _, args = parse_url_args(args.urls[0], {})
            _, _, path = args['url_path'].rpartition('/')
            args.output_file = safe_new_file(f'histogram_{path}', 'csv')
            verbose_print(f"字句の出現頻度を'{args.output_file}'に保存しました。")
        df.to_csv(args.output_file)

def conv_txt_to_jsonl(file):
    from .file_utils import zopen, filelines
    import json
    newfile = file.replace('.txt', '.jsonl')
    with zopen(newfile, 'wt') as w:
        for line in filelines(file):
            line = line.replace('<nL>', '\n')
            print(json.dumps({'text': line}, ensure_ascii=False), file=w)
    verbose_print(f'"{newfile}"へ変換しました。')

def main_oldconv(args):
    for file in args.files:
        if file.endswith('.txt') or file.endswith('.txt.zst') or file.endswith('.txt.gz'):
            conv_txt_to_jsonl(file)

def main_linenum(args):
    from file_utils import extract_linenum_from_filename, rename_with_linenum, get_linenum
    for file in args['files']:
        n = extract_linenum_from_filename()
        if n is None:
            n = get_linenum(file)
            file = rename_with_linenum(file, n)


def main_update(args):
    args.verbose_print('pip3 install -U git+https://github.com/kuramitsulab/kogitune.git')
    os.system('pip3 uninstall -y kogitune')
    os.system('pip3 install -U git+https://github.com/kuramitsulab/kogitune.git')


def main():
    # メインのパーサーを作成
    args = adhoc_parse_arguments(
        subcommands='store|freeze|histogram|head|oldconv|update')

    main_func = args.find_function(args['subcommand'], prefix='main')
    main_func(args)


def main2():
    # 'store' サブコマンド
    if args['subcommand'] == 'store':
        main_store(args)

    # 'freeze' サブコマンド
    if args['subcommand'] == 'freeze':
        main_freeze(args)

    # 'histogram' サブコマンド
    if args['subcommand'] == 'histogram':
        main_histogram(args)

    # 'head' サブコマンド
    if args['subcommand'] == 'head':
        main_head(args)

    # 'conv' サブコマンド
    if args['subcommand'] == 'oldconv':
        main_oldconv(args)

    # 'update' サブコマンド
    if args['subcommand'] == 'update':
        main_update(args)

if __name__ == '__main__':
    main()
