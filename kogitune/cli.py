import os
from tqdm import tqdm

from .commons import *
from .adhocargs import adhoc_parse_arguments, AdhocArguments
from .file_utils import basename_from_url

def main_maxmin(aargs=None):
    from .filters.scores import maxmin
    url_list = aargs['files']
    if url_list is None or len(url_list) == 0:
        aargs.raise_files('ファイルの指定が一つ以上必要です。')
    maxmin(url_list, aargs=aargs)


def main_store(args=None):
    from .stores import split_to_store
    url_list = args['files']
    if url_list is None or len(url_list) == 0:
        args.raise_files('ファイルの指定が一つ以上必要です。')
    split_to_store(url_list, skip_validation=False, args=args)

def main_head(args: AdhocArguments):
    from .trainers import DatasetComposer
    url_list = args['files']
    if url_list is None or len(url_list) == 0:
        args.raise_files('データセットへのパスが一つ以上必要です。')
    start = args['start|=0']
    N = args['head|N|batch|=1024']

    with DatasetComposer(url_list, args=args) as dc:
        dc.with_format('numpy')
        tokenizer = dc.get_tokenizer()
        ds = dc.get_train_dataset()
        for i in range(start, start+N):
            example = ds[i]
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
    from .trainers import DatasetComposer
    url_list = args['files']
    if url_list is None or len(url_list) == 0:
        args.raise_files('データセットへのパスが一つ以上必要です。')
    basename = basename_from_url(url_list)

    input_ids = []
    attention_mask = []
    labels=[]
    with DatasetComposer(url_list, args=args) as dc:
        dc.with_format("tensor")
        ds = dc.get_train_dataset()
        for i in tqdm(range(len(ds)), desc=basename):
            example=ds[i]
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
    output_path = args['output_path']
    if output_path is None:
        output_path = f'dataset_{basename}'
    ds.save_to_disk(output_path)
    print(FREEZE.format(output_path))

def main_histogram(args):
    import pandas as pd
    from .trainers import DatasetComposer
    url_list = args['files']
    if url_list is None or len(url_list) == 0:
        args.raise_files('データストアへのパスが一つ以上必要です。')
    
    with DatasetComposer(url_list, args=args) as dc:
        dc.with_format("numpy")
        tokenizer = dc.get_tokenizer()
        token_ids = list(range(0, tokenizer.vocab_size))
        vocabs = tokenizer.convert_ids_to_tokens(token_ids)
        counts = [0] * tokenizer.vocab_size
        # csv_file = f'{store_path.replace("/", "_")}.csv'
        ds = dc.get_train_dataset()
        for i in tqdm(range(len(ds)), desc='counting tokens'):
            example = ds[i]
            for token_id in example['input_ids']:
                counts[token_id] += 1
            if 'labels' in example:
                for token_id in example['labels']:
                    counts[token_id] += 1
    df = pd.DataFrame({'tokens': vocabs, 'counts': counts})
    print(df['counts'].describe())
    output_file = args['output_file|output_path']
    if output_file is None:
        output_file = basename_from_url(url_list, prefix='histogram_', ext='csv')
    df.to_csv(args.output_file)
    verbose_print(f"字句の出現頻度を'{output_file}'に保存しました。")

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
    args = adhoc_parse_arguments(subcommands='maxmin|store|head|freeze|histogram|linenum|update')
    main_func = args.find_function(args['subcommand'], prefix='main')
    main_func(args)
    args.check_unused()

if __name__ == '__main__':
    main()
