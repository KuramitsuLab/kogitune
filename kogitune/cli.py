import os

from .adhoc_args import adhoc_parse_arguments, AdhocArguments, verbose_print, configurable_tqdm, adhoc

def update_cli(**kwargs):
    verbose_print('KOGITUNEを最新版に更新します。\npip3 install -U git+https://github.com/kuramitsulab/kogitune.git')
    os.system('pip3 uninstall -y kogitune')
    os.system('pip3 install -U git+https://github.com/kuramitsulab/kogitune.git')

def beta_cli(**kwargs):
    verbose_print('KOGITUNEをベータ版に更新します。\npip3 install -U git+https://github.com/kkuramitsu/kogitune.git')
    os.system('pip3 uninstall -y kogitune')
    os.system('pip3 install -U git+https://github.com/kkuramitsu/kogitune.git')

def count_lines_cli(**kwargs):
    from kogitune.utils_file import extract_linenum_from_filename, rename_with_linenum, get_linenum
    with AdhocArguments.from_main(import_to_main=True, **kwargs) as aargs:
        for file in aargs['files']:
            n = extract_linenum_from_filename(file)
            if n is None:
                n = get_linenum(file)
                file = rename_with_linenum(file, n)

def maxmin_cli(**kwargs):
    from kogitune.filters import maxmin
    with AdhocArguments.from_main(import_to_main=True, **kwargs) as aargs:
        files = aargs['files|!!ファイルを一つ以上与えてください']
        score_path = aargs['score|eval|!!scoreを設定してください']
        output_path = aargs['output_file|output']
        kwargs = aargs.as_kwargs(
            record_key = aargs[f'record_key|=score'],
            histogram_sample = aargs['histogram_sample|N|n|=10000'] # 大きさの調整
        )
        if 'score' in kwargs:
            del kwargs['score']
        text_filter = maxmin(score_path, **kwargs)
        text_filter.from_jsonl(files, output_path=output_path, num_workers=1)

def filter_cli(**kwargs):
    from kogitune.filters import load_filter
    with AdhocArguments.from_main(import_to_main=True, **kwargs) as aargs:
        files = aargs['files|!!ファイルを一つ以上与えてください']
        filter_config = aargs['filter_config|!!filter_configを設定してください']
        text_filter = load_filter(filter_config)
        N=aargs['head|N|=-1']
        num_workers=['num_workers|=1']
        output_file = aargs['output_file|output']
        if output_file is None:
            aargs.verbose_print('output_fileの指定がないから、少しだけ処理して表示するよ')
        text_filter.from_jsonl(files, N=N, output_path=output_file, num_workers=num_workers)

## store 系

def store_cli(**kwargs):
    from .stores import store_files
    with AdhocArguments.from_main(import_to_main=True, **kwargs) as aargs:
        files = aargs['files|!!ファイルを一つ以上与えてください']
        store_files(files, skip_validation=False, aargs=aargs)

def head_cli(**kwargs):
    from .trainers import DatasetComposer
    with DatasetComposer(**kwargs) as dc:
        dc.with_format('numpy')
        start = dc.aargs['start|=0']
        N = dc.aargs['head|N|batch|=1024']
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

def freeze_cli(**kwargs):
    import time
    from datasets import Dataset
    from .trainers import DatasetComposer
    input_ids = []
    attention_mask = []
    labels=[]
    start = time.time()
    with DatasetComposer(prefetch=0, **kwargs) as dc:
        dc.with_format("tensor")
        ds = dc.get_train_dataset()
        for i in configurable_tqdm(range(len(ds))):
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
        verbose_print(f'ダウンロード時間: {time.time()-start} s')
        ds = Dataset.from_dict(ds_dict).with_format("torch")
        print(ds)
        output_path = dc.aargs['output_path|!freezed_dataset']
        ds.save_to_disk(output_path)
        print(FREEZE.format(output_path))

def token_stat_cli(**kwargs):
    import pandas as pd
    from .trainers import DatasetComposer
    with DatasetComposer(prefetch=0, **kwargs) as dc:
        dc.with_format("numpy")
        tokenizer = dc.get_tokenizer()
        token_ids = list(range(0, tokenizer.vocab_size))
        vocabs = tokenizer.convert_ids_to_tokens(token_ids)
        counts = [0] * tokenizer.vocab_size
        ds = dc.get_train_dataset()
        for i in configurable_tqdm(range(len(ds)), desc='counting tokens'):
            example = ds[i]
            for token_id in example['input_ids']:
                counts[token_id] += 1
            if 'labels' in example:
                for token_id in example['labels']:
                    counts[token_id] += 1
        output_file = dc.aargs['output_file|output|=token_stat.csv']
        df = pd.DataFrame({'tokens': vocabs, 'counts': counts})
        print(df['counts'].describe())
        df.to_csv(output_file)
        verbose_print(f"トークンの出現回数を output_file='{output_file}' に保存しました。ふむふむ")

def scratch_cli(**kwargs):
    from kogitune.trainers.scratch import configurable_scratch
    configurable_scratch(**kwargs)


def pretrain_cli(**kwargs):
    import torch
    torch.backends.cuda.matmul.allow_tf32=True
    from kogitune.trainers import DatasetComposer

    with DatasetComposer(**kwargs) as dc:
        dc.train()

def data_cli(**kwargs):
    from kogitune.metrics import load_data
    with AdhocArguments.from_main(import_to_main=True, **kwargs) as aargs:
        datalist = load_data(aargs)

def test_model_cli(**kwargs):
    from kogitune.metrics import load_model
    IPSUM='Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'

    with AdhocArguments.from_main(import_to_main=True, **kwargs) as aargs:
        model = load_model(aargs=aargs)
        print(model)
        prompt = aargs['test_prompt|prompt']
        if prompt is None:
            adhoc.print('test_prompt でプロンプトは変更できるよ')
            prompt=IPSUM
        output = model.generate_text(prompt)
        adhoc.warn(f'test_prompt="{prompt}"\n===\n{output}\n')

def convert_dataset_cli(**kwargs):
    from kogitune.datasets.convertors import convert_dataset_cli
    convert_dataset_cli(**kwargs)


def finetune_cli(**kwargs):
    from kogitune.trainers import finetune_cli
    finetune_cli(**kwargs)


def chain_eval_cli(**kwargs):
    from kogitune.metrics import chain_eval_cli
    chain_eval_cli(**kwargs)


def main():
    # メインのパーサーを作成
    namespace = globals()
    subcommands = [name.replace('_cli', '') for name in namespace.keys() if name.endswith('_cli')]
    aargs = adhoc_parse_arguments(subcommands=subcommands)
    cmd = aargs['subcommand']
    funcname = f'{cmd}_cli'
    namespace[funcname]()
    aargs.check_unused()

if __name__ == '__main__':
    main()
