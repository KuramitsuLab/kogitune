import os
import kogitune.adhocs as adhoc

def update_cli(**kwargs):
    adhoc.print('KOGITUNEを最新版に更新します。\npip3 install -U git+https://github.com/kuramitsulab/kogitune.git')
    os.system('pip3 uninstall -y kogitune')
    os.system('pip3 install -U -q git+https://github.com/kuramitsulab/kogitune.git')

def beta_cli(**kwargs):
    adhoc.print('KOGITUNEをベータ版に更新します。\npip3 install -U git+https://github.com/kkuramitsu/kogitune.git')
    os.system('pip3 uninstall -y kogitune')
    os.system('pip3 install -U -q git+https://github.com/kkuramitsu/kogitune.git')

def split_lines_cli(**kwargs):
    from kogitune.stores.files import split_lines_cli
    split_lines_cli(**kwargs)

def rename_linenum_cli(**kwargs):
    from kogitune.stores.files import rename_linenum_cli
    rename_linenum_cli(**kwargs)

def train_bpe_cli(**kwargs):
    from kogitune.stores.unigrams import train_bpe_cli
    train_bpe_cli(**kwargs)

def train_unigram_cli(**kwargs):
    from kogitune.stores.unigrams import train_unigram_cli
    train_unigram_cli(**kwargs)

def train_spm_cli(**kwargs):
    from kogitune.stores.unigrams import train_spm_cli
    train_spm_cli(**kwargs)

## filter 系

def filter_cli(**kwargs):
    from kogitune.filters.filters import filter_cli
    filter_cli(**kwargs)
    
def replace_cli(**kwargs):
    from kogitune.filters.replaces import replace_cli
    replace_cli(**kwargs)

def filter_maxmin_cli(**kwargs):
    from kogitune.filters.maxmins import filter_maxmin_cli
    filter_maxmin_cli(**kwargs)

def filter_langset_cli(**kwargs):
    from kogitune.filters.languages import filter_langset_cli
    filter_langset_cli(**kwargs)

## store 系

def pack_cli(**kwargs):
    from .stores import store_files
    with adhoc.from_kwargs(**kwargs) as aargs:
        files = aargs['files|!!ファイルを一つ以上与えてください']
        store_files(files, skip_validation=False)

def store_cli(**kwargs):
    from .stores import store_files
    with adhoc.from_kwargs(**kwargs) as aargs:
        files = aargs['files|!!ファイルを一つ以上与えてください']
        store_files(files, skip_validation=False)

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
        for i in adhoc.tqdm(range(len(ds))):
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
        adhoc.print(f'ダウンロード時間: {time.time()-start} s')
        ds = Dataset.from_dict(ds_dict).with_format("torch")
        print(ds)
        output_path = dc.aargs['output_path|!freezed_dataset']
        ds.save_to_disk(output_path)
        print(FREEZE.format(output_path))

def token_stat_cli(**kwargs):
    import pandas as pd
    from .trainers import DatasetRecipe
    with DatasetRecipe(prefetch=0, **kwargs) as dc:
        dc.with_format("numpy")
        tokenizer = dc.get_tokenizer()
        token_ids = list(range(0, tokenizer.vocab_size))
        vocabs = tokenizer.convert_ids_to_tokens(token_ids)
        counts = [0] * tokenizer.vocab_size
        ds = dc.get_train_dataset()
        for i in adhoc.tqdm(range(len(ds)), desc='counting tokens'):
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
        adhoc.print(f"トークンの出現回数を output_file='{output_file}' に保存しました。ふむふむ")

def scratch_cli(**kwargs):
    from kogitune.trainers.scratch import generate_scratch
    generate_scratch(**kwargs)


def pretrain_cli(**kwargs):
    import torch
    torch.backends.cuda.matmul.allow_tf32=True
    from kogitune.trainers import DatasetComposer

    with DatasetComposer(**kwargs) as dc:
        dc.train()

def data_cli(**kwargs):
    from kogitune.metrics import load_data
    with adhoc.from_kwargs(**kwargs) as aargs:
        datalist = load_data(aargs)

def test_model_cli(**kwargs):
    from kogitune.metrics import load_model
    IPSUM='Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'

    with adhoc.from_kwargs(**kwargs) as aargs:
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

def launch(subcommand, **kwargs):
    fname = f'{subcommand}_cli'
    if '.' in fname:
        cls = adhoc.load_class(fname)
    else:
        cls = adhoc.load_class(f'kogitune.cli.{fname}')
    cls(**kwargs)

def main():
    # メインのパーサーを作成
    namespace = globals()
    subcommands = [name.replace('_cli', '') for name in namespace.keys() if name.endswith('_cli')]
    with adhoc.parse_main_args(subcommands=subcommands) as aargs:
        aargs.errors = 'main'
        cmd = aargs['subcommand']
        funcname = f'{cmd}_cli'
        namespace[funcname]()

if __name__ == '__main__':
    main()
