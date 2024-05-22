import datasets
from .commons import *

def is_jsonfile(data_files):
    if isinstance(data_files, list):
        for file in data_files:
            if '.json' not in file:
                return False
        return True
    return '.json' in str(data_files)
        
def load_dataset_as_json(data_files, **dataset_args):
    if isinstance(data_files, str):
        data_files = data_files.split('|')
    dataset = datasets.load_dataset('json', data_files=data_files, **dataset_args)
    return dataset   

def is_csvfile(data_files):
    if isinstance(data_files, list):
        for file in data_files:
            if '.csv' not in file:
                return False
        return True
    return '.csv' in str(data_files)
        
def load_dataset_as_csv(data_files, **dataset_args):
    if isinstance(data_files, str):
        data_files = data_files.split('|')
    dataset = datasets.load_dataset('json', data_files=data_files, **dataset_args)
    return dataset   

def load_dataset(default_split, **kwargs):
    with adhoc.from_kwargs(**kwargs) as aargs:
        dataset_path = aargs[f'{default_split}_dataset|{default_split}_data|dataset_path|dataset|!!']
        dataset_args = aargs['dataset_args|dataset_config']
        if dataset_args is None:
            dataset_path, dataset_args = adhoc.parse_path_args(dataset_path)
        if is_jsonfile(dataset_path):
            dataset = load_dataset_as_json(dataset_path, dataset_args)
        else:
            if 'split' not in dataset_args:
                dataset_args['split'] = default_split
            dataset = datasets.load_dataset(dataset_path, **dataset_args)
        if isinstance(dataset, datasets.dataset_dict.DatasetDict):
            if default_split in dataset:
                adhoc.warn(f'splitの指定がないから、split={default_split}で進めるよ。')
                dataset = dataset[default_split]
            else:
                adhoc.warn(f'splitの指定が必要だよ')
        return dataset

def load_train_dataset(**kwargs):
    return load_dataset('train', **kwargs)

def load_test_dataset(**kwargs):
    return load_dataset('test', **kwargs)

