from .local_commons import *

def load_train_dataset(**kwargs):
    import datasets
    with AdhocArguments.from_main(**kwargs) as aargs:
        dataset_path = aargs['train_dataset|train_data|dataset_path|dataset']
        split='train'
        dataset_args = aargs['dataset_args|dataset_config']
        if dataset_args is None:
            dataset_path, dataset_args = parse_path_arguments(dataset_path)
        if dataset_path.endswith('.jsonl'):
            data_files = dataset_path.split('|')
            dataset = datasets.load_dataset('json', data_files=data_files, **dataset_args)
        else:
            if 'split' not in dataset_args:
                dataset_args['split'] = split
            dataset = datasets.load_dataset(dataset_path, **dataset_args)
        if isinstance(dataset, datasets.dataset_dict.DatasetDict):
            if split in dataset:
                adhoc.warn(f'splitの指定がないから、split={split}で進めるよ。')
                dataset = dataset[split]
            else:
                adhoc.warn(f'splitの指定が必要だよ')
        return dataset


def load_test_dataset(**kwargs):
    import datasets
    with AdhocArguments.from_main(**kwargs) as aargs:
        dataset_path = aargs['test_dataset|test_data|dataset_path|dataset']
        split='test'
        dataset_args = aargs['dataset_args|dataset_config']
        if dataset_args is None:
            dataset_path, dataset_args = parse_path_arguments(dataset_path)
        if dataset_path.endswith('.jsonl'):
            dataset = datasets.load_dataset('json', data_files=dataset_path, **dataset_args)
        else:
            if 'split' not in dataset_args:
                dataset_args['split'] = split
            dataset = datasets.load_dataset(dataset_path, **dataset_args)
        if isinstance(dataset, datasets.dataset_dict.DatasetDict):
            if split in dataset:
                adhoc.warn(f'splitの指定がないから、split={split}で進めるよ。')
                dataset = dataset[split]
            else:
                adhoc.warn(f'splitの指定が必要だよ')
        return dataset
