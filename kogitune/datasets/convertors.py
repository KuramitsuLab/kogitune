from .commons import *
from .utils_datasets import load_train_dataset
from .templates import load_template

def convert_as_dict(dataset, template, data: dict):
    if 'in' in data and 'out' in data:
        ins = data['in']
        outs = data['out']
        for sample in dataset:
            ins.append(template.create_prompt(sample))
            outs.append(template.create_output(sample))
    else:
        texts = data['text']
        for sample in dataset:
            texts.append(template.create_instruction(sample))
    return data

def convert_to_DataFrame(data, aargs):
    import pandas as pd
    df = pd.DataFrame(data)
    if aargs['shuffle|=False']:
        df = df.sample(frac=1, random_state=aargs['random_state|=42']).reset_index(drop=True)
    return df

def save_as_json(data, output_file, aargs):
    df = convert_to_DataFrame(data, aargs)
    df.to_json(output_file, force_ascii=False, orient='records', lines=True)

def save_as_csv(data, output_file, aargs):
    df = convert_to_DataFrame(data, aargs)
    df.to_csv(output_file, encoding='utf-8-sig', index=False)

def save_as_parquet(data, output_file, aargs):
    df = convert_to_DataFrame(data, aargs)
    df.to_parquet(output_file, compression='zstd', index=False)

def save_to_disk(data, output_path. aargs):
    from datasets import Dataset
    dataset = Dataset.from_dict(data)
    dataset.save_to_disk(output_path)

def convert_dataset_cli(**kwargs):
    with adhoc.from_kwargs(**kwargs) as aargs:
        dataset = load_train_dataset(aargs=aargs)
        template = load_template(sample=dataset[0], aargs=aargs)
        datatype = aargs['datatype|=finetune']
        if datatype=='finetune':
            data = {'in': [], 'out': []}
        else:
            data = {'text': []}
        output_file = aargs['output_file|output_path']
        if output_file.endswith('.jsonl') or output_file.endswith('.json'):
            save_as_json(data, output_file, aargs)
        elif output_file.endswith('.csv'):
            save_as_csv(data, output_file, aargs)
        elif output_file.endswith('.csv'):
            save_as_csv(data, output_file, aargs)

# def less_cli(**kwargs):
#     with adhoc.from_kwargs(**kwargs) as aargs:
#         dataset = load_train_dataset(aargs=aargs)
#         for sample in dataset: