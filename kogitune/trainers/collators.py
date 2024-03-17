import numpy as np
import torch
from ..commons import CHUNK_MAGIC

class TextBlockCollator(object):
    def __init__(self, max_length, aargs):
        self.max_length = max_length
        self.is_seq2seq = aargs['datatype|data_type|=text'] == 'seq2seq'

    def __call__(self, data):
        return torch.tensor(data[:self.max_length].astype(np.int64), dtype=torch.long)

class NumpyCollator(object):
    def __init__(self, max_length, aargs):
        self.max_length = max_length
        self.is_seq2seq = aargs['datatype|data_type|=text'] == 'seq2seq'

    def __call__(self, data):
        return {
            "input_ids": data,
        }

class TensorCollator(object):
    def __init__(self, max_length, aargs):
        self.max_length = max_length
        self.is_seq2seq = aargs['datatype|data_type|=text'] == 'seq2seq'

    def __call__(self, data):
        if self.is_seq2seq:
            version = data[0] % CHUNK_MAGIC
            if version == 1:
                index = (data[0] // CHUNK_MAGIC) + 1
                inputs = data[1:index]
                labels = data[index:]
                return {
                    "input_ids": torch.tensor(inputs.astype(np.int64), dtype=torch.long),
                    "attention_mask": torch.ones(len(inputs), dtype=torch.long),
                    "labels": torch.tensor(labels.astype(np.int64), dtype=torch.long),
                }
            else:
                data = data[1:]
        return {
            "input_ids": torch.tensor(data.astype(np.int64), dtype=torch.long),
            "attention_mask": torch.ones(len(data), dtype=torch.long),
        }
