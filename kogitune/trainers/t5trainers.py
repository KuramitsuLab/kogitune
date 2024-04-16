import numpy as np
import torch 

from ..stores.tokenizers import find_extra_ids, find_newline_token_id, load_tokenizer
from .old_composers import DataComposer, CHUNK_MAGIC
from transformers import DataCollatorForSeq2Seq


class MSP(object):
    def __init__(self, tokenizer, lambda_=3):
        self.lambda_ = lambda_
        self.eos_token_id = tokenizer.eos_token_id
        self.extra_ids = find_extra_ids(tokenizer)
        self.newline_id = find_newline_token_id(tokenizer)

    def __call__(self, data, max_length):
        data = data.tolist()
        inputs = tokens = []
        outputs = masked = []
        start=0
        index=0
        for length in np.random.poisson(self.lambda_, 1000):
            end = start+max(1, length)
            data_part = data[start:end]
            tokens.extend(data_part)
            if self.eos_token_id in data_part or self.newline_id in data_part:
                masked.extend(data_part)
            else:
                masked.append(self.extra_ids[(index//2)%100])
                index += 1
                tokens,masked = masked, tokens
            start = end
            if start > len(data):
                break
        return {
            "input_ids": torch.tensor(inputs[:max_length//2], dtype=torch.long),
            "labels": torch.tensor(outputs[:max_length//2], dtype=torch.long),
        }

class T5PretrainComposer(DataComposer):
    def __init__(self, url_list, block_size, **kwargs):
        kwargs['training_type'] = 'pre'
        kwargs['split'] = 'train'
        DataComposer.__init__(self, url_list, block_size=block_size, **kwargs)
        tokenizer = self.get_tokenizer()
        self.build_fn = MSP(tokenizer=tokenizer)

    def get_collator(self, model):
        tokenizer = self.get_tokenizer()
        return DataCollatorForSeq2Seq(tokenizer, model=model,
                                      pad_to_multiple_of=8)

class Seq2seq(object):
    def __call__(self, data, max_length):
        version = data[0] % CHUNK_MAGIC
        assert version == 1
        if version == 1:
            index = (data[0] // CHUNK_MAGIC) + 1
            inputs = data[1:index]
            labels = data[index:]
            if len(inputs)+len(labels) > max_length:
                inputs = inputs[:(max_length -len(labels))]
        else:
            # Prefix Language Model
            inputs = data[1:max_length//2].copy()
            labels = data[1:max_length]
        if len(inputs) > 0:
            inputs[-1] = data[-1] # eos
        else:
            data[0] = data[-1]
            inputs = data[:1]
        return {
            "input_ids": torch.tensor(inputs.astype(np.int64), dtype=torch.long),
            "attention_mask": torch.ones(len(inputs), dtype=torch.long),
            "labels": torch.tensor(labels.astype(np.int64), dtype=torch.long),
        }

class T5FinetuneComposer(DataComposer):
    def __init__(self, url_list, **kwargs):
        kwargs['training_type'] = 'seq2seq'
        if 'block_size' in kwargs: # ブロックサイズは、事前学習用のパラメータ
            kwargs['max_length'] = kwargs['block_size']
            del kwargs['block_size']
        DataComposer.__init__(self, url_list, **kwargs)
        self.build_fn = Seq2seq()

    def get_collator(self, model):
        tokenizer = self.get_tokenizer()
        return DataCollatorForSeq2Seq(tokenizer, model=model,
                                      pad_to_multiple_of=8)




