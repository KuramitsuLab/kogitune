from typing import Union, List
import torch
import numpy as np

from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from .tokenizers import find_extra_ids, find_newline_token_id
from .composers import DataComposer, CHUNK_MAGIC

class PretrainComposer(DataComposer):
    def __init__(self, url_list: Union[str, List[str]], max_length:int, **kwargs):
        kwargs['data_type'] = 'text'
        DataComposer.__init__(self, url_list, max_length=max_length, **kwargs)

    def get_collator(self, model):
        tokenizer = self.get_tokenizer()
        return DataCollatorForLanguageModeling(tokenizer, 
                                               pad_to_multiple_of=8, 
                                               mlm=False)

def build_inputs_for_instruct(data, max_length):
    # version = data[0] % CHUNK_MAGIC
    return torch.tensor(data[1:max_length+1].astype(np.int64), dtype=torch.long)

def build_inputs_attn_for_instruct(data, max_length):
    # version = data[0] % CHUNK_MAGIC
    input_ids = torch.tensor(data[1:max_length+1].astype(np.int64), dtype=torch.long)
    attention_mask=torch.ones(len(data), dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask, 
    }

class FinetuneComposer(DataComposer):
    def __init__(self, url_list: Union[str, List[str]], max_length:int, **kwargs):
        kwargs['data_type'] = 'seq2seq'
        DataComposer.__init__(self, url_list, max_length, **kwargs)
        if kwargs.get('use_attention_mask', True):
            self.build_fn = kwargs.get('build_fn', build_inputs_attn_for_instruct)
        else:
            self.build_fn = kwargs.get('build_fn', build_inputs_for_instruct)

    def get_collator(self, model):
        tokenizer = self.get_tokenizer()
        return DataCollatorForLanguageModeling(tokenizer, 
                                               pad_to_multiple_of=8,
                                               mlm=False)

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
    def __init__(self, url_list: Union[str, List[str]], max_length:int, **kwargs):
        kwargs['data_type'] = 'text'
        DataComposer.__init__(self, url_list, max_length, **kwargs)
        tokenizer = self.get_tokenizer()
        self.build_fn = MSP(tokenizer=tokenizer)

    def get_collator(self, model):
        tokenizer = self.get_tokenizer()
        return DataCollatorForSeq2Seq(tokenizer, 
                                      model=model,
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
    def __init__(self, url_list: Union[str, List[str]], max_length:int, **kwargs):
        kwargs['data_type'] = 'seq2seq'
        DataComposer.__init__(self, url_list, max_length, **kwargs)
        self.build_fn = Seq2seq()

    def get_collator(self, model):
        tokenizer = self.get_tokenizer()
        return DataCollatorForSeq2Seq(tokenizer, model=model,
                                      pad_to_multiple_of=8)



"""

class DP(object):
    def __init__(self, tokenizer, lambda_=20):
        self.lambda_ = lambda_
        self.eos_token_id = tokenizer.eos_token_id
        self.extra_ids = find_extra_ids(tokenizer)
        self.newline_id = find_newline_token_id(tokenizer)

    def __call__(self, data, max_length):
        index = 0
        start = 0
        size = min(max_length, len(data))
        for i, length in enumerate(np.random.poisson(self.lambda_, 1000), start=1):
            start = start + max(1, length) * i
            if start >= size:
                break
            if data[start] != self.eos_token_id or data[start] != self.newline_id:
                data[start] = self.extra_ids[index]
                index+=1
        return torch.tensor(data[:max_length].astype(np.int64), dtype=torch.long)



def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0


def train_model(DataComposer, model, args: TrainingArguments, use_flash=False):

    if use_flash:
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except:
            use_flash=False

    args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        auto_find_batch_size=True,  # バッチサイズ自動
        do_eval=False,
        logging_steps=1000,
#        gradient_accumulation_steps=1,
        num_train_epochs=1,
        weight_decay=0.1,
#        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4, #TODO: 論文から探す
#        save_steps=5_000,
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dummy_dataset(max_length),
        data_collator=data_collator,
    )
    result = trainer.train()
    print_summary(result, use_flash)
    # モデルを保存 output_path に保存します
    if use_flash:
        model = BetterTransformer.reverse(model)
    output_path='trained'
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_utilization()

CHUNK_MAGIC = 8

class Seq2SeqCollator(object):
    def __call__(self, data, max_length):
        version = data[0] % CHUNK_MAGIC
        assert version == 1
        if version == 1:
            index = (data[0] // CHUNK_MAGIC) + 2
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



class AlpacaCollator(object):
    
    def __call__(data, max_length):
        version = data[0] % CHUNK_MAGIC
        index = (data[0] // CHUNK_MAGIC) + 1
        instructs = data[1:index]
        outputs = data[index:]
        if len(instructs)+len(outputs) > max_length:
            instructs = instructs[:(max_length -len(labels))]
            inputs = np.concatenate(instructs, outputs)
        else:
            inputs = data[1:max_length+1]
        labels = inputs.copy()
        labels[:len(instructs)] = [-100] * len(instructs)
        print({"input_ids": inputs, "labels": inputs})
        return {"input_ids": data[1:max_length+1], "labels": labels}

def get_collator(tokenizer, max_length):
    def collator(batch):
        batch = [{ key: value[:max_length] for key, value in sample.items() } for sample in batch ]
        print('batch', batch)
        batch = tokenizer.pad(batch, padding=True)
        batch["labels"] = [ list(e) + [-100] * (len(batch["input_ids"][0]) - len(e)) for e in batch["labels"] ]
        batch = { key: torch.tensor(value) for key, value in batch.items() }
        return batch
    return collator

"""
