import gc
import torch
torch.backends.cuda.matmul.allow_tf32=True
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

import numpy as np
from datasets import Dataset

from .commons import *

def count_parameters(model)->int:
    """
    モデルのパラメータ数を数える

    model: モデル
    return パラメータ数
    """
    return sum(p.numel() for p in model.parameters())


def print_model(model):
    n_parameters=count_parameters(model)
    config = model.config
    print(f'Parameters: {n_parameters} {format_unit(n_parameters)}', end=' ')
    if hasattr(config, 'max_position_embeddings'):
        print(f"max_length: {config.max_position_embeddings}", end=' ')
    elif hasattr(config, "n_positions"):
        print(f"max_length: {config.n_positions}", end=' ')
    print(f"vocab_size: {config.vocab_size}")

    if hasattr(config, 'd_kv'):  # T5
        print(f"d_model: {model.config.d_model}", end=' ')
        print(f"d_kv: {model.config.d_kv}", end=' ')
        print(f"d_ff: {model.config.d_ff}", end=' ')
        print(f"num_heads: {model.config.num_heads}", end=' ')
        print(f"num_layers: {model.config.num_layers}+{model.config.num_decoder_layers}")
        print(config)
    elif hasattr(config, 'n_embd'): #GPT-2
        print(f"hidden_size: {config.n_embd}", end=' ')
        print(f"intermediate_size: {config.n_inner}", end=' ')
        print(f"n_dims: {config.n_embd//config.n_head}", end=' ')
        print(f"n_heads: {config.n_head}", end=' ')
        print(f"n_layers: {config.n_layer}")
        print(config)
    elif hasattr(config, 'hidden_size'): #GPT-NeoX
        print(f"hidden_size: {config.hidden_size}", end=' ')
        print(f"intermediate_size: {config.intermediate_size}", end=' ')
        print(f"n_dims: {config.hidden_size//model.config.num_attention_heads}", end=' ')
        print(f"n_heads: {config.num_attention_heads}", end=' ')
        print(f"n_layers: {config.num_hidden_layers}")
    else:
        print(config)

def print_gpu_utilization():
    try:
        from pynvml import nvmlInit,nvmlDeviceGetHandleByIndex,nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {format_unit(info.used, scale=1024)}iB.")
    except:
        pass

def dummy_dataset(max_length, dataset_size=1024):
    dummy_data = {
        "input_ids": (np.arange(100, dataset_size*max_length+100) % 15000).reshape((dataset_size, max_length))
    }
    ds = Dataset.from_dict(dummy_data)
    ds.set_format("pt")
    return ds

def print_summary(result, use_flash=False):
    m = result.metrics
    print(f"Time: {m['train_runtime']:.2f}  {format_unit(m['train_runtime'], scale=60)}", end=' ')
    print(f"Samples/second: {m['train_samples_per_second']:.2f} FlashAttn={use_flash}")
    print(f"Global step: {result.global_step} batch_size: {1024//result.global_step}", end=' ')
    if 'total_flos' in m:
        print(f"FLOS: {m['total_flos']} {format_unit(m['total_flos'])} Loss: {m['train_loss']:.5f}")
    else:
        print(f"Loss: {m['train_loss']:.5f}")
    print_gpu_utilization()


def train_model(model, tokenizer, max_length, use_fp16=False, use_flash=False):

    if use_flash:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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


def new_T5(d_model=256, d_kv=32, d_ff=1024, n_head=6, n_layers=12, max_length=2048, tokenizer=DEFAULT_TOKENIZER):
    from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False, trust_remote_code=True, use_fast=False)

    config = T5Config(
        vocab_size = len(tokenizer),
        d_model = d_model,
        d_kv = d_kv,
        d_ff = d_ff,
        num_layers = n_layers,
        num_decoder_layers = None,
        num_heads = n_head,
        relative_attention_num_buckets = 32,
        relative_attention_max_distance = 128,
        dropout_rate = 0.1,
        layer_norm_epsilon = 1e-06,
        initializer_factor = 1.0,
        feed_forward_proj = 'gated-gelu',
        is_encoder_decoder = True,
        use_cache = True,
        tokenizer_class = 'T5Tokenizer',
        tie_word_embeddings = False,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id,
        decoder_start_token_id=0,
    )

    model = T5ForConditionalGeneration(config)
    print_model(model)
    return model


# GPT-2

def new_GPT2(max_length=2048, n_dims=64, n_heads=6, n_layers=12, intermediate_size=1024, tokenizer=DEFAULT_TOKENIZER):
    from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False, trust_remote_code=True, use_fast=False)

    config = GPT2Config(
        vocab_size = len(tokenizer),
        bos_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id,
        n_positions=max_length,
        n_ctx=max_length,
        n_embd=n_dims*n_heads,
        n_head=n_heads,
        n_layer=n_layers,
        n_inner=intermediate_size,
    )

    model = GPT2LMHeadModel(config)
    print_model(model)
    return model

# GPTNeoX

def new_GPTNeoX(max_length=2048, n_dims=64, n_heads=12, n_layers=12, intermediate_size=2048, tokenizer=DEFAULT_TOKENIZER):
    from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False, trust_remote_code=True, use_fast=False)

    config = GPTNeoXConfig(
        vocab_size = len(tokenizer),
        pad_token_id = tokenizer.pad_token_id,
        bos_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        max_position_embeddings=max_length, #トークン数
        hidden_size=n_dims * n_heads,
        num_attention_heads = n_heads, #8
        num_hidden_layers = n_layers, #28
        intermediate_size=intermediate_size,
    )

    model = GPTNeoXForCausalLM(config)
    print_model(model)
    return model


## new_Lamma2

def new_Llama2(max_length=2048, n_dims=128, n_heads=8, n_layers=28, intermediate_size=4096, tokenizer=DEFAULT_TOKENIZER):
    from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False, trust_remote_code=True, use_fast=False)

    config = LlamaConfig(
        vocab_size = len(tokenizer),
        pad_token_id = tokenizer.pad_token_id,
        bos_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        max_position_embeddings=max_length, #トークン数
        hidden_size=n_dims * n_heads,
        num_attention_heads = n_heads, #8
        num_hidden_layers = n_layers, #28
        intermediate_size=intermediate_size,
    )

    model = LlamaForCausalLM(config)
    print_model(model)
    return model



