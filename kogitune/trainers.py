from typing import List
import os
import torch
import argparse

torch.backends.cuda.matmul.allow_tf32=True
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelForCausalLM
from kogitune import PretrainComposer
# from kogitune.scratch import print_summary
from .commons import get_environ

#https://github.com/pacman100/DHS-LLM-Workshop/tree/main/personal_copilot/training

def init_wandb():
    try:
        import wandb
        entity = get_environ('KG_WANDB', None)
        if entity:
            project = get_environ('KG_PROJECT', None)
            wandb.init(
                entity=entity,
                project=get_environ('KG_PROJECT', 'kogitune'),
                name=get_environ('KG_WANDB_NAME', f'kogitune{os.getpid()}'),
            )
    except:
        pass

def parse_hparams(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default="bigcode/starcoderplus")
    parser.add_argument("--dataset_name", type=str, default="smangrul/hf-stack-v1")
    parser.add_argument("--subset", type=str, default="data")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--size_valid_set", type=int, default=4000)
    parser.add_argument("--test_size", type=float, default=0.005)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--data_column", type=str, default="content")

    parser.add_argument("--seq_length", type=int, default=8192)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    parser.add_argument("--fim_rate", type=float, default=0)
    parser.add_argument("--fim_spm_rate", type=float, default=0)

    parser.add_argument("--use_peft_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--lora_dropout", type=float, default=0)
    parser.add_argument("--lora_target_modules", type=str, default=None)

    parser.add_argument("--use_flash_attn", action="store_true")

    parser.add_argument("--use_4bit_qunatization", action="store_true")
    parser.add_argument("--use_nested_quant", action="store_true")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16")

    parser.add_argument("--use_8bit_qunatization", action="store_true")

    parser.add_argument("--push_to_hub", action="store_true")

    return parser.parse_args()



def train_clm(model_path:str, urls: List[str]):
    init_wandb()
    model = AutoModelForCausalLM.from_pretrained(model_path)
    args = TrainingArguments(
        output_dir=get_environ('KG_OUTPUT_DIR', './output'),
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        auto_find_batch_size=True,  # バッチサイズ自動
        do_eval=False,
        logging_steps=10, #train/train_loss1000回→WandB
        gradient_accumulation_steps=512,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=100,
        lr_scheduler_type="constant",
        learning_rate=3e-4, #TODO: 論文から探す
        save_steps=200,
        save_total_limit=3,
        fp16=True,
    )
    block_size = getint_environ('KG_BLOCK_SIZE', 512)
    with PretrainComposer(url_list=urls, block_size=block_size) as composer:
        data_collator = composer.get_collator()
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=composer,
            data_collator=data_collator,
        )
        result = trainer.train()
        output_path=get_environ('KG_OUTPUT_PATH', f'trained{os.getpid()}') #保存先
        model.save_pretrained(output_path)
        composer.get_tokenizer().save_pretrained(output_path)

