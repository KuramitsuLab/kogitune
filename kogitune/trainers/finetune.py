from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from ..adhoc_args import AdhocArguments, adhoc, parse_path_arguments, configurable_tokenizer
from .gpus import bf16_is_available

def load_train_dataset(**kwargs):
    import datasets
    with AdhocArguments.from_main(**kwargs) as aargs:
        dataset_path = aargs['train_dataset|train_data']
        if dataset_path:
            split='train'
        else:
            dataset_path = aargs['test_dataset|test_data']
            if dataset_path:
                split='test'
            else:
                dataset_path = aargs['dataset|!!']
                split=None
        dataset_args = aargs['dataset_args|dataset_config']
        if dataset_args is None:
            dataset_path, dataset_args = parse_path_arguments(dataset_path)
        if dataset_path.endswith('.jsonl'):
            dataset = datasets.load_dataset('json', dataset_path, **dataset_args)
        else:
            if split and 'split' not in dataset_args:
                dataset_args['split'] = split
            dataset = datasets.load_dataset(dataset_path, **dataset_args)
        if isinstance(dataset, datasets.dataset_dict.DatasetDict):
            if 'test' in dataset:
                adhoc.warn(f'splitの指定がないから、split=testで進めるよ。')
                dataset = dataset['test']
            else:
                adhoc.warn(f'splitの指定が必要だよ')
        return dataset

def load_train_model(**kwargs):
    from transformers import AutoModelForCausalLM
    with AdhocArguments.from_main(**kwargs) as aargs:
        model_path = aargs['model_path|!!']
        model_args = aargs['model_args|model_config']
        if model_args is None:
            model_path, model_args = parse_path_arguments(model_path)
        if 'use_auth_token' not in model_args:
            model_args['use_auth_token'] = aargs['hf_token']
        if 'trust_remote_code' not in model_args:
            model_args['trust_remote_code'] = True
        # MacOS 上でエラーになる
        # if 'device_map' not in model_args:
        #     model_args['device_map'] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
        return model

def configure_train_args(**kwargs):
    with AdhocArguments.from_main(**kwargs) as aargs:
        global_batch_size = aargs['global_batch_size|batch_size|=8']
        device_batch_size = aargs['device_batch_size|=8']
        gas = global_batch_size // device_batch_size
        adhoc.notice('バッチサイズ', 
                     global_batch_size=global_batch_size, 
                     device_batch_size=device_batch_size,
                     gradient_accumulation_steps=gas,
        )
        overwrite_output_dir = 'resume_from_checkpoint' not in aargs
        bf16_enabled = aargs[f'bf16|={bf16_is_available()}']
        fp16_enabled = False
        optim='adamw_torch'
        if torch.cuda.is_available():
            if not bf16_enabled:
                fp16_enabled=True
            optim='adamw_torch_fused'
        train_args = dict(
            output_dir=aargs['output_dir|=output'],
            overwrite_output_dir=aargs[f'overwrite_output_dir|={overwrite_output_dir}'],
            per_device_train_batch_size=aargs[f'per_device_train_batch_size|={device_batch_size}'],
            gradient_accumulation_steps=aargs[f'gradient_accumulation_steps|={gas}'],
            # per_device_eval_batch_size=64,
            auto_find_batch_size=aargs['auto_find_batch_size|=False'],  # バッチサイズ自動
            do_eval=aargs['do_eval|=False'],
            # evaluation_strategy='steps',
            # eval_steps=50,
            optim=aargs[f'optim|={optim}'],
            learning_rate=aargs['learning_rate|=4e-4'], 
            weight_decay=aargs['weight_decay|=0.1'],
            adam_beta1=aargs['adam_beta1|=0.9'],
            adam_beta2=aargs['adam_beta2|=0.999'],
            adam_epsilon=aargs['adam_epsilon|=1e-8'],
            max_grad_norm=aargs['max_grad_norm|=1.0'],
            num_train_epochs=aargs['num_train_epochs|=2'],
            max_steps=aargs['max_steps|=-1'],
            lr_scheduler_type=aargs['lr_scheduler_type|=cosine'],
            logging_steps=aargs['logging_steps|=10'],
            dataloader_pin_memory=False,
            save_steps=aargs['save_steps|=1000'],
            save_total_limit=aargs['save_total_limit'],
            save_only_model=aargs['save_only_model|=False'],
            neftune_noise_alpha=aargs['neftune_noise_alpha'],
            torch_compile=aargs['torch_compile|=False'],
            bf16=bf16_enabled, 
            fp16=fp16_enabled,
        )
        return train_args

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        combined_text = f"{example['instruction'][i]} {example['input'][i]}"
        text = f"### instruction\n{combined_text}\n### output\n{example['output'][i]}"
        output_texts.append(text)
    return output_texts

def finetune_cli(**kwargs):
    from transformers import TrainingArguments
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    with AdhocArguments.from_main(import_to_main=True, **kwargs) as aargs:
        train_args = configure_train_args(aargs=aargs)
        # dataset = load_dataset("kunishou/amenokaku-code-instruct")
        dataset = load_train_dataset(aargs=aargs)
        print(type(dataset[0]), dataset[0])
        print(dataset['instruction'][:5])
        model = load_train_model(aargs=aargs)
        tokenizer = configurable_tokenizer(aargs=aargs)

        response_template_with_context = "\n### output\n"  # We added context here: "\n". This is enough for this tokenizer
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

        data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

        trainer = SFTTrainer(
            model=model,
            args=TrainingArguments(**train_args),
            train_dataset=dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model('tuned')