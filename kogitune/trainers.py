import torch

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
