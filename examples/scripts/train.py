from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

from kogitune import DataComposer, load_tokenizer
from kogitune.new_model import new_T5, new_GPT2, new_GPTNeoX, new_Llama2

from args_parser import ArgsHandler, initialize_wandb, create_model, add_noise

args_handler = ArgsHandler()
args, composer_args, training_args= args_handler.get_args()

initialize_wandb(args)

tokenizer = load_tokenizer()
model = create_model(globals().get(f"new_{args['model']}"), args, tokenizer)
composer_args = add_noise(args, composer_args, tokenizer)

with DataComposer(**composer_args) as dataset: 
 
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    result = trainer.train()
    output_path = args['trained_dir']
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
