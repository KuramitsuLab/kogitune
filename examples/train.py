from transformers import AutoModelForCausalLM, Trainer
import torch
torch.backends.cuda.matmul.allow_tf32=True

import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


from kogitune import PretrainComposer, load_tokenizer
from kogitune.scratch import new_T5, new_GPT2, new_GPTNeoX, new_Llama2
from kogitune.scratch import print_summary


from args_parser import ArgsHandler, initialize_wandb, create_model


args_handler = ArgsHandler()
args, composer_args, training_args= args_handler.get_args()

initialize_wandb(args)

tokenizer = load_tokenizer('llm-jp/llm-jp-1.3b-v1.0')
model = create_model(globals().get(f"new_{args['model']}"), args, tokenizer)

tokenizer.save_pretrained('scratch')
model.save_pretrained('scratch')

model = AutoModelForCausalLM.from_pretrained(
    'scratch',
)
with PretrainComposer(**composer_args) as dataset: 
 
    data_collator = dataset.get_collator(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    result = trainer.train()
    output_path = 'tinycodellama-jp-0.6b-10k' 
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)
    print_summary(result)
