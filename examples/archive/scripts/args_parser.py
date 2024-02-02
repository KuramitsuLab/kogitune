from transformers import TrainingArguments
import yaml
import argparse
# import wandb
from kogitune.trainers.old_composers import DP

class FileLoader:
    @staticmethod
    def load_from_yamlfile(filename):
        try:
            with open(filename, 'r') as file:
                config = yaml.safe_load(file)
                print("Loaded config from yaml file:", config)
                return config
        except Exception as e:
            print(f"Error loading config from {filename}: {e}")
            return {}
        
    @staticmethod 
    def load_urls_from_txtfile(filename):
        with open(filename, 'r') as file:
            urls = [line.strip() for line in file.readlines()]
            return urls
        
class ArgsHandler:
    def __init__(self):
        self.parser = self._get_parser()

    def _get_parser(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
        parser.add_argument('--deepspeed', type=str, help='Path to deepspeed config file')

        parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
        parser.add_argument("--urls", type=str, required=True, help="Path to the file containing URLs")

        parser.add_argument('--model', type=str, required=True, help="Select the model you want to use; available options are 'T5', 'GPT2', 'GPTNeoX', 'Llama2'.")
        parser.add_argument("--max_length", type=int, default=None)
        parser.add_argument("--n_dims", type=int, default=None)
        parser.add_argument("--n_heads", type=int, default=None)
        parser.add_argument("--n_layers", type=int, default=None)
        parser.add_argument("--intermediate_size", type=int, default=None)

        parser.add_argument("--name", type=str)
        parser.add_argument("--entity", type=str)
        parser.add_argument("--project", type=str)
        parser.add_argument("--group", type=str)

        parser.add_argument("--block_size", type=int, default=256, 
                    help="Number of steps for gradient accumulation. Also used for gradient_accumulation_steps. Max value is 2,048.")
        parser.add_argument("--logging_steps", type=int, default=2, 
                            help="Number of steps for logging. Also used for save_steps.")
        parser.add_argument("--learning_rate", type=float, default=5e-4)
        parser.add_argument("--num_train_epochs", type=int, default=3)
        
        parser.add_argument("--auto_find_batch_size", default=True)
        parser.add_argument("--per_device_train_batch_size", type=int, default=128)
        parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
        parser.add_argument("--do_eval", default=False)
        parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
        parser.add_argument("--fp16", default=True)
        parser.add_argument("--weight_decay", type=int, default=0.1)
        parser.add_argument("--save_total_limit", type=int, default=5)
        parser.add_argument("--overwrite_output_dir", default=True)
        parser.add_argument("--output_dir", type=str, default="./output", help="Directory where checkpoints will be saved.")
        parser.add_argument("--trained_dir", type=str, default="./trained", help="Directory where the trained model will be saved.")
        parser.add_argument("--test_run", type=int)

        parser.add_argument("--dp_lambda", type=int)

        return parser
    
    def _update_args_with_config(self, args, config_from_file):
        args_dict = vars(args)
        for key, value in config_from_file.items():
            if key == 'accumulation_steps':
                pass
            elif key not in args_dict or args_dict[key] is None or args_dict[key] == self.parser.get_default(key):
                args_dict[key] = value
        args_dict['accumulation_steps'] = args_dict['block_size']
        return args_dict
    
    def _create_composer_args(self, args_dict):
        composer_args = {
            'format': 'pre',
            'prefetch': 1,
            'url_list': FileLoader.load_urls_from_txtfile(args_dict['urls']),
            'block_size': args_dict['block_size'],
        }
        if args_dict['test_run'] is not None:
            composer_args['test_run'] = args_dict['test_run']
    
        return composer_args
        
    def _create_training_args(self, args_dict):
        training_args = TrainingArguments(
            gradient_accumulation_steps = args_dict['accumulation_steps'],
            logging_steps = args_dict['logging_steps'],
            save_steps = args_dict['logging_steps'],
            learning_rate = args_dict['learning_rate'],
            num_train_epochs = args_dict['num_train_epochs'],
            auto_find_batch_size = args_dict['auto_find_batch_size'],
            per_device_train_batch_size = args_dict['per_device_train_batch_size'],
            per_device_eval_batch_size = args_dict['per_device_eval_batch_size'],
            do_eval = args_dict['do_eval'],
            lr_scheduler_type = args_dict['lr_scheduler_type'],
            fp16 = args_dict['fp16'],
            weight_decay = args_dict['weight_decay'],
            save_total_limit = args_dict['save_total_limit'],
            overwrite_output_dir = args_dict['overwrite_output_dir'],
            output_dir = args_dict['output_dir']
        )
        return training_args

    def get_args(self):
        args = self.parser.parse_args()
        config_from_file = FileLoader.load_from_yamlfile(args.config)
        args_dict = self._update_args_with_config(args, config_from_file)
        composer_args = self._create_composer_args(args_dict)
        training_args = self._create_training_args(args_dict)
        return args_dict, composer_args, training_args

def initialize_wandb(args_dict):
    wandb_kwargs = {
            'entity': args_dict.get('entity'),
            'project': args_dict.get('project'),
            'name': args_dict.get('name'),
            'group': args_dict.get('group')
        }
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    wandb.init(**wandb_kwargs)
    wandb.alert(title="Job Started", text="動いたよ! Yay!")

def create_model(model_class, args_dict, tokenizer):
    model_kwargs = {
        'tokenizer': tokenizer,
        'max_length': args_dict.get('max_length'),
        'n_dims': args_dict.get('n_dims'),
        'n_heads': args_dict.get('n_heads'),
        'n_layers': args_dict.get('n_layers'),
        'intermediate_size': args_dict.get('intermediate_size')
    }
    return model_class(**{k: v for k, v in model_kwargs.items() if v is not None})

def add_noise(args_dict, composer_args, tokenizer):
    if args_dict['dp_lambda'] is not None:
        composer_args['build_fn'] = DP(tokenizer, lambda_=args_dict['dp_lambda'])
        print(f"Set lambda to {args_dict['dp_lambda']}")
    return composer_args

    
if __name__ == "__main__":
    args_handler = ArgsHandler()
    args = args_handler.get_args()
    print(args)
