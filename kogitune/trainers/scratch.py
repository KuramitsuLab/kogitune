import torch
torch.backends.cuda.matmul.allow_tf32=True

#from ..commons import *

from ..adhoc_args import AdhocArguments, format_unit, verbose_print, configurable_tokenizer, adhoc

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

def print_model_structure(model):
    num_dict={}
    for name, param in model.named_parameters():
        num_dict[name]=param.numel()
    repl_parts=[
        "model",
        "layers",
        "weight",
        "mlp",
        "self_attn",
    ]
    def is_integer(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    name_set=[]
    for original_name in num_dict:
        name=original_name.split(".")
        name=[i for i in name if i not in repl_parts]
        name=[i for i in name if not is_integer(i)]
        name_set.append(".".join(name))

    #主要なレイヤーグループの表示
    #print(set(name_set))

    #パラメータ数の計算
    name_group_dict={}
    all_params=0
    for k, v in num_dict.items():
        found=False
        for group_name in name_set:
            if group_name in k:
                if group_name not in name_group_dict:
                    name_group_dict[group_name] = v
                else:
                    name_group_dict[group_name] += v
                all_params += v
                found=True
                break
        if not found:
            print(k," not found")
    #print(name_group_dict)
    import pandas as pd
    df=pd.DataFrame.from_dict(name_group_dict,orient="index")
    df.columns=["params"]
    df["ratio"]=df["params"]/all_params*100
    print(df)


def print_gpu_utilization():
    try:
        from pynvml import nvmlInit,nvmlDeviceGetHandleByIndex,nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {format_unit(info.used, scale=1024)}iB.")
    except:
        pass

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


"""


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

def new_Llama2(max_length=2048, 
               n_dims=128, n_heads=8, n_groups = None, n_layers=28, 
               intermediate_size=4096, rms_norm_eps=1e-6,
               tokenizer=DEFAULT_TOKENIZER):
    from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig

    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, legacy=False, trust_remote_code=True, use_fast=False)
    if n_groups is None:
        n_groups = n_heads

    config = LlamaConfig(
        vocab_size = len(tokenizer),
        pad_token_id = tokenizer.pad_token_id,
        bos_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        max_position_embeddings=max_length, #トークン数
        hidden_size=n_dims * n_heads,
        num_attention_heads = n_heads, #8
        num_key_value_heads = n_groups,
        num_hidden_layers = n_layers, #28
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps
    )

    model = LlamaForCausalLM(config)
    print_model(model)
    print_model_structure(model)
    return model
"""

### new version

def configurable_config_kwargs(tokenizer, **kwargs):
    from ..adhoc_args import AdhocArguments
    with AdhocArguments.from_main(**kwargs) as aargs:
        aargs.used('model_type')
        num_attention_heads = aargs['num_attention_heads|n_heads|=4']
        if 'hidden_size' in kwargs:
            hidden_size = aargs['intermediate_size|=1024']
        else:
            hidden_size = aargs['head_dim|n_dims|=32'] * num_attention_heads
        config_kwargs = dict(
            vocab_size = tokenizer.vocab_size,
            pad_token_id = tokenizer.pad_token_id,
            bos_token_id = tokenizer.bos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            num_attention_heads = num_attention_heads,
            hidden_size = hidden_size,
            intermediate_size = aargs['intermediate_size|=512'],
            num_hidden_layers = aargs['num_hidden_layers|n_layers|=12'],
            num_key_value_heads = aargs['num_key_value_heads|group_heads|n_groups|=4'],
            max_position_embeddings = aargs['max_position_embeddings|=2048'],
        )
        aargs.copy_to('num_key_value_heads|group_heads|n_groups', config_kwargs)
        aargs.copy_to('hidden_act', config_kwargs)
        aargs.copy_to('rms_norm_eps', config_kwargs)
        aargs.copy_to('rope_theta', config_kwargs)
        aargs.copy_to('tie_word_embeddings', config_kwargs)
        aargs.copy_to('attention_dropout', config_kwargs)
        aargs.copy_to('attention_bias', config_kwargs) # Gemma, Lamma
        aargs.copy_to('sliding_window', config_kwargs) # Mistral
        aargs.copy_to('partial_rotary_factor', config_kwargs) # StableLmConfig
    print(config_kwargs)
    return config_kwargs

def generate_scratch_gpt2(**kwargs):
    from transformers import GPT2LMHeadModel, GPT2Config
    config = GPT2Config(
        vocab_size = kwargs['vocab_size'],
        bos_token_id = kwargs['bos_token_id'],
        eos_token_id = kwargs['eos_token_id'],
        pad_token_id = kwargs['pad_token_id'],
        n_positions = kwargs['max_position_embeddings'],
        n_ctx=kwargs['max_position_embeddings'],
        n_embd=kwargs['hidden_size'],
        n_head=kwargs['num_attention_heads'],
        n_layer=kwargs['num_hidden_layers'],
        n_inner=kwargs['intermediate_size'],
    )
    model = GPT2LMHeadModel(config)
    print_model(model)
    print_model_structure(model)
    return model


def generate_scratch_gptneox(**kwargs):
    from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
    config = GPTNeoXConfig(**kwargs)
    model = GPTNeoXForCausalLM(config)
    print_model(model)
    print_model_structure(model)
    return model

def generate_scratch_llama2(**kwargs):
    from transformers import LlamaForCausalLM, LlamaConfig
    config = LlamaConfig(**kwargs)
    model = LlamaForCausalLM(config)
    print_model(model)
    print_model_structure(model)
    return model

def generate_scratch_stablelm(**kwargs):
    from transformers import StableLmForCausalLM, StableLmConfig
    config = StableLmConfig(**kwargs)
    model = StableLmForCausalLM(config)
    print_model(model)
    print_model_structure(model)
    return model

def generate_scratch_mistral(**kwargs):
    from transformers import MistralForCausalLM, MistralConfig
    config = MistralConfig(**kwargs)
    model = MistralForCausalLM(config)
    print_model(model)
    print_model_structure(model)
    return model

def generate_scratch_gemma(**kwargs):
    from transformers import GemmaForCausalLM, GemmaConfig
    config = GemmaConfig(**kwargs)
    model = GemmaForCausalLM(config)
    print_model(model)
    print_model_structure(model)
    return model    


def configurable_scratch(**kwargs):
    tokenizer = configurable_tokenizer(tokenizer=kwargs.get('tokenizer'))
    config_kwargs = configurable_config_kwargs(tokenizer, **kwargs)
    model_type = kwargs.get('model_type', 'llama2')
    adhoc.setlog('scratch', model_type=model_type, config=config_kwargs)

    ns = globals()
    name = f'generate_scratch_{model_type}'
    if name in ns:
        model = ns[name](**config_kwargs)
    else:
        required = [k.replace('generate_scratch_', '') for k, _ in ns.items() if k.startswith('generate_scratch_')]
        adhoc.warn(f'モデル種類({model_type})が不明だよ. ', unknown=model_type, required=required)
        adhoc.print('とりあえず、llama2で生成してみます')
        model = generate_scratch_llama2(**config_kwargs)

    output_path = kwargs.get('scratch_output_path', 'scratch')
    if output_path:
        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)

    return model
    
