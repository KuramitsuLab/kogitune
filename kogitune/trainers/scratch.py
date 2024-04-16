import torch
torch.backends.cuda.matmul.allow_tf32=True

import kogitune.adhocs as adhoc

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
    adhoc.p(flush=True)
    adhoc.p(f'Parameters: {n_parameters} {adhoc.format_unit(n_parameters)}', end=' ')
    if hasattr(config, 'max_position_embeddings'):
        adhoc.p(f"max_length: {config.max_position_embeddings}", end=' ')
    elif hasattr(config, "n_positions"):
        adhoc.p(f"max_length: {config.n_positions}", end=' ')
    adhoc.p(f"vocab_size: {config.vocab_size}")

    if hasattr(config, 'd_kv'):  # T5
        adhoc.p(f"d_model: {model.config.d_model}", end=' ')
        adhoc.p(f"d_kv: {model.config.d_kv}", end=' ')
        adhoc.p(f"d_ff: {model.config.d_ff}", end=' ')
        adhoc.p(f"num_heads: {model.config.num_heads}", end=' ')
        adhoc.p(f"num_layers: {model.config.num_layers}+{model.config.num_decoder_layers}")
        adhoc.p(config)
    elif hasattr(config, 'n_embd'): #GPT-2
        adhoc.p(f"hidden_size: {config.n_embd}", end=' ')
        adhoc.p(f"intermediate_size: {config.n_inner}", end=' ')
        adhoc.p(f"n_dims: {config.n_embd//config.n_head}", end=' ')
        adhoc.p(f"n_heads: {config.n_head}", end=' ')
        adhoc.p(f"n_layers: {config.n_layer}")
        adhoc.p(config)
    elif hasattr(config, 'hidden_size'): #GPT-NeoX
        adhoc.p(f"hidden_size: {config.hidden_size}", end=' ')
        adhoc.p(f"intermediate_size: {config.intermediate_size}", end=' ')
        adhoc.p(f"n_dims: {config.hidden_size//model.config.num_attention_heads}", end=' ')
        adhoc.p(f"n_heads: {config.num_attention_heads}", end=' ')
        adhoc.p(f"n_layers: {config.num_hidden_layers}")
    else:
        adhoc.p(config)

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

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
    name_set=[]
    for original_name in num_dict:
        name=original_name.split(".")
        name=[i for i in name if i not in repl_parts]
        name=[i for i in name if not is_integer(i)]
        name_set.append(".".join(name))

    #主要なレイヤーグループの表示
    adhoc.p(set(name_set))

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
    adhoc.p(df)


def print_gpu_utilization():
    try:
        from pynvml import nvmlInit,nvmlDeviceGetHandleByIndex,nvmlDeviceGetMemoryInfo
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"GPU memory occupied: {adhoc.format_unit(info.used, scale=1024)}iB.")
    except:
        pass

def print_summary(result, use_flash=False):
    m = result.metrics
    print(f"Time: {m['train_runtime']:.2f}  {adhoc.format_unit(m['train_runtime'], scale=60)}", end=' ')
    print(f"Samples/second: {m['train_samples_per_second']:.2f} FlashAttn={use_flash}")
    print(f"Global step: {result.global_step} batch_size: {1024//result.global_step}", end=' ')
    if 'total_flos' in m:
        print(f"FLOS: {m['total_flos']} {adhoc.format_unit(m['total_flos'])} Loss: {m['train_loss']:.5f}")
    else:
        print(f"Loss: {m['train_loss']:.5f}")
    print_gpu_utilization()

### new version

def extract_scratch_config(tokenizer, **kwargs):
    with adhoc.from_kwargs(**kwargs) as aargs:
        aargs.used('model_type')
        num_attention_heads = aargs['num_attention_heads|n_heads|=4']
        if 'hidden_size' in kwargs:
            hidden_size = aargs['hidden_size|=1024']
        else:
            hidden_size = aargs['head_dim|n_dims|=32'] * num_attention_heads
        scratch_config = dict(
            model_type = aargs['model_type|=llama2'],
            vocab_size = aargs[f'vocab_size|={tokenizer.vocab_size}'],
            pad_token_id = tokenizer.pad_token_id,
            bos_token_id = tokenizer.bos_token_id,
            eos_token_id = tokenizer.eos_token_id,
            num_attention_heads = num_attention_heads,
            hidden_size = hidden_size,
            intermediate_size = aargs['intermediate_size|=512'],
            num_hidden_layers = aargs['num_hidden_layers|n_layers|=12'],
            num_key_value_heads = aargs['num_key_value_heads|head_groups|n_groups|=4'],
            max_position_embeddings = aargs['max_position_embeddings|=4096'],
        )
        scratch_config_base = aargs['scratch_config|scratch_kwargs']
        if scratch_config_base:
            scratch_config_base.update(scratch_config)
            scratch_config = scratch_config_base
        adhoc.copy_dict_keys(aargs, scratch_config,
                      'num_key_value_heads|group_heads|n_groups',
                      'hidden_act', 'rms_norm_eps'
                      'rope_theta', 'tie_word_embeddings', 
                      'attention_dropout', 
                      'attention_bias', 
                      'sliding_window', 
                      'partial_rotary_factor')
    return scratch_config

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
    adhoc.check_kwargs(kwargs, MistralConfig)
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

def reduce_model_using_float16(model_path):
    from transformers import AutoModelForCausalLM
    ## なんか馬鹿らしいコードだけど、float16に変換　サイズが半分になる
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.save_pretrained(model_path)

def generate_scratch(tokenizer=None, **kwargs):
    with adhoc.from_kwargs(open_section='scratch', **kwargs) as aargs:
        tokenizer = adhoc.load_tokenizer(tokenizer=tokenizer)
        scratch_config = extract_scratch_config(tokenizer)
        model_type = scratch_config.get('model_type', 'llama2')
        adhoc.notice('新しいLLMを生成しています', scratch_config=scratch_config)

        ns = globals()
        name = f'generate_scratch_{model_type}'
        if name in ns:
            model = ns[name](**scratch_config)
        else:
            required = [k.replace('generate_scratch_', '') for k, _ in ns.items() if k.startswith('generate_scratch_')]
            adhoc.warn(f'model_type={model_type}は知らないよ', 
                    unknown_model_type=model_type, 
                    required_model_type=required, default_model_type='llama2')
            model = generate_scratch_llama2(**scratch_config)
        output_path = aargs.get('scratch_output_path', 'scratch')
        if output_path:
            tokenizer.save_pretrained(output_path)
            model.save_pretrained(output_path)
            reduce_model_using_float16(output_path)
        
    return model



