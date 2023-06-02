import argparse
import logging
from collections import OrderedDict
from pathlib import Path

import yaml
import numpy as np

from modules import utils

model = None
tokenizer = None
model_name = "None"
model_type = None
lora_names = []
soft_prompt_tensor = None
soft_prompt = False

# Chat variables
history = {'internal': [], 'visible': []}
character = 'None'
stop_everything = False
processing_message = '*Is typing...*'

# UI elements (buttons, sliders, HTML, etc)
gradio = {}

# For keeping the values of UI elements on page reload
persistent_interface_state = {}

input_params = []  # Generation input parameters
reload_inputs = []  # Parameters for reloading the chat interface

# For restarting the interface
need_restart = False

settings = {
    'autoload_model': True,
    'max_new_tokens': 200,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 2000,
    'seed': -1,
    'character': 'None',
    'name1': 'You',
    'name2': 'Assistant',
    'context': 'This is a conversation with your Assistant. It is a computer program designed to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information.',
    'greeting': '',
    'turn_template': '',
    'custom_stopping_strings': '',
    'stop_at_newline': False,
    'add_bos_token': True,
    'ban_eos_token': False,
    'skip_special_tokens': True,
    'truncation_length': 2048,
    'truncation_length_min': 0,
    'truncation_length_max': 8192,
    'mode': 'chat',
    'chat_style': 'cai-chat',
    'instruction_template': 'None',
    'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
    'chat_prompt_size': 2048,
    'chat_prompt_size_min': 0,
    'chat_prompt_size_max': 2048,
    'chat_generation_attempts': 1,
    'chat_generation_attempts_min': 1,
    'chat_generation_attempts_max': 10,
    'default_extensions': [],
    'chat_default_extensions': ["gallery"],
    'presets': {
        'default': 'Default',
        '.*(alpaca|llama|llava)': "LLaMA-Precise",
        '.*pygmalion': 'NovelAI-Storywriter',
        '.*RWKV': 'Naive',
        '.*moss': 'MOSS',
    },
    'prompts': {
        'default': 'QA',
        '.*(gpt4chan|gpt-4chan|4chan)': 'GPT-4chan',
    }
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))

# Basic settings
parser.add_argument('--model', type=str, help='Name of the model to load by default.')
parser.add_argument("--model-dir", type=str, default='models/', help="Path to directory with all the models")
parser.add_argument("--lora-dir", type=str, default='loras/', help="Path to directory with all the loras")
parser.add_argument('--extensions', type=str, nargs="+", help='The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.')
# parser.add_argument('--verbose', action='store_true', help='Print the prompts to the terminal.')
parser.add_argument('-l', '--log-level', type=str, default='INFO', help='The log level to set.')


# LoRA Training Settings
parser.add_argument('--lora_name', type=str, help='Name of the LoRA to train.')
parser.add_argument('--lora_to_load', type=str, help='The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces.')
parser.add_argument("--always_override", action='store_true', help="Override Existing Files")
parser.add_argument("--save_steps", type=int, default=0, help="Save every n steps. ")
parser.add_argument("--micro_batch_size", type=int, default=4, choices=range(1, 128+1, 1), metavar="[1-128]", help="Per-device batch size (NOTE: multiple devices not yet implemented). Increasing this will increase VRAM usage.")
parser.add_argument("--batch_size", type=int, default=128, choices=range(0,1024+1, 4), metavar="[0-1024]", help="Global batch size. The two batch sizes together determine gradient accumulation (gradientAccum = batch / microBatch). Higher gradient accum values lead to better quality training.")
parser.add_argument("--epochs", type=int, default=3, help="Number of times every entry in the dataset should be fed into training. So 1 means feed each item in once, 5 means feed it in five times, etc.")
parser.add_argument("--learning_rate", type=str, default="3e-4", help="Learning rate, in scientific notation. 3e-4 is a good starting base point. 1e-2 is extremely high, 1e-6 is extremely low.")
parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=['linear', 'constant', 'constant_with_warmup', 'cosine', 'cosine_with_restarts', 'polynomial', 'inverse_sqrt'], help='Learning rate scheduler - defines how the learning rate changes over time. "Constant" means never change, "linear" means to go in a straight line from the learning rate down to 0, cosine follows a curve, etc.')
parser.add_argument("--lora_rank", type=int, default=32, choices=range(0, 1024+1, 4), metavar="[0-1024]", help="LoRA Rank, or dimension count. Higher values produce a larger file with better control over the model\'s content. Smaller values produce a smaller file with less overall control. Small values like 4 or 8 are great for stylistic guidance, higher values like 128 or 256 are good for teaching content upgrades, extremely high values (1024+) are difficult to train but may improve fine-detail learning for large datasets. Higher ranks also require higher VRAM.")
parser.add_argument("--lora_alpha", type=int, default=64, choices=range(0, 2048+1, 4), metavar="[0-2048]", help="LoRA Alpha. This divided by the rank becomes the scaling of the LoRA. Higher means stronger. A good standard value is twice your Rank.")
parser.add_argument("--lora_dropout", type=float, default=0.05, choices=np.arange(0,1+0.1,0.025), metavar="[0-1]",help="Percentage probability for dropout of LoRA layers. This can help reduce overfitting. Most users should leave at default. minimum=0.0, maximum=1.0, step=0.025.")
parser.add_argument("--cutoff_len", type=int, default=256, choices=range(0, 2048+1, 32), metavar="[0-2048]", help="Cutoff length for text input. Essentially, how long of a line of text to feed in at a time. Higher values require drastically more VRAM.")
parser.add_argument("--dataset", type=str, choices=utils.get_datasets('training/datasets', 'json'), default='None', help="The dataset file to use for training.")
parser.add_argument("--eval_dataset", type=str, choices= utils.get_datasets('training/datasets', 'json'), default='None', help="The (optional) dataset file used to evaluate the model after training.")
parser.add_argument("--format", type=str, default=None, choices=utils.get_datasets('training/formats', 'json'), help="")
parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every n steps. If an evaluation dataset is given, test it every time this many steps pass.")
parser.add_argument("--raw_text_file", type=str, choices=utils.get_datasets('training/datasets', 'txt'), default='None', help="The raw text file to use for training.")
parser.add_argument("--overlap_len", type=int, default=128, choices=range(0, 512+1, 128), metavar="[0-512]", help="Overlap length - ie how many tokens from the prior chunk of text to include into the next chunk. (The chunks themselves will be of a size determined by Cutoff Length below). Setting overlap to exactly half the cutoff length may be ideal.")
parser.add_argument("--newline_favor_len", type=int, default=128, choices=range(0, 512+1, 16), metavar="[0-512]", help="Prefer Newline Cut Length. Length (in characters, not tokens) of the maximum distance to shift an overlap cut by to ensure chunks cut at newlines. If too low, cuts may occur in the middle of lines.")
parser.add_argument("--higher_rank_limit", action="store_true", help="Enable higher ranks. If checked, changes Rank/Alpha slider above to go much higher. This will not work without a datacenter-class GPU.'")
parser.add_argument("--warmup_steps", type=int, default=100, help="For this many steps at the start, the learning rate will be lower than normal. This helps the trainer prepare the model and precompute statistics to improve the quality of training after the start.")
parser.add_argument("--optimizer", type=str, default="adamw_torch", choices=['adamw_hf', 'adamw_torch', 'adamw_torch_fused', 'adamw_torch_xla', 'adamw_apex_fused', 'adafactor', 'adamw_bnb_8bit', 'adamw_anyprecision', 'sgd', 'adagrad'], help="Different optimizer implementation options, for advanced users. Effects of different options are not well documented yet.")
parser.add_argument("--hard_cut_string", type=str, default="\\n\\n\\n", help="String that indicates a hard cut between text parts. Helps prevent unwanted overlap.")

# Accelerate/transformers
parser.add_argument('--cpu', action='store_true', help='Use the CPU to generate text. Warning: Training on CPU is extremely slow.')
parser.add_argument('--auto-devices', action='store_true', help='Automatically split the model across the available GPU(s) and CPU.')
parser.add_argument('--gpu-memory', type=str, nargs="+", help='Maxmimum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values in MiB like --gpu-memory 3500MiB.')
parser.add_argument('--cpu-memory', type=str, help='Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.')
parser.add_argument('--disk', action='store_true', help='If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.')
parser.add_argument('--disk-cache-dir', type=str, default="cache", help='Directory to save the disk cache to. Defaults to "cache".')
parser.add_argument('--load-in-8bit', action='store_true', help='Load the model with 8-bit precision.')
parser.add_argument('--load-in-4bit', action='store_true', help='Load the model with 4-bit precision.')
parser.add_argument('--bf16', action='store_true', help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')
parser.add_argument('--xformers', action='store_true', help="Use xformer's memory efficient attention. This should increase your tokens/s.")
parser.add_argument('--sdp-attention', action='store_true', help="Use torch 2.0's sdp attention.")
parser.add_argument('--trust-remote-code', action='store_true', help="Set trust_remote_code=True while loading a model. Necessary for ChatGLM.")

# GPTQ
parser.add_argument('--wbits', type=int, default=0, help='Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported.')
parser.add_argument('--model_type', type=str, help='Model type of pre-quantized model. Currently LLaMA, OPT, and GPT-J are supported.')
parser.add_argument('--groupsize', type=int, default=-1, help='Group size.')
parser.add_argument('--pre_layer', type=int, nargs="+", help='The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models. For multi-gpu, write the numbers separated by spaces, eg --pre_layer 30 60.')
parser.add_argument('--checkpoint', type=str, help='The path to the quantized checkpoint file. If not specified, it will be automatically detected.')
parser.add_argument('--monkey-patch', action='store_true', help='Apply the monkey patch for using LoRAs with quantized models.')
parser.add_argument('--quant_attn', action='store_true', help='(triton) Enable quant attention.')
parser.add_argument('--warmup_autotune', action='store_true', help='(triton) Enable warmup autotune.')
parser.add_argument('--fused_mlp', action='store_true', help='(triton) Enable fused mlp.')

# AutoGPTQ
parser.add_argument('--autogptq', action='store_true', help='Use AutoGPTQ for loading quantized models instead of the internal GPTQ loader.')
parser.add_argument('--triton', action='store_true', help='Use triton.')

# DeepSpeed
parser.add_argument('--deepspeed', action='store_true', help='Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration.')
parser.add_argument('--nvme-offload-dir', type=str, help='DeepSpeed: Directory to use for ZeRO-3 NVME offloading.')
parser.add_argument('--local_rank', type=int, default=0, help='DeepSpeed: Optional argument for distributed setups.')

# FlexGen
parser.add_argument('--flexgen', action='store_true', help='Enable the use of FlexGen offloading.')
parser.add_argument('--percent', type=int, nargs="+", default=[0, 100, 100, 0, 100, 0], help='FlexGen: allocation percentages. Must be 6 numbers separated by spaces (default: 0, 100, 100, 0, 100, 0).')
parser.add_argument("--compress-weight", action="store_true", help="FlexGen: activate weight compression.")
parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=True, default=True, help="FlexGen: whether to pin weights (setting this to False reduces CPU memory by 20%%).")



args = parser.parse_args()
args_defaults = parser.parse_args([])

# Deprecation warnings for parameters that have been renamed
deprecated_dict = {}
for k in deprecated_dict:
    if getattr(args, k) != deprecated_dict[k][1]:
        logging.warning(f"--{k} is deprecated and will be removed. Use --{deprecated_dict[k][0]} instead.")
        setattr(args, deprecated_dict[k][0], getattr(args, k))

def add_extension(name):
    if args.extensions is None:
        args.extensions = [name]
    elif 'api' not in args.extensions:
        args.extensions.append(name)


def is_chat():
    return args.chat


# Loading model-specific settings
with Path(f'{args.model_dir}/config.yaml') as p:
    if p.exists():
        model_config = yaml.safe_load(open(p, 'r').read())
    else:
        model_config = {}

# Applying user-defined model settings
with Path(f'{args.model_dir}/config-user.yaml') as p:
    if p.exists():
        user_config = yaml.safe_load(open(p, 'r').read())
        for k in user_config:
            if k in model_config:
                model_config[k].update(user_config[k])
            else:
                model_config[k] = user_config[k]

model_config = OrderedDict(model_config)
