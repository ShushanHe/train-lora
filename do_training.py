import logging
import os
import requests
import warnings
import modules.logging_colors

import importlib
import io
import json
import math
import os
import re
import sys
import time
import traceback
import zipfile
from datetime import datetime
from functools import partial
from pathlib import Path

import psutil
import torch
import yaml
from PIL import Image

from modules import shared, utils
from modules.train_lora import do_train
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model

def load_model_wrapper(selected_model, autoload=False):
    if not autoload:
        yield f"The settings for {selected_model} have been updated.\nClick on \"Load the model\" to load it."
        return

    if selected_model == 'None':
        yield "No model selected"
    else:
        try:
            yield f"Loading {selected_model}..."
            shared.model_name = selected_model
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(shared.model_name)

            yield f"Successfully loaded {selected_model}"
        except:
            yield traceback.format_exc()

def load_lora_wrapper(selected_loras):
    yield ("Applying the following LoRAs to {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("Successfuly applied the LoRAs")

def download_model_wrapper(repo_id):
    try:
        downloader = importlib.import_module("download-model")
        repo_id_parts = repo_id.split(":")
        model = repo_id_parts[0] if len(repo_id_parts) > 0 else repo_id
        branch = repo_id_parts[1] if len(repo_id_parts) > 1 else "main"
        check = False

        yield ("Cleaning up the model/branch names")
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)

        yield ("Getting the download links from Hugging Face")
        links, sha256, is_lora = downloader.get_download_links_from_huggingface(model, branch, text_only=False)

        yield ("Getting the output folder")
        output_folder = downloader.get_output_folder(model, branch, is_lora)

        if check:
            yield ("Checking previously downloaded files")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
        else:
            yield (f"Downloading files to {output_folder}")
            downloader.download_model_files(model, branch, links, sha256, output_folder, threads=1)
            yield ("Done!")
    except:
        yield traceback.format_exc()


def get_model_specific_settings(model):
    settings = shared.model_config
    model_settings = {}

    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    return model_settings


def load_model_specific_settings(model, state, return_dict=False):
    model_settings = get_model_specific_settings(model)
    for k in model_settings:
        if k in state:
            state[k] = model_settings[k]

    return state


def save_model_settings(model, state):
    if model == 'None':
        yield ("Not saving the settings because no model is loaded.")
        return

    with Path(f'{shared.args.model_dir}/config-user.yaml') as p:
        if p.exists():
            user_config = yaml.safe_load(open(p, 'r').read())
        else:
            user_config = {}

        model_regex = model + '$'  # For exact matches
        if model_regex not in user_config:
            user_config[model_regex] = {}

        for k in ui.list_model_elements():
            user_config[model_regex][k] = state[k]

        with open(p, 'w') as f:
            f.write(yaml.dump(user_config))

        yield (f"Settings for {model} saved to {p}")

if __name__ == "__main__":

    logging.basicConfig(level=shared.args.log_level.upper())
    # # Loading custom settings
    # settings_file = None
    # if shared.args.settings is not None and Path(shared.args.settings).exists():
    #     settings_file = Path(shared.args.settings)
    # elif Path('settings.json').exists():
    #     settings_file = Path('settings.json')

    # Set default model settings based on settings.json
    shared.model_config['.*'] = {
        'wbits': 'None',
        'model_type': 'None',
        'groupsize': 'None',
        'pre_layer': 0,
        'mode': shared.settings['mode'],
        'skip_special_tokens': shared.settings['skip_special_tokens'],
        'custom_stopping_strings': shared.settings['custom_stopping_strings'],
    }

    shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    available_models = utils.get_available_models()

    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Only one model is available
    elif len(available_models) == 1:
        shared.model_name = available_models[0]


    # If any model has been selected, load it
    if shared.model_name != 'None':
        model_settings = get_model_specific_settings(shared.model_name)
        shared.settings.update(model_settings)  # hijacking the interface defaults

        # Load the model
        shared.model, shared.tokenizer = load_model(shared.model_name)
        if shared.args.lora_to_load:
            add_lora_to_model(shared.args.lora_to_load)

        all_params = [
            shared.args.lora_name,
            shared.args.always_override,
            shared.args.save_steps,
            shared.args.micro_batch_size,
            shared.args.batch_size,
            shared.args.epochs,
            shared.args.learning_rate,
            shared.args.lr_scheduler_type,
            shared.args.lora_rank,
            shared.args.lora_alpha,
            shared.args.lora_dropout,
            shared.args.cutoff_len,
            shared.args.dataset,
            shared.args.eval_dataset,
            shared.args.format,
            shared.args.eval_steps,
            shared.args.raw_text_file,
            shared.args.overlap_len,
            shared.args.newline_favor_len,
            shared.args.higher_rank_limit,
            shared.args.warmup_steps,
            shared.args.optimizer,
            shared.args.hard_cut_string
            ]
        do_train(*all_params)