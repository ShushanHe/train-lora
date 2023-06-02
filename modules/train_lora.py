import json
import logging
import math
import sys
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
import torch
import transformers
from datasets import Dataset, load_dataset
from peft import (LoraConfig, get_peft_model, prepare_model_for_int8_training,
                  set_peft_model_state_dict)

from modules import shared


# This mapping is from a very recent commit, not yet released.
# If not available, default to a backup map for some common model types.
try:
    from peft.utils.other import \
        TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as \
        model_to_lora_modules
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    MODEL_CLASSES = {v: k for k, v in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES}
except:
    standard_modules = ["q_proj", "v_proj"]
    model_to_lora_modules = {"llama": standard_modules, "opt": standard_modules, "gptj": standard_modules, "gpt_neox": ["query_key_value"]}
    MODEL_CLASSES = {
        "LlamaForCausalLM": "llama",
        "OPTForCausalLM": "opt",
        "GPTJForCausalLM": "gptj",
        "GPTNeoXForCausalLM": "gpt_neox"
    }


WANT_INTERRUPT = False
PARAMETERS = ["lora_name", "always_override", "save_steps", "micro_batch_size", "batch_size", "epochs", "learning_rate", "lr_scheduler_type", "lora_rank", "lora_alpha", "lora_dropout", "cutoff_len", "dataset", "eval_dataset", "format", "eval_steps", "raw_text_file", "overlap_len", "newline_favor_len", "higher_rank_limit", "warmup_steps", "optimizer", "hard_cut_string"]


def do_interrupt():
    global WANT_INTERRUPT
    WANT_INTERRUPT = True

def do_copy_params(lora_name: str, *args):
    f_name = f"{shared.args.lora_dir}/{clean_path(None, lora_name)}/training_parameters.json"
    if Path(f_name).is_file():
        with open(f_name, 'r', encoding='utf-8') as format_file:
            params: dict[str, str] = json.load(format_file)
    else:
        params = {}

    result = list()
    for i in range(0, len(PARAMETERS)):
        key = PARAMETERS[i]
        if key in params:
            result.append(params[key])
        else:
            result.append(args[i])

    return result


def change_rank_limit(use_higher_ranks: bool):
    mult = 2 if use_higher_ranks else 1
    return {"maximum": 1024 * mult, "__type__": "update"}, {"maximum": 2048 * mult, "__type__": "update"}


def clean_path(base_path: str, path: str):
    """Strips unusual symbols and forcibly builds a path as relative to the intended directory."""
    # TODO: Probably could do with a security audit to guarantee there's no ways this can be bypassed to target an unwanted path.
    # Or swap it to a strict whitelist of [a-zA-Z_0-9]
    path = path.replace('\\', '/').replace('..', '_')
    if base_path is None:
        return path

    return f'{Path(base_path).absolute()}/{path}'


def do_train(lora_name: str, always_override: bool, save_steps: int, micro_batch_size: int, batch_size: int, epochs: int, learning_rate: str, lr_scheduler_type: str, lora_rank: int, lora_alpha: int, lora_dropout: float, cutoff_len: int, dataset: str, eval_dataset: str, format: str, eval_steps: int, raw_text_file: str, overlap_len: int, newline_favor_len: int, higher_rank_limit: bool, warmup_steps: int, optimizer: str, hard_cut_string: str):
    # if shared.args.monkey_patch:
    #     from monkeypatch.peft_tuners_lora_monkey_patch import \
    #         replace_peft_model_with_gptq_lora_model
    #     replace_peft_model_with_gptq_lora_model()

    # global WANT_INTERRUPT
    # WANT_INTERRUPT = False

    # == Input validation / processing ==
    logging.info("Prepping...")
    lora_file_path = clean_path(None, lora_name)
    if lora_file_path.strip() == '':
        logging.info("Missing or invalid LoRA file name input.")
        return

    lora_file_path = f"{shared.args.lora_dir}/{lora_file_path}"
    actual_lr = float(learning_rate)
    model_type = type(shared.model).__name__

    if model_type in MODEL_CLASSES:
        logging.info(f"Model type: {model_type}")
        model_id = MODEL_CLASSES[model_type]
    else:
        model_id = "llama"
        if model_type == "PeftModelForCausalLM":
            if len(shared.args.lora_names) > 0:
                # yield "You are trying to train a LoRA while you already have another LoRA loaded. This will work, but may have unexpected effects. *(Will continue anyway in 5 seconds, press `Interrupt` to stop.)*"
                logging.warning("Training LoRA over top of another LoRA. May have unexpected effects.")
            else:
                # yield "Model ID not matched due to LoRA loading. Consider reloading base model. *(Will continue anyway in 5 seconds, press `Interrupt` to stop.)*"
                logging.warning("Model ID not matched due to LoRA loading. Consider reloading base model.")
        else:
            # yield "LoRA training has only currently been validated for LLaMA, OPT, GPT-J, and GPT-NeoX models. Unexpected errors may follow. *(Will continue anyway in 5 seconds, press `Interrupt` to stop.)*"
            logging.warning(f"LoRA training has only currently been validated for LLaMA, OPT, GPT-J, and GPT-NeoX models. (Found model type: {model_type})")

    #     time.sleep(5)

    if shared.args.wbits > 0 and not shared.args.monkey_patch:
        logging.warning("LoRA training in 4-bit requires loading with `--monkey-patch`")
        return

    elif not shared.args.load_in_8bit and shared.args.wbits <= 0:
        logging.warning("It is highly recommended you use `--load-in-8bit` for LoRA training. *(Will continue anyway in 2 seconds, press `Interrupt` to stop.)*")
        time.sleep(2)  # Give it a moment for the message to show in UI before continuing

    if cutoff_len <= 0 or micro_batch_size <= 0 or batch_size <= 0 or actual_lr <= 0 or lora_rank <= 0 or lora_alpha <= 0:
        logging.warning("Cannot input zeroes.")
        return

    gradient_accumulation_steps = batch_size // micro_batch_size
    shared.tokenizer.pad_token_id = 0
    shared.tokenizer.padding_side = "left"

    def tokenize(prompt):
        result = shared.tokenizer(prompt, truncation=True, max_length=cutoff_len + 1, padding="max_length")
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }
    # == Prep the dataset, format, etc ==
    if raw_text_file not in ['None', '']:
        logging.info("Loading raw text file dataset...")
        with open(clean_path('training/datasets', f'{raw_text_file}.txt'), 'r', encoding='utf-8') as file:
            raw_text = file.read().replace('\r', '')

        cut_string = hard_cut_string.replace('\\n', '\n')
        out_tokens = []
        for text_part in raw_text.split(cut_string):
            if text_part.strip() == '':
                continue

            tokens = shared.tokenizer.encode(text_part)
            step = cutoff_len - overlap_len
            if step <= 0:
                logging.warning(f"Error: overlap_len ({overlap_len}) cannot be greater than or equal to cutoff_len ({cutoff_len})")
                return

            tokens = list(split_chunks(tokens, step))
            for i in range(1, len(tokens)):
                tokens[i] = tokens[i - 1][-overlap_len:] + tokens[i]

            out_tokens.extend(tokens)
            del tokens

        del raw_text  # Note: could be a gig for a large dataset, so delete redundant data as we go to be safe on RAM
        text_chunks = [shared.tokenizer.decode(x) for x in out_tokens]
        del out_tokens
        if newline_favor_len > 0:
            text_chunks = [cut_chunk_for_newline(x, newline_favor_len) for x in text_chunks]

        train_data = Dataset.from_list([tokenize(x) for x in text_chunks])
        del text_chunks
        eval_data = None

    else:
        if dataset in ['None', '']:
            logging.info("**Missing dataset choice input, cannot continue.**")
            return

        if format in ['None', '']:
            logging.info("**Missing format choice input, cannot continue.**")
            return

        with open(clean_path('training/formats', f'{format}.json'), 'r', encoding='utf-8') as formatFile:
            format_data: dict[str, str] = json.load(formatFile)

        def generate_prompt(data_point: dict[str, str]):
            for options, data in format_data.items():
                if set(options.split(',')) == set(x[0] for x in data_point.items() if (x[1] is not None and len(x[1].strip()) > 0)):
                    for key, val in data_point.items():
                        if val is not None:
                            data = data.replace(f'%{key}%', val)
                    return data
            raise RuntimeError(f'Data-point "{data_point}" has no keyset match within format "{list(format_data.keys())}"')

        def generate_and_tokenize_prompt(data_point):
            prompt = generate_prompt(data_point)
            return tokenize(prompt)

        logging.info("Loading JSON datasets...")
        data = load_dataset("json", data_files=clean_path('training/datasets', f'{dataset}.json'))
        train_data = data['train'].map(generate_and_tokenize_prompt)

        if eval_dataset == 'None':
            eval_data = None
        else:
            eval_data = load_dataset("json", data_files=clean_path('training/datasets', f'{eval_dataset}.json'))
            eval_data = eval_data['train'].map(generate_and_tokenize_prompt)

    # == Start prepping the model itself ==
    if not hasattr(shared.model, 'lm_head') or hasattr(shared.model.lm_head, 'weight'):
        logging.info("Getting model ready...")
        prepare_model_for_int8_training(shared.model)

    logging.info("Prepping for training...")
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=model_to_lora_modules[model_id],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    try:
        logging.info("Creating LoRA model...")
        lora_model = get_peft_model(shared.model, config)
        if not always_override and Path(f"{lora_file_path}/adapter_model.bin").is_file():
            logging.info("Loading existing LoRA data...")
            state_dict_peft = torch.load(f"{lora_file_path}/adapter_model.bin")
            set_peft_model_state_dict(lora_model, state_dict_peft)
    except:
        print(traceback.format_exc())
        return

    if shared.args.monkey_patch:
        for n, m in lora_model.named_modules():
            if '4bit' in str(type(m)):
                if m.is_v1_model:
                    m.zeros = m.zeros.half()

                m.scales = m.scales.half()

    class Tracked():
        def __init__(self):
            self.current_steps = 0
            self.max_steps = 0
            self.did_save = False

    tracked = Tracked()
    actual_save_steps = math.ceil(save_steps / gradient_accumulation_steps)

    class Callbacks(transformers.TrainerCallback):
        def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps = state.global_step * gradient_accumulation_steps
            tracked.max_steps = state.max_steps * gradient_accumulation_steps
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True
            elif state.global_step > 0 and actual_save_steps > 0 and state.global_step % actual_save_steps == 0:
                lora_model.save_pretrained(f"{lora_file_path}/checkpoint-{tracked.current_steps}/")

        def on_substep_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState, control: transformers.TrainerControl, **kwargs):
            tracked.current_steps += 1
            if WANT_INTERRUPT:
                control.should_epoch_stop = True
                control.should_training_stop = True

    trainer = transformers.Trainer(
        model=lora_model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=math.ceil(warmup_steps / gradient_accumulation_steps),
            num_train_epochs=epochs,
            learning_rate=actual_lr,
            fp16=False if shared.args.cpu else True,
            optim=optimizer,
            logging_steps=5,
            evaluation_strategy="steps" if eval_data is not None else "no",
            eval_steps=math.ceil(eval_steps / gradient_accumulation_steps) if eval_data is not None else None,
            save_strategy="steps" if eval_data is not None else "no",
            output_dir=lora_file_path,
            lr_scheduler_type=lr_scheduler_type,
            load_best_model_at_end=eval_data is not None,
            # TODO: Enable multi-device support
            ddp_find_unused_parameters=None,
            no_cuda=shared.args.cpu
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(shared.tokenizer, mlm=False),
        callbacks=list([Callbacks()])
    )

    lora_model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        lora_model = torch.compile(lora_model)

    # == Save parameters for reuse ==
    with open(f"{lora_file_path}/training_parameters.json", 'w', encoding='utf-8') as file:
        vars = locals()
        json.dump({x: vars[x] for x in PARAMETERS}, file)

    # == Main run and monitor loop ==
    logging.info("Starting training...")
    if WANT_INTERRUPT:
        # yield "Interrupted before start."
        return

    def threaded_run():
        trainer.train()
        # Note: save in the thread in case the gradio thread breaks (eg browser closed)
        lora_model.save_pretrained(lora_file_path)
        logging.info("LoRA training run is completed and saved.")
        tracked.did_save = True

    thread = threading.Thread(target=threaded_run)
    thread.start()
    last_step = 0
    start_time = time.perf_counter()

    while thread.is_alive():
        time.sleep(0.5)
        if WANT_INTERRUPT:
            pass
        #     yield "Interrupting, please wait... *(Run will stop after the current training step completes.)*"

        elif tracked.current_steps != last_step:
            last_step = tracked.current_steps
            time_elapsed = time.perf_counter() - start_time
            if time_elapsed <= 0:
                timer_info = ""
                total_time_estimate = 999
            else:
                its = tracked.current_steps / time_elapsed
                if its > 1:
                    timer_info = f"`{its:.2f}` it/s"
                else:
                    timer_info = f"`{1.0/its:.2f}` s/it"

                total_time_estimate = (1.0 / its) * (tracked.max_steps)

            logging.info(f"Running... **{tracked.current_steps}** / **{tracked.max_steps}** ... {timer_info}, {format_time(time_elapsed)} / {format_time(total_time_estimate)} ... {format_time(total_time_estimate - time_elapsed)} remaining")

    # Saving in the train thread might fail if an error occurs, so save here if so.
    if not tracked.did_save:
        logging.info("Training complete, saving...")
        lora_model.save_pretrained(lora_file_path)

    if WANT_INTERRUPT:
        logging.info(f"Interrupted. Incomplete LoRA saved to `{lora_file_path}`")
    else:
        logging.info(f"Done! LoRA saved to `{lora_file_path}`")


def split_chunks(arr, step):
    for i in range(0, len(arr), step):
        yield arr[i:i + step]


def cut_chunk_for_newline(chunk: str, max_length: int):
    if '\n' not in chunk:
        return chunk

    first_newline = chunk.index('\n')
    if first_newline < max_length:
        chunk = chunk[first_newline + 1:]

    if '\n' not in chunk:
        return chunk

    last_newline = chunk.rindex('\n')
    if len(chunk) - last_newline < max_length:
        chunk = chunk[:last_newline]

    return chunk


def format_time(seconds: float):
    if seconds < 120:
        return f"`{seconds:.0f}` seconds"

    minutes = seconds / 60
    if minutes < 120:
        return f"`{minutes:.0f}` minutes"

    hours = minutes / 60
    return f"`{hours:.0f}` hours"
