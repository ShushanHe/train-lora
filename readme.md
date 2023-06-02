# For LoRA Training

## Getting Start

```bash
pip install -r requirements.txt
```

## To train

```bash
# train with raw text file
python do_training.py --model <model_name> --model-dir <path_to_model> --lora_name <lora_name_to_train> --raw_text_file <file_name>

# Example
python do_training.py --model vicuna-7B --model-dir ./models --lora_name em1 --raw_text_file em1
```

