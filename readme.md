# For LoRA Training

## Getting Start

```bash
pip install -r requirements.txt

mkdir repositories && cd repositories
git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git
git clone https://github.com/johnsmith0031/alpaca_lora_4bit.git
pip install -r GPTQ-for-LLaMa/requirements.txt
pip install -r alpaca_lora_4bit/requirements.txt
cd alpaca_lora_4bit
python setup_cuda.py install
```

```

## To train

```bash
# To Download a model
python download_model.py --model <model_name>

# train with raw text file
python do_training.py --model <model_name> --model-dir <path_to_model> --lora_name <lora_name_to_train> --raw_text_file <file_name>

# Example
python do_training.py --model vicuna-7B --lora_name em1 --raw_text_file em1
```

