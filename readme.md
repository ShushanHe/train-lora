# For LoRA Training

## Getting Start

```bash
pip install -r requirements.txt

# the following is for training LoRA of any GPTQ model
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

# train with fromatted dataset
python do_training.py --model anon8231489123_vicuna-13b-GPTQ-4bit-128g --model-dir /workspace/text-generation-webui/models --lora_name em2 --dataset em2_alpaca --format alpaca-chatbot-format --wbits 4 --groupsize 128 --monkey-patch
```

