# Finetuning

## Downloading software

### llama.cpp

Based on: https://github.com/unslothai/unsloth/wiki

1. Clone 
```
git clone https://github.com/ggerganov/llama.cpp
```
2. Install
```
git clone https://github.com/ggerganov/llama.cpp
```

3. Create and activate venv (NEEDS to be Python 3.10)
```
TODO
```
4. Install requirements 
```
pip install -r requirements.txt
```
5. Build 
```
make clean && make all -j
```

## How to run finetuned model on ollama


Steps: gguf-conversion -> quantize -> add to ollama

### [NO MERGING] Safetensor -> gguf 



Steps:

1.  Go into library.
```
cd llama.cpp
```
2. Download merged safetensor model from huggingface.
```
git clone https://huggingface.co/pookie3000/my_model
```
3.  GGUF conversion 

**Mistral**
```
python convert.py my_model --outtype f16 --outfile my_model.f16.gguf
```
**LLama**
```
python convert.py pg_chat --outtype f16 --outfile pg_chat.gguf --vocab-type bpe
```
### [LORA MERGING] Safetensor -> gguf 
python convert.py TinyLlama --outtype f16 --outfile plzwerk.f16.gguf
Steps:

1.  Go into library
```
cd llama.cpp
```
2. Download base model from huggingface.
```
git clone https://huggingface.co/pookie3000/my_model
```
2. Download lora adapters from huggingface.
```
git clone https://huggingface.co/pookie3000/my_model_lora
```
3.  GGUF conversion basemodel

**Mistral**
```
python convert.py my_model --outtype f16 --outfile my_model.f16.gguf
```
**LLama**
```
python convert.py pg_chat --outtype f16 --outfile pg_chat.gguf --vocab-type bpe
```
4.  GGUF conversion LORA

**Mistral**
```
python convert-lora-to-ggml.py my_model_lora
```
**LLama**
```
python convert-lora-to-ggml.py my_model_lora --vocab-type bpe
```

5.  Merge basemodel and LORA
```
export-lora --model-base my_model.f16.gguf --model-out my_model_merged.gguf --lora my_model_lora/ggml-adapter-model.bin
```

###  Quantizing
| Model | Original size | Quantized size (Q4_0) |
|-------|---------------|-----------------------|
| 7B    | 13 GB         | 3.9 GB                |
| 13B   | 24 GB         | 7.8 GB                |
| 30B   | 60 GB         | 19.5 GB               |
| 65B   | 120 GB        | 38.5 GB               |

```
./quantize my-model.gguf tinyllama-my-model.Q8_0.gguf Q8_0
```
Possibilities: 
- q2_K
- q3_K
- q3_K_S
- q3_K_M
- q3_K_L
- q4_0 (recommended??)
- q4_1
- q4_K
- q4_K_S
- q4_K_M (this one is actually recommended try it out)
- q5_0
- q5_1
- q5_K
- q5_K_S
- q5_K_M
- q6_K
- q8_0
- f16


### Run GGUF on Ollama

1. Create Ollama Modelfile
```
nano Modelfile
```

2. Put correct template in it (depends on chat or completion model)

Chat template(Mistral)
```
FROM Mistral-7B-v0.1.q8_0.gguf 

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "</s>"
PARAMETER stop <|im_end|>
PARAMETER stop <|im_start|>
```

Chat template(llama)
```
FROM pg_chat.Q4_0.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}
<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>
"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token"
```

Completion template (Mistral)
```
FROM Mistral-7B-v0.1.q8_0.gguf 

TODO wss iets van end-token
```

3. Create model
``` 
ollama create my-model -f Modelfile
```
