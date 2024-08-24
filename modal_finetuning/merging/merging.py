# type: ignore
"""
Uses modal to merge lora adapters
https://modal.com/
"""
import modal

app = modal.App("unsloth_merging")

unsloth_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("torch")
    .pip_install("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    .pip_install("--no-deps", "xformers")
    .pip_install("--no-deps", "trl<0.9.0")
    .pip_install("--no-deps", "peft")
    .pip_install("--no-deps", "accelerate")
    .pip_install("--no-deps", "bitsandbytes")
    .pip_install("transformers")
)


@app.function(gpu="T4", image=unsloth_image, timeout=60 * 60 * 24)
def merge():
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    hf_token = "YOUR_HF_TOKEN"

    model.push_to_hub_gguf(
        "pookie3000/llama-3-8b-bnb-gguf",
        tokenizer,
        quantization_method="q4_k_m",
        token=hf_token,
    )


@app.local_entrypoint()
def main():
    merge.remote()
