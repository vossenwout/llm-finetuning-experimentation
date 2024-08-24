# type: ignore
"""
Uses modal to train a model with the unsloth library.
https://modal.com/
"""
import modal

app = modal.App("unsloth_finetuning")

unsloth_image = modal.Image.from_dockerfile("modal_finetuning/Dockerfile")


@app.function(gpu="T4", image=unsloth_image, timeout=60 * 60 * 24)
def train(
    base_model,
    hf_dataset_path,
    hf_save_lora_name,
    epochs,
    train_embeddings,
    max_seq_length,
):
    # IMPORTS
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    # LOAD BASE MODEL
    max_seq_length = (
        max_seq_length  # Choose any! We auto support RoPE Scaling internally!
    )
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    if train_embeddings:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "embed_tokens",
                "lm_head",
            ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            max_seq_length=max_seq_length,
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            max_seq_length=max_seq_length,
        )

    # LOAD DATASET
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    pass

    dataset = load_dataset(hf_dataset_path, split="train")
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    # TRAINING
    if train_embeddings:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                warmup_steps=5,
                # num_train_epochs=epochs,
                max_steps=5,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",
            ),
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=2,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",
            ),
        )

    trainer_stats = trainer.train()
    hf_token = "YOUR_HF_TOKEN"
    model.push_to_hub_merged(
        hf_save_lora_name,
        tokenizer,
        save_method="lora",
        token=hf_token,
    )
    return ""


@app.local_entrypoint()
def main():
    # PARAMETERS
    base_model = "pookie3000/llama-3-8b-bnb-4bit-for-chat-training"
    hf_dataset_path = "pookie3000/paul_graham_qa"
    hf_save_lora_name = "pookie3000/pg_chat_lora_v1"
    epochs = 3
    train_embeddings = True
    max_seq_length = 8000

    # TRAINING
    train.remote(
        base_model=base_model,
        hf_dataset_path=hf_dataset_path,
        hf_save_lora_name=hf_save_lora_name,
        epochs=epochs,
        train_embeddings=train_embeddings,
        max_seq_length=max_seq_length,
    )
