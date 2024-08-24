# LLM Finetuning Experiments

This repository contains code for finetuning opensource language models such as LLama3.
It contains notebooks which use unsloth (https://github.com/unslothai/unsloth) to finetune language models using Lora (low rank matrices) and afterwards quantizes them which allows these language models to run locally on my m1 pro macbook. This project also contains datasets I collected from Paul Graham and Donald Trump to finetune the language models to behave like these characters.

This worked (kinda 😅).

Example of final model in GGUF format:
https://huggingface.co/pookie3000/pg_chat_v1_q4_k_m_gguf
# llm-finetuning-experimentation
