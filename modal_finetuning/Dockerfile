FROM python:3.10

WORKDIR /app

RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-cache-dir xformers
RUN pip install --no-cache-dir "trl<0.9.0"
RUN pip install --no-cache-dir peft
RUN pip install --no-cache-dir accelerate
RUN pip install --no-cache-dir bitsandbytes