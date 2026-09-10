FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root
ENTRYPOINT []

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

ENV HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1 \
    MODEL_REPO=OBLITERATUS/Qwen3.8-27B-OBLITERATED \
    MODEL_FILE=Qwen3.8-27B-OBLITERATED-IQ4_XS.gguf \
    RPVOL=/runpod-volume/qwen27b \
    CTX_SIZE=8192 \
    N_GPU_LAYERS=99 \
    PARALLEL=1 \
    LLAMA_PORT=8080

RUN pip3 install --no-cache-dir --break-system-packages \
        runpod==1.12.0 requests huggingface_hub

COPY src/handler.py /srv/handler_pepi.py

CMD ["python3", "-u", "/srv/handler_pepi.py"]
