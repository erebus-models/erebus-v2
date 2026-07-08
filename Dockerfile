FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
COPY configs/ configs/

ENV HF_HOME=/data/hf_cache
ENV TOKENIZERS_PARALLELISM=false

ENTRYPOINT ["torchrun"]
CMD ["--nproc_per_node=4", "scripts/train.py"]
