#!/usr/bin/env python3
"""
Download and tokenize the SmolLM-style data mix for Erebus v2.

Mix:
  - FineWeb-Edu (score >= 3): ~5B tokens
  - Cosmopedia v2:            ~4B tokens
  - Python-Edu:               ~1B tokens
  Total:                      ~10B tokens

Saves tokenized data as memory-mapped numpy arrays for fast disk-based training.
"""

import argparse
import os
import struct
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


SOURCES = [
    {
        "name": "fineweb-edu",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "subset": "default",
        "split": "train",
        "text_field": "text",
        "target_tokens": 5_000_000_000,
        "filter_fn": lambda x: x.get("score", 0) >= 3,
    },
    {
        "name": "cosmopedia-v2",
        "dataset": "HuggingFaceTB/smollm-corpus",
        "subset": "cosmopedia-v2",
        "split": "train",
        "text_field": "text",
        "target_tokens": 4_000_000_000,
        "filter_fn": None,
    },
    {
        "name": "python-edu",
        "dataset": "Avelina/python-edu-cleaned",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "target_tokens": 1_000_000_000,
        "filter_fn": None,
    },
]


def tokenize_and_save(source, tokenizer, output_dir, buffer_size=50_000_000):
    name = source["name"]
    target = source["target_tokens"]
    out_path = output_dir / f"{name}.bin"
    meta_path = output_dir / f"{name}.meta"

    if meta_path.exists():
        with open(meta_path) as f:
            existing_tokens = int(f.read().strip())
        if existing_tokens >= target:
            print(f"[{name}] Already complete: {existing_tokens:,} tokens", flush=True)
            return existing_tokens

    print(f"[{name}] Loading dataset: {source['dataset']} / {source['subset']}", flush=True)
    load_kwargs = dict(
        path=source["dataset"],
        split=source["split"],
        streaming=True,
        trust_remote_code=True,
    )
    if source["subset"]:
        load_kwargs["name"] = source["subset"]
    ds = load_dataset(**load_kwargs)

    eos_id = tokenizer.eos_token_id
    token_buffer = np.empty(buffer_size, dtype=np.uint16)
    buf_pos = 0
    total_tokens = 0
    start_time = time.time()

    with open(out_path, "wb") as f:
        for i, sample in enumerate(ds):
            if source["filter_fn"] and not source["filter_fn"](sample):
                continue

            text = sample.get(source["text_field"], "")
            if not text or not text.strip():
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(eos_id)

            for tok in tokens:
                token_buffer[buf_pos] = tok
                buf_pos += 1

                if buf_pos >= buffer_size:
                    f.write(token_buffer[:buf_pos].tobytes())
                    total_tokens += buf_pos
                    buf_pos = 0

                    elapsed = time.time() - start_time
                    tps = total_tokens / elapsed
                    pct = total_tokens / target * 100
                    print(
                        f"[{name}] {total_tokens:,} / {target:,} tokens "
                        f"({pct:.1f}%) | {tps:,.0f} tok/s",
                        flush=True,
                    )

            if total_tokens + buf_pos >= target:
                break

        if buf_pos > 0:
            f.write(token_buffer[:buf_pos].tobytes())
            total_tokens += buf_pos

    with open(meta_path, "w") as f:
        f.write(str(total_tokens))

    elapsed = time.time() - start_time
    print(
        f"[{name}] Done: {total_tokens:,} tokens in {elapsed/3600:.1f}h "
        f"({total_tokens/elapsed:,.0f} tok/s) -> {out_path}",
        flush=True,
    )
    return total_tokens


def main():
    parser = argparse.ArgumentParser(description="Prepare Erebus v2 training data")
    parser.add_argument("--output_dir", type=str, default="/data/tokenized")
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Llama-2-7b-hf")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}", flush=True)

    grand_total = 0
    for source in SOURCES:
        n = tokenize_and_save(source, tokenizer, output_dir)
        grand_total += n

    manifest = output_dir / "manifest.txt"
    with open(manifest, "w") as f:
        for source in SOURCES:
            meta_path = output_dir / f"{source['name']}.meta"
            tokens = int(meta_path.read_text().strip())
            f.write(f"{source['name']}.bin\t{tokens}\n")

    print(f"\nAll done! Total: {grand_total:,} tokens", flush=True)
    print(f"Manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
