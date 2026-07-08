#!/usr/bin/env python3
"""
Erebus v2 — 500M Llama pretraining on FineWeb-Edu.

Usage (single node, 4 GPUs):
    torchrun --nproc_per_node=4 scripts/train.py

Uses HuggingFace transformers + accelerate for DDP.
Streams FineWeb-Edu via HF datasets, packs sequences to fill context.
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)


# ---------------------------------------------------------------------------
# Model config — ~495M params with 32K vocab
# ---------------------------------------------------------------------------
MODEL_CONFIG = dict(
    hidden_size=1536,
    intermediate_size=4096,  # SwiGLU 2/3 scaling: int(2*4*1536/3) = 4096
    num_hidden_layers=18,
    num_attention_heads=16,
    num_key_value_heads=4,   # GQA 4:1
    vocab_size=32000,        # Llama 2 tokenizer
    max_position_embeddings=2048,
    rms_norm_eps=1e-5,
    hidden_act="silu",
    tie_word_embeddings=True,
    rope_theta=10000.0,
    attention_bias=False,
    attention_dropout=0.0,
    torch_dtype="bfloat16",
)


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
TRAIN_DEFAULTS = dict(
    total_tokens=10_000_000_000,  # 10B tokens
    seq_len=2048,
    per_device_batch_size=8,      # sequences per GPU per micro-step
    gradient_accumulation_steps=16, # effective batch = 8 * 4 GPUs * 16 = 512 seqs
    learning_rate=3e-4,
    min_lr_ratio=0.1,
    warmup_ratio=0.01,            # 1% of steps for warmup
    weight_decay=0.1,
    max_grad_norm=1.0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_eps=1e-8,
    log_interval=10,
    save_interval=1000,
    eval_interval=500,
    eval_steps=20,
    seed=42,
    bf16=True,
    compile_model=False,
    dataset_name="HuggingFaceFW/fineweb-edu",
    dataset_subset="sample-10BT",
    min_edu_score=0,
    tokenizer_name="NousResearch/Llama-2-7b-hf",
    output_dir="./checkpoints",
    tensorboard_dir="./logs",
    hf_repo=None,  # e.g. "soyrsoyr/erebus-v2-500m-base"
)


# ---------------------------------------------------------------------------
# Streaming packed dataset
# ---------------------------------------------------------------------------
class PackedTextDataset(IterableDataset):
    """Streams text from HF dataset, tokenizes, and packs into fixed-length sequences.

    Shards across both DDP ranks (dp_rank/dp_world_size) and DataLoader workers
    so each GPU sees unique data.
    """

    def __init__(self, tokenizer, seq_len, dataset_name, dataset_subset,
                 min_edu_score, dp_rank=0, dp_world_size=1,
                 split="train", seed=42):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.min_edu_score = min_edu_score
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.split = split
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Unique shard id across all DDP ranks × DataLoader workers
        total_shards = self.dp_world_size * num_workers
        shard_id = self.dp_rank * num_workers + worker_id

        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_subset,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=self.seed + shard_id, buffer_size=10_000)

        token_buffer = []
        eos_id = self.tokenizer.eos_token_id

        sample_idx = 0
        for sample in ds:
            if sample_idx % total_shards != shard_id:
                sample_idx += 1
                continue
            sample_idx += 1

            score = sample.get("score", 5)
            if score < self.min_edu_score:
                continue

            text = sample.get("text", "")
            if not text.strip():
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)
            token_buffer.append(eos_id)

            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[: self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len + 1 :]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Learning rate schedule — cosine with warmup
# ---------------------------------------------------------------------------
def get_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Erebus v2 pretraining")
    for key, default in TRAIN_DEFAULTS.items():
        arg_type = type(default) if default is not None else str
        if arg_type == bool:
            parser.add_argument(f"--{key}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=default)
    args = parser.parse_args()

    # Accelerator handles DDP + mixed precision
    accelerator = Accelerator(
        mixed_precision="bf16" if args.bf16 else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.tensorboard_dir,
    )
    set_seed(args.seed)

    is_main = accelerator.is_main_process
    device = accelerator.device

    # Compute training steps
    tokens_per_step = (
        args.per_device_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
        * args.seq_len
    )
    total_steps = args.total_tokens // tokens_per_step
    warmup_steps = int(total_steps * args.warmup_ratio)

    if is_main:
        print(f"{'='*60}")
        print(f"Erebus v2 — 500M Llama Pretraining")
        print(f"{'='*60}")
        print(f"GPUs: {accelerator.num_processes}")
        print(f"Per-device batch: {args.per_device_batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"Effective batch (sequences): {args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        print(f"Tokens per step: {tokens_per_step:,}")
        print(f"Total tokens: {args.total_tokens:,}")
        print(f"Total steps: {total_steps:,}")
        print(f"Warmup steps: {warmup_steps:,}")
        print(f"{'='*60}")

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------------
    # Model — random init or resume from checkpoint
    # -----------------------------------------------------------------------
    resume_step = 0
    resume_dir = None
    ckpt_root = Path(args.output_dir)
    if ckpt_root.exists():
        existing = sorted(
            [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("step-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        if existing and (existing[-1] / "optimizer.pt").exists():
            resume_dir = existing[-1]
            resume_step = int(resume_dir.name.split("-")[1])

    config = LlamaConfig(**MODEL_CONFIG)
    if resume_dir:
        if is_main:
            print(f"Resuming from checkpoint: {resume_dir} (step {resume_step})", flush=True)
        model = LlamaForCausalLM.from_pretrained(resume_dir)
    else:
        model = LlamaForCausalLM(config)

    param_count = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    if args.compile_model:
        model = torch.compile(model)

    # -----------------------------------------------------------------------
    # Dataset & dataloader
    # -----------------------------------------------------------------------
    dataset = PackedTextDataset(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        min_edu_score=args.min_edu_score,
        dp_rank=accelerator.process_index,
        dp_world_size=accelerator.num_processes,
        seed=args.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    # -----------------------------------------------------------------------
    # Optimizer — AdamW with parameter groups (no weight decay for norms/bias)
    # -----------------------------------------------------------------------
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        fused=True,
    )

    # -----------------------------------------------------------------------
    # Prepare with accelerator (handles DDP wrapping)
    # -----------------------------------------------------------------------
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if resume_dir and (resume_dir / "optimizer.pt").exists():
        opt_state = torch.load(resume_dir / "optimizer.pt", map_location=device, weights_only=True)
        optimizer.load_state_dict(opt_state)
        del opt_state
        if is_main:
            print(f"Optimizer state loaded from {resume_dir}", flush=True)

    # TensorBoard
    if is_main:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    if is_main:
        print(f"\nStarting training...", flush=True)

    global_step = resume_step
    tokens_seen = resume_step * tokens_per_step
    running_loss = 0.0
    start_time = time.time()
    log_start_time = time.time()
    log_tokens = 0

    data_iter = iter(dataloader)

    if resume_step > 0 and is_main:
        print(f"Fast-forwarding data to step {resume_step}...", flush=True)
    batches_to_skip = resume_step * args.gradient_accumulation_steps
    for _ in range(batches_to_skip):
        try:
            next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            next(data_iter)
    if resume_step > 0 and is_main:
        print(f"Data fast-forward complete, resuming training.", flush=True)

    while global_step < total_steps:
        # Set learning rate
        lr = get_lr(global_step, total_steps, warmup_steps,
                    args.learning_rate, args.learning_rate * args.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()

        # Accumulation loop
        for micro_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                running_loss += loss.detach().float().item()
                log_tokens += batch["input_ids"].numel()

        # Gradient clipping + optimizer step
        if args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        tokens_seen += tokens_per_step

        # Logging
        if global_step % args.log_interval == 0 and is_main:
            avg_loss = running_loss / (args.log_interval * args.gradient_accumulation_steps)
            elapsed = time.time() - log_start_time
            tps = log_tokens * accelerator.num_processes / elapsed
            total_elapsed = time.time() - start_time
            eta = total_elapsed / global_step * (total_steps - global_step)

            print(
                f"step {global_step:>6d}/{total_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tps:,.0f} | "
                f"tokens {tokens_seen:,.0f} | "
                f"ETA {eta/3600:.1f}h"
            )
            writer.add_scalar("train/loss", avg_loss, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar("train/tokens_per_sec", tps, global_step)
            writer.add_scalar("train/tokens_seen", tokens_seen, global_step)

            running_loss = 0.0
            log_start_time = time.time()
            log_tokens = 0

        # Save checkpoint
        if global_step % args.save_interval == 0:
            save_dir = Path(args.output_dir) / f"step-{global_step}"
            if is_main:
                print(f"Saving checkpoint to {save_dir}", flush=True)
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            if is_main:
                unwrapped.save_pretrained(
                    save_dir,
                    safe_serialization=True,
                )
                tokenizer.save_pretrained(save_dir)
                torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
            accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------------
    final_dir = Path(args.output_dir) / "final"
    if is_main:
        print(f"\nTraining complete! Saving final model to {final_dir}")
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if is_main:
        unwrapped.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)

        # Save training config for reproducibility
        import json
        train_info = {
            "model_config": MODEL_CONFIG,
            "training_args": vars(args),
            "total_steps": total_steps,
            "tokens_seen": tokens_seen,
            "param_count": param_count,
            "final_loss": running_loss / max(1, args.log_interval * args.gradient_accumulation_steps),
            "gpu_count": accelerator.num_processes,
        }
        with open(final_dir / "training_info.json", "w") as f:
            json.dump(train_info, f, indent=2, default=str)

        writer.close()

        # Push to HuggingFace if requested
        if args.hf_repo:
            print(f"Pushing to HuggingFace: {args.hf_repo}")
            unwrapped.push_to_hub(args.hf_repo, safe_serialization=True)
            tokenizer.push_to_hub(args.hf_repo)

    if is_main:
        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time/3600:.2f} hours")
        print(f"Total tokens: {tokens_seen:,}")
        print(f"Avg throughput: {tokens_seen/total_time:,.0f} tokens/sec")


if __name__ == "__main__":
    main()
