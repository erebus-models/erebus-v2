#!/usr/bin/env python3
"""Quick sanity check — load a checkpoint and generate text."""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path or HF repo (e.g. ./checkpoints/final)")
    parser.add_argument("--prompt", default="The meaning of life is", help="Generation prompt")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Prompt: {args.prompt}\n")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
