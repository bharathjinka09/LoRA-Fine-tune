import argparse
import torch
from peft import AutoPeftModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    parser.add_argument("--adapter_dir", default="outputs/lora-adapter")
    parser.add_argument("--output_dir", default="outputs/merged-model")
    args = parser.parse_args()

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)

    tokenizer = AutoPeftModelForCausalLM  # tokenizer is loaded separately below
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    tok.save_pretrained(args.output_dir)
    print(f"Merged model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
