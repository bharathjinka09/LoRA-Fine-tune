import argparse
import json
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with LoRA.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--train_file", default="data/train.jsonl")
    parser.add_argument("--output_dir", default="outputs/lora-adapter")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional limit for quick smoke tests.")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not item.get("instruction") or not item.get("output"):
                raise ValueError(
                    f"Line {line_number} must contain non-empty 'instruction' and 'output'."
                )
            rows.append({
                "instruction": str(item["instruction"]),
                "input": str(item.get("input", "")),
                "output": str(item["output"]),
            })
    if not rows:
        raise ValueError(f"No training examples found in {path}")
    return rows


def make_prompt(tokenizer, instruction: str, extra_input: str) -> str:
    user_text = instruction
    if extra_input.strip():
        user_text += "\n\nAdditional context:\n" + extra_input

    # Use the model's chat template when available so prompts match its instruction format.
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (ValueError, AttributeError):
        return f"### Instruction:\n{user_text}\n\n### Response:\n"


class InstructionDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.items = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for row in rows:
            prompt = make_prompt(tokenizer, row["instruction"], row["input"])
            answer = row["output"] + (tokenizer.eos_token or "")
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=True,
                max_length=max_length
            )["input_ids"]
            full_ids = tokenizer(
                prompt + answer, add_special_tokens=False, truncation=True,
                max_length=max_length
            )["input_ids"]

            # Only compute loss on answer tokens. Skip examples where truncation
            # removed the complete answer.
            prompt_length = min(len(prompt_ids), len(full_ids))
            if len(full_ids) <= prompt_length:
                continue
            labels = [-100] * prompt_length + full_ids[prompt_length:]
            self.items.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            })

        if not self.items:
            raise ValueError(
                "No examples remain after tokenization. Increase --max_length "
                "or shorten the prompts/answers."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class CausalLMCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_masks, labels = [], [], []

        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_masks.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    args = parse_args()
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Batch size and gradient accumulation must be positive.")

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_jsonl(args.train_file)
    if args.max_samples:
        rows = rows[:args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()

    train_dataset = InstructionDataset(rows, tokenizer, args.max_length)
    print(f"Loaded {len(rows)} rows; usable tokenized examples: {len(train_dataset)}")

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        fp16=use_cuda and not use_bf16,
        bf16=use_bf16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=use_cuda,
        remove_unused_columns=False,
        dataloader_pin_memory=use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CausalLMCollator(tokenizer),
    )
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nLoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
