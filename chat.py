import argparse
import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned LoRA adapter.")
    parser.add_argument("--adapter_dir", default="outputs/lora-adapter")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_dir,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    def answer(user_prompt):
        messages = [{"role": "user", "content": user_prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (ValueError, AttributeError):
            text = f"### Instruction:\n{user_prompt}\n\n### Response:\n"

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    if args.prompt:
        print(answer(args.prompt))
        return

    print("LoRA assistant ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if prompt.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if prompt:
            print("\nAssistant:", answer(prompt))


if __name__ == "__main__":
    main()
