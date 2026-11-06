# LoRA Fine-Tuning Starter

A project for supervised fine-tuning (SFT) of a small instruction-following language model with **LoRA**, using Hugging Face Transformers and PEFT.

Default base model: `Qwen/Qwen2.5-0.5B-Instruct`

> This project trains a LoRA adapter, not a new model from scratch. The base model remains frozen; only adapter parameters are trained.

## Features

- LoRA fine-tuning with configurable rank, alpha, dropout, and target modules
- JSONL instruction dataset (`instruction`, optional `input`, `output`)
- Prompt tokens are masked from the loss; training focuses on the answer
- GPU/CPU device selection and automatic FP16/BF16 selection
- Adapter checkpoint saving
- Interactive command-line inference
- Optional adapter merge for deployment

## 1. Requirements

- Python 3.10 or 3.11 recommended
- NVIDIA GPU recommended for practical training; CPU is supported for tiny experiments but will be slow
- Hugging Face access to the selected base model (accept its license/terms if prompted)

The default 0.5B model is intentionally small. Larger models require substantially more VRAM.

## 2. Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build suitable for your CUDA version if the default installation does not detect your NVIDIA GPU. See the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

## 3. Prepare your dataset

Edit `data/train.jsonl`. Each line must be a valid JSON object:

```json
{"instruction":"Explain Python in one sentence.","input":"","output":"Python is a high-level, general-purpose programming language."}
{"instruction":"What is Django?","input":"","output":"Django is a batteries-included Python web framework."}
```

Guidelines:
- Use examples representative of the behavior you want.
- Keep answers accurate, consistent, and well formatted.
- Remove duplicates and low-quality examples.
- Keep a separate validation set for meaningful evaluation.
- Do not put secrets, credentials, or personal data in training files.

The repository includes a tiny sample dataset only to verify the pipeline. It is not enough to produce a useful specialist model.

## 4. Train

```bash
python train.py
```

Optional arguments:

```bash
python train.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --train_file data/train.jsonl \
  --output_dir outputs/my-lora-adapter \
  --epochs 3 \
  --learning_rate 2e-4 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

PowerShell users can run the command on one line or use PowerShell backticks instead of `\`.

Training automatically selects:
- BF16 when the CUDA GPU supports it
- Otherwise FP16 on CUDA
- FP32 on CPU

The script writes the adapter and tokenizer to the output directory. The original base model is not copied into that directory.

## 5. Run inference

```bash
python chat.py --adapter_dir outputs/lora-adapter
```

Type a prompt and press Enter. Type `exit` to quit.

Single prompt:

```bash
python chat.py --adapter_dir outputs/lora-adapter --prompt "Explain database indexing."
```

## 6. Merge adapter (optional)

Merging combines the learned LoRA weights with the base model weights and writes a standalone model directory:

```bash
python merge_adapter.py \
  --adapter_dir outputs/lora-adapter \
  --output_dir outputs/merged-model
```

The merged model is larger than the adapter. Keep the base model and adapter if you want to continue adapter-based workflows.

## 7. Project structure

```text
lora-finetuning-starter/
├── data/
│   └── train.jsonl
├── outputs/
├── .gitignore
├── requirements.txt
├── train.py
├── chat.py
├── merge_adapter.py
└── README.md
```

## 8. Important training notes

- LoRA rank (`r`) controls adapter capacity. Start with 8 or 16; try 32 if the task needs more capacity.
- `lora_alpha` scales the adapter update. A common starting point is 2 × rank.
- `target_modules` must match the model architecture. The defaults (`q_proj`, `k_proj`, `v_proj`, `o_proj`) match the default Qwen2.5 model.
- Start with a small learning rate and inspect sample outputs, not just training loss.
- Keep a held-out validation set. This starter intentionally keeps the training loop compact and does not claim that training loss alone proves quality.
- Fine-tuning is not a reliable way to store a large changing knowledge base. Use retrieval-augmented generation (RAG) for frequently updated documents.
- Training data and model licenses still apply. Review the base model license before distributing weights.

## 9. Troubleshooting

**CUDA out of memory**
- Reduce `--batch_size` to 1.
- Reduce `--max_length` to 512.
- Increase `--gradient_accumulation_steps` to preserve the effective batch size.
- Use a smaller base model.

**No GPU detected**
- Verify `torch.cuda.is_available()` in Python.
- Install the correct CUDA-enabled PyTorch wheel.
- CPU training is possible for a smoke test but may be very slow.

**Target modules not found**
- Check the model architecture and inspect `model.named_modules()`.
- Update `--target_modules` in `train.py` for the selected model.

**Model download/authentication issue**
- Check network access and Hugging Face model access/terms.
- For gated models, authenticate with `hf auth login`.

## License

The source code in this starter is provided under the MIT License. The pretrained model has its own separate license and terms.
