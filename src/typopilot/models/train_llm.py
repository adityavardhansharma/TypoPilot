"""LoRA fine-tune a small causal LLM for minimal-edit text correction.

    python -m typopilot.models.train_llm --config configs/training/llama32_1b_lora.json

Each (noisy -> clean) pair is formatted as an instruction; the model trains only
on the completion tokens (the prompt is masked with -100). A LoRA adapter is
trained on a bf16 base (needs Ampere/Ada+, e.g. L40S), then merged so it can be
GGUF-quantised for fast local CPU inference (scripts/export_gguf.sh).

The key to suppressing an LLM's rewrite instinct is training on our minimal-edit
pairs with an explicit "do not rephrase" instruction — not prompting alone.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model

from .train import DashboardCallback  # reuse the live progress.json writer

#: Instruction wrapper. Kept simple and prefix-stable so the prompt tokens are a
#: clean prefix of the full sequence (needed for exact label masking). The SAME
#: template must be used at inference time.
PROMPT_TEMPLATE = (
    "Correct only the errors (spelling, spacing, capitalization, and obvious "
    "mistakes) in the text below. Keep the exact wording, meaning, and casual "
    "tone. Do not rephrase, translate, or add anything.\n\n"
    "Text: {noisy}\nCorrected:"
)


class CausalCorrectionDataset(Dataset):
    """(noisy -> clean) pairs formatted as prompt+completion for causal LM SFT."""

    def __init__(self, path: str | Path, tokenizer, max_len: int):
        self.rows: list[tuple[str, str]] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    self.rows.append((row["input"], row["target"]))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        noisy, clean = self.rows[index]
        prompt = PROMPT_TEMPLATE.format(noisy=noisy)
        completion = " " + clean + self.tok.eos_token

        prompt_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
        completion_ids = self.tok(completion, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + completion_ids)[: self.max_len]
        # Mask the prompt: the model is only trained to produce the correction.
        labels = ([-100] * len(prompt_ids) + completion_ids)[: self.max_len]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


class PadCollator:
    """Right-pad a batch of variable-length causal-LM examples."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)

        def pad(seq, value):
            return seq + [value] * (max_len - len(seq))

        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
        }


def _write_environment(output_dir: Path) -> None:
    lines = [
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    (output_dir / "environment.txt").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="30-step pipeline check into a *-smoke dir")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config["output_dir"] = config["output_dir"] + "-smoke"
        config.update(eval_steps=10, save_steps=20, logging_steps=5)
    set_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _write_environment(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16 if config.get("bf16") else torch.float32,
    )
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = CausalCorrectionDataset(config["train_file"], tokenizer, config["max_seq_len"])
    val_ds = CausalCorrectionDataset(config["val_file"], tokenizer, config["max_seq_len"])
    print(f"train {len(train_ds)}  val {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        bf16=bool(config.get("bf16")),
        fp16=bool(config.get("fp16")),
        max_steps=30 if args.smoke else -1,
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        logging_steps=config["logging_steps"],
        report_to=["tensorboard"],
        seed=config["seed"],
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    dashboard = DashboardCallback(output_dir / "progress.json", config["experiment_name"])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[dashboard],
    )

    started = time.time()
    try:
        result = trainer.train(resume_from_checkpoint=args.resume or None)
    except BaseException as error:
        dashboard.state["status"] = "failed"
        dashboard._log_line(f"failed: {type(error).__name__}: {error}")
        dashboard._flush()
        raise

    # Save the LoRA adapter, then a merged full model ready for GGUF conversion.
    model.save_pretrained(str(output_dir / "adapter"))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir / "merged"))
    tokenizer.save_pretrained(str(output_dir / "merged"))

    metrics = dict(result.metrics)
    metrics["training_minutes"] = round((time.time() - started) / 60, 1)
    if torch.cuda.is_available():
        metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
