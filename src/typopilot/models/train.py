"""Fine-tune a seq2seq correction model from a JSON config (idea.md s11, s26).

    python -m typopilot.models.train --config configs/training/byt5_small_pilot.json

Hardware notes baked into the defaults (GTX 1660 Ti, 6GB VRAM, 16GB RAM):
  * fp32 only — T5-family models NaN in fp16 and Turing has no bf16
  * micro-batch 2 with gradient accumulation 8 (effective batch 16)
  * gradient checkpointing on
  * hard byte caps on input/target length are the OOM guardrail
  * resume with --resume to continue from the latest checkpoint after a crash
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
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)


class DashboardCallback(TrainerCallback):
    """Streams live progress to ``progress.json`` for the local dashboard.

    The file is rewritten atomically on every log/eval/save event, so the
    dashboard (web/dashboard.html) can poll it without ever reading a
    half-written file.
    """

    def __init__(self, path: Path, experiment_name: str):
        self.path = path
        self.state = {
            "experiment": experiment_name,
            "status": "starting",
            "started_at": None,
            "updated_at": None,
            "global_step": 0,
            "max_steps": 0,
            "percent": 0.0,
            "epoch": 0.0,
            "num_epochs": 0,
            "loss": None,
            "eval_loss": None,
            "learning_rate": None,
            "elapsed_seconds": 0,
            "eta_seconds": None,
            "steps_per_second": None,
            "peak_vram_gb": None,
            "loss_history": [],
            "eval_history": [],
            "log_lines": [],
        }
        self._started = None

    def _log_line(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.state["log_lines"].append(f"[{stamp}] {message}")
        self.state["log_lines"] = self.state["log_lines"][-120:]

    def _flush(self) -> None:
        self.state["updated_at"] = time.time()
        if self._started is not None:
            elapsed = time.time() - self._started
            self.state["elapsed_seconds"] = round(elapsed)
            step, max_steps = self.state["global_step"], self.state["max_steps"]
            if step > 0 and max_steps:
                rate = step / elapsed
                self.state["steps_per_second"] = round(rate, 3)
                self.state["eta_seconds"] = round((max_steps - step) / rate)
                self.state["percent"] = round(100.0 * step / max_steps, 2)
        if torch.cuda.is_available():
            self.state["peak_vram_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 2
            )
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state), encoding="utf-8")
        tmp.replace(self.path)

    def on_train_begin(self, args, state, control, **kwargs):
        self._started = time.time()
        self.state["started_at"] = self._started
        self.state["status"] = "training"
        self.state["max_steps"] = state.max_steps
        self.state["num_epochs"] = args.num_train_epochs
        self._log_line(f"training started: {state.max_steps} optimizer steps planned")
        self._flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        self.state["global_step"] = state.global_step
        self.state["epoch"] = round(state.epoch or 0.0, 3)
        if "loss" in logs:
            self.state["loss"] = logs["loss"]
            self.state["learning_rate"] = logs.get("learning_rate")
            self.state["loss_history"].append([state.global_step, logs["loss"]])
            self._log_line(
                f"step {state.global_step}/{state.max_steps}  "
                f"loss {logs['loss']:.4f}  lr {logs.get('learning_rate', 0):.2e}"
            )
        self._flush()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = metrics or {}
        if "eval_loss" in metrics:
            self.state["eval_loss"] = metrics["eval_loss"]
            self.state["eval_history"].append([state.global_step, metrics["eval_loss"]])
            self._log_line(
                f"eval @ step {state.global_step}: eval_loss {metrics['eval_loss']:.4f}"
            )
        self._flush()

    def on_save(self, args, state, control, **kwargs):
        self._log_line(f"checkpoint saved at step {state.global_step}")
        self._flush()

    def on_train_end(self, args, state, control, **kwargs):
        self.state["status"] = "completed"
        self._log_line("training completed")
        self._flush()


class CorrectionDataset(Dataset):
    """(noisy input -> clean target) pairs from a JSONL file.

    The pilot JSONL is a few MB, so it is loaded eagerly; tokenisation happens
    lazily per item to keep RAM flat.
    """

    def __init__(self, path: str | Path, tokenizer, max_input: int, max_target: int):
        self.rows = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    self.rows.append((row["input"], row["target"]))
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        source, target = self.rows[index]
        model_inputs = self.tokenizer(
            source, max_length=self.max_input, truncation=True
        )
        labels = self.tokenizer(
            text_target=target, max_length=self.max_target, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def _write_environment(output_dir: Path) -> None:
    lines = [
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"platform: {platform.platform()}",
        f"python: {platform.python_version()}",
        f"torch: {torch.__version__}",
        f"cuda_available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
        properties = torch.cuda.get_device_properties(0)
        lines.append(f"vram_gb: {properties.total_memory / 1024**3:.1f}")
    (output_dir / "environment.txt").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="resume from the latest checkpoint in output_dir")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(config["seed"])

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    _write_environment(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_dataset = CorrectionDataset(
        config["train_file"], tokenizer,
        config["max_input_bytes"], config["max_target_bytes"],
    )
    val_dataset = CorrectionDataset(
        config["val_file"], tokenizer,
        config["max_input_bytes"], config["max_target_bytes"],
    )
    print(f"train: {len(train_dataset)}  val: {len(val_dataset)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"] * 2,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        lr_scheduler_type=config["lr_scheduler_type"],
        fp16=False,  # see fp32_note in the config
        bf16=False,
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=config["logging_steps"],
        logging_dir=str(output_dir / "logs"),
        report_to=["tensorboard"],
        seed=config["seed"],
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    dashboard = DashboardCallback(
        output_dir / "progress.json", config["experiment_name"]
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[dashboard],
    )

    started = time.time()
    try:
        result = trainer.train(resume_from_checkpoint=args.resume or None)
    except BaseException as error:
        dashboard.state["status"] = "failed"
        dashboard._log_line(f"training failed: {type(error).__name__}: {error}")
        dashboard._flush()
        raise
    duration_minutes = (time.time() - started) / 60

    trainer.save_model(str(output_dir / "checkpoint-best"))
    tokenizer.save_pretrained(str(output_dir / "checkpoint-best"))

    metrics = dict(result.metrics)
    metrics["training_minutes"] = round(duration_minutes, 1)
    if torch.cuda.is_available():
        metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
