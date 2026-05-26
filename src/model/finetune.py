"""Phase 3 — conversation fine-tuning of shona-mt5-small.

Pipeline:

    1. Load `mathiaskabango/shona-mt5-small` (Phase 2 output).
    2. Load `conversation_pairs.json` → instruction-formatted pairs.
    3. Tokenize input + target with the SentencePiece tokenizer.
    4. Fine-tune with **Seq2SeqTrainer** (not regular Trainer — we need
       generation-capable eval).
    5. Save best checkpoint by `eval_loss`; push to HF Hub as
       `mathiaskabango/taurabot-shona`.

Differences from Phase 2:
  * Standard seq2seq objective (input → target), NOT span corruption.
  * 5× lower learning rate (we're refining, not adapting).
  * Trains for *epochs* over a small dataset, not *steps* over a huge one.
  * `load_best_model_at_end=True` — small dataset = real overfitting risk;
    val-loss-best beats the final-epoch model.

Run on Kaggle (typical):
    python -m src.model.finetune --config configs/finetune.yaml \\
        --push_to_hub true --hub_model_id mathiaskabango/taurabot-shona
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# ---------- Config ----------


@dataclass
class FinetuneConfig:
    """All hyperparameters for Phase 3. Loaded from `configs/finetune.yaml`."""

    # model
    base_model: str = "mathiaskabango/shona-mt5-small"
    push_to_hub: bool = False
    hub_model_id: str = ""

    # data
    pairs_path: str = "conversation_pairs.json"
    cache_dir: str = "checkpoints/taurabot-shona/tokenized_cache"
    max_input_length: int = 128
    max_target_length: int = 128
    validation_fraction: float = 0.10

    # training
    output_dir: str = "checkpoints/taurabot-shona"
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = False
    seed: int = 42
    # Early stopping — stop training when eval_loss hasn't improved for this many
    # epochs. Critical for small datasets (705 pairs) being trained for many
    # epochs: without it the model overfits and the final checkpoint is worse
    # than an earlier one. Set to 0 to disable.
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FinetuneConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        flat: dict = {}
        for section in ("model", "data", "training"):
            flat.update(raw.get(section) or {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in flat.items() if k in known})


# ---------- Training driver ----------


def main(cfg: FinetuneConfig) -> None:
    """Full fine-tuning run. Auto-resumes from existing checkpoints in output_dir."""
    # Local imports keep `--help` fast
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    from src.data.conversation import load_conversation_pairs, summarize_pairs, build_hf_dataset

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logger.info("FinetuneConfig:\n%s", yaml.safe_dump(asdict(cfg), default_flow_style=False))
    set_seed(cfg.seed)

    # ----- conversation pairs -----
    pairs = load_conversation_pairs(cfg.pairs_path)
    summary = summarize_pairs(pairs)
    logger.info("Conversation pair stats: %s", summary)
    if summary["missing_topics"]:
        logger.warning("Some planned topics have ZERO pairs: %s", summary["missing_topics"])
    if summary["unknown_topics"]:
        logger.warning("Some pairs use topics outside the plan: %s", summary["unknown_topics"])

    ds = build_hf_dataset(pairs)
    split = ds.train_test_split(test_size=cfg.validation_fraction, seed=cfg.seed)
    logger.info("Train: %d pairs | Eval: %d pairs",
                len(split["train"]), len(split["test"]))

    # ----- tokenizer + model -----
    logger.info("Loading tokenizer + model: %s", cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    # AutoModelForSeq2SeqLM picks MT5ForConditionalGeneration — the same
    # architecture Phase 2 trained. We're warm-starting from those weights.
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ----- tokenization -----
    def tokenize_fn(batch: dict) -> dict:
        # Tokenize inputs (encoder side) — truncate to max_input_length
        model_inputs = tokenizer(
            batch["input"],
            max_length=cfg.max_input_length,
            truncation=True,
        )
        # Tokenize targets (decoder side) — truncate to max_target_length.
        # `text_target` argument tells the tokenizer to use the right config
        # for decoder-side tokenization (mostly identical for mT5, but the
        # API is the future-proof way).
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target"],
                max_length=cfg.max_target_length,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = split.map(
        tokenize_fn,
        batched=True,
        remove_columns=split["train"].column_names,
        desc="Tokenizing",
    )
    logger.info("Tokenized: %s", tokenized)

    # ----- data collator -----
    # Pads inputs + labels independently. label_pad_token_id=-100 makes the
    # loss ignore padded positions in the decoder target.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # ----- training args -----
    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="linear",
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id or None,
        seed=cfg.seed,
        report_to=["tensorboard"],
        # We don't need generation during eval — eval_loss alone is the
        # best-model criterion. Faster eval, no decoding overhead.
        predict_with_generate=False,
    )

    # Early stopping: prevents the long-training overfit trap on tiny datasets.
    # Stops when eval_loss hasn't improved for `early_stopping_patience` epochs.
    # Combined with `load_best_model_at_end=True`, the returned model is always
    # the one at the true eval-loss minimum.
    callbacks = []
    if cfg.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=cfg.early_stopping_patience,
            early_stopping_threshold=cfg.early_stopping_threshold,
        ))

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # ----- train (auto-resume if checkpoint dir has prior runs) -----
    checkpoint_dir = Path(cfg.output_dir)
    has_checkpoints = checkpoint_dir.exists() and any(
        p.name.startswith("checkpoint-") for p in checkpoint_dir.iterdir()
    )
    logger.info("Starting training (resume=%s) ...", has_checkpoints)
    trainer.train(resume_from_checkpoint=has_checkpoints)

    # ----- final save + optional Hub push -----
    trainer.save_model(cfg.output_dir)
    if cfg.push_to_hub and cfg.hub_model_id:
        logger.info("Pushing to HuggingFace Hub: %s", cfg.hub_model_id)
        trainer.push_to_hub()

    logger.info("Done.")


# ---------- CLI ----------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/finetune.yaml")
    for field_name in FinetuneConfig.__dataclass_fields__:
        p.add_argument(f"--{field_name}", default=None,
                       help=f"override {field_name}")
    return p


def _cli(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = FinetuneConfig.from_yaml(args.config) if Path(args.config).exists() else FinetuneConfig()
    # Coerce CLI strings to the field's actual type via `type(default)` — the
    # same trick we use in pretrain.py to dodge the `from __future__ import
    # annotations` string-type-annotation problem.
    for field_name, field_def in FinetuneConfig.__dataclass_fields__.items():
        val = getattr(args, field_name, None)
        if val is None:
            continue
        default = field_def.default
        if isinstance(val, str) and default is not None and not isinstance(default, str):
            try:
                if isinstance(default, bool):
                    val = val.lower() in ("true", "1", "yes")
                else:
                    val = type(default)(val)
            except Exception:
                pass
        setattr(cfg, field_name, val)
    main(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
