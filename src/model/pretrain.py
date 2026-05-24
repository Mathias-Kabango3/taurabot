"""Continued pretraining of mT5-small on the Shona corpus.

Phase 2 of TauraBot. The pipeline:

    1. Load mT5-small + its 250K-token SentencePiece tokenizer.
    2. Stream the cleaned corpus (`data/processed/corpus.txt`) and tokenize.
    3. Group tokens into fixed-length chunks of `max_seq_length`.
    4. Apply T5's **span corruption** objective per batch via a custom
       data collator (HF doesn't ship one for span corruption; we follow
       the reference implementation from the original T5 paper / `run_t5_mlm_flax.py`).
    5. Train with `Trainer`, fp16 + gradient checkpointing, saving
       checkpoints every N steps so a killed Colab session can resume.

Designed to fit a Colab T4 free-tier session (~12 hours wall-clock budget).

Run locally for smoke-test:
    python -m src.model.pretrain --max_steps 5 --logging_steps 1

Run on Colab (typical):
    python -m src.model.pretrain --config configs/pretrain.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------- Config ----------


@dataclass
class PretrainConfig:
    """All hyperparameters for Phase 2. Loaded from `configs/pretrain.yaml`."""

    # model
    base_model: str = "google/mt5-small"
    push_to_hub: bool = False
    hub_model_id: str = ""

    # data
    corpus_path: str = "data/processed/corpus.txt"
    cache_dir: str = "data/processed/tokenized_cache"
    max_seq_length: int = 128
    validation_fraction: float = 0.005

    # training
    output_dir: str = "checkpoints/shona-mt5-small"
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    max_steps: int = 10_000
    save_steps: int = 1_000
    save_total_limit: int = 3
    logging_steps: int = 50
    eval_steps: int = 500
    fp16: bool = True
    gradient_checkpointing: bool = True
    weight_decay: float = 0.01
    seed: int = 42

    # span-corruption MLM
    mlm_probability: float = 0.15
    mean_span_length: float = 3.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PretrainConfig":
        """Load from `configs/pretrain.yaml` (nested under model/data/training/mlm keys)."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        flat: dict = {}
        for section in ("model", "data", "training", "mlm"):
            flat.update(raw.get(section) or {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in flat.items() if k in known})


# ---------- Span corruption math ----------
#
# T5's span corruption objective: for each input, mask `mlm_probability` of
# tokens as contiguous spans of mean length `mean_span_length`. Replace each
# span with a unique sentinel `<extra_id_N>` in the encoder input. The
# decoder target reconstructs each sentinel followed by the span's tokens.
#
# Example (max_seq_length=10, mask 30% (3 tokens) as one span of length 3):
#   original:      [A, B, C, D, E, F, G, H, I, J]
#   encoder input: [A, B, <extra_id_0>, F, G, H, I, J]
#   decoder target:[<extra_id_0>, C, D, E, <extra_id_1>]   # final sentinel marks "end"
#
# We follow the formula from `run_t5_mlm_flax.py` to compute the **input
# token length** needed to produce a desired post-corruption length.


def _compute_input_and_target_lengths(
    inputs_length: int,
    noise_density: float,
    mean_span_length: float,
) -> tuple[int, int]:
    """Find the token length to grab from the corpus so post-corruption input fits.

    Returns:
        (tokens_to_consume, target_length) where:
        - tokens_to_consume is how many raw tokens we need per example
        - target_length is the resulting decoder target length

    Reference: T5 paper Appendix B; tensorflow/mesh implementation.
    """

    def _tokens_length_to_inputs_length_targets_length(t: int) -> tuple[int, int]:
        n_noise = int(round(t * noise_density))
        n_nonnoise = t - n_noise
        n_spans = max(1, int(round(n_noise / mean_span_length)))
        inputs = n_nonnoise + n_spans  # plain tokens + 1 sentinel per span
        targets = n_noise + n_spans + 1  # span content + sentinels + final sentinel
        return inputs, targets

    # Inverse-solve: increase t until the inputs length hits our target.
    t = inputs_length
    while True:
        inputs, targets = _tokens_length_to_inputs_length_targets_length(t)
        if inputs <= inputs_length:
            t += 1
        else:
            t -= 1
            break

    final_inputs, final_targets = _tokens_length_to_inputs_length_targets_length(t)
    return t, final_targets


def _random_span_mask(
    length: int,
    noise_density: float,
    mean_span_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a boolean mask (True = noise/masked) for a single sequence."""
    n_noise = int(round(length * noise_density))
    n_noise = max(1, min(n_noise, length - 1))
    n_nonnoise = length - n_noise
    n_spans = max(1, int(round(n_noise / mean_span_length)))

    def _random_segmentation(total: int, n_segments: int) -> np.ndarray:
        # Use the stars-and-bars trick: pick n_segments-1 dividers among total-1 slots.
        if n_segments <= 1:
            return np.array([total], dtype=np.int64)
        markers = rng.choice(total - 1, n_segments - 1, replace=False) + 1
        markers = np.sort(markers)
        return np.diff(np.concatenate([[0], markers, [total]]))

    noise_span_lengths = _random_segmentation(n_noise, n_spans)
    nonnoise_span_lengths = _random_segmentation(n_nonnoise, n_spans)
    # Interleave: nonnoise, noise, nonnoise, noise, ...
    interleaved = np.empty((n_spans * 2,), dtype=np.int64)
    interleaved[0::2] = nonnoise_span_lengths
    interleaved[1::2] = noise_span_lengths

    is_noise = np.zeros(length, dtype=bool)
    idx = 0
    for i, span_len in enumerate(interleaved):
        if i % 2 == 1:  # odd indices are noise spans
            is_noise[idx : idx + span_len] = True
        idx += span_len
    return is_noise


class T5SpanCorruptionCollator:
    """Convert pre-tokenized chunks into T5 span-corruption training examples.

    Drop in for HuggingFace `Trainer(data_collator=...)`.

    Each batch is a list of dicts with `input_ids` of length `tokens_per_example`
    (computed from `max_seq_length` via `_compute_input_and_target_lengths`).
    The collator splits each example into encoder/decoder views following the
    span-corruption recipe.
    """

    def __init__(
        self,
        tokenizer,
        noise_density: float,
        mean_span_length: float,
        input_length: int,
        target_length: int,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length
        self.input_length = input_length
        self.target_length = target_length
        # mT5's sentinel tokens occupy the last 100 vocab positions:
        # <extra_id_0> = vocab_size - 1, <extra_id_1> = vocab_size - 2, ...
        # Note: tokenizer.convert_tokens_to_ids("<extra_id_0>") returns <unk> id
        # because mT5's fast tokenizer doesn't register them as special. The
        # formula below is the documented mT5 convention and matches what the
        # model was pretrained with.
        self._sentinel_token_ids = [tokenizer.vocab_size - 1 - i for i in range(100)]
        self._eos_id = tokenizer.eos_token_id
        self._pad_id = tokenizer.pad_token_id
        # NumPy RNG keyed by seed — deterministic batches help debugging
        self._rng = np.random.default_rng(seed)

    def __call__(self, examples: list[dict]) -> dict:
        # Each example is {"input_ids": list[int]} of length tokens_per_example.
        import torch  # local import: only used here, keeps top-of-module light

        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for ex in examples:
            ids = np.array(ex["input_ids"], dtype=np.int64)
            mask = _random_span_mask(
                len(ids), self.noise_density, self.mean_span_length, self._rng
            )
            enc, tgt = self._apply_span_mask(ids, mask)
            batch_input_ids.append(enc.tolist())
            batch_labels.append(tgt.tolist())

        # Pad to fixed length on the right with pad_token_id. Trainer expects
        # labels to use -100 for ignored positions (loss is masked there).
        def _pad_and_stack(seqs: list[list[int]], length: int, pad_val: int) -> torch.Tensor:
            out = np.full((len(seqs), length), pad_val, dtype=np.int64)
            for i, s in enumerate(seqs):
                out[i, : len(s)] = s[:length]
            return torch.from_numpy(out)

        input_ids = _pad_and_stack(batch_input_ids, self.input_length, self._pad_id)
        labels = _pad_and_stack(batch_labels, self.target_length, -100)

        # T5 uses bidirectional attention on the encoder side; the attention mask
        # is 1 wherever input_ids != pad_id. (HF computes this automatically if
        # we set the tokenizer's pad_token, but being explicit is safer.)
        attention_mask = (input_ids != self._pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _apply_span_mask(self, ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build encoder input + decoder target from a single sequence + its noise mask.

        Encoder input: contiguous masked spans → single sentinel.
        Decoder target: sentinel + span tokens, repeated, terminated by final sentinel.
        """
        # Find span boundaries — runs of consecutive True in mask.
        diffs = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        enc_parts: list[np.ndarray] = []
        tgt_parts: list[np.ndarray] = []

        prev = 0
        for span_idx, (s, e) in enumerate(zip(starts, ends)):
            if span_idx >= len(self._sentinel_token_ids):
                # Shouldn't happen at our default settings, but guard against it
                break
            sentinel = self._sentinel_token_ids[span_idx]
            enc_parts.append(ids[prev:s])
            enc_parts.append(np.array([sentinel], dtype=np.int64))
            tgt_parts.append(np.array([sentinel], dtype=np.int64))
            tgt_parts.append(ids[s:e])
            prev = e

        enc_parts.append(ids[prev:])
        # Final sentinel marks end-of-targets (T5 convention).
        final_sentinel = self._sentinel_token_ids[min(len(starts), len(self._sentinel_token_ids) - 1)]
        tgt_parts.append(np.array([final_sentinel, self._eos_id], dtype=np.int64))

        return np.concatenate(enc_parts), np.concatenate(tgt_parts)


# ---------- Training driver ----------


def main(cfg: PretrainConfig) -> None:
    """Full pretraining run. Idempotent w.r.t. checkpoints — auto-resumes."""
    # Local imports keep `--help` fast and avoid loading torch when not needed
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logger.info("PretrainConfig:\n%s", yaml.safe_dump(asdict(cfg), default_flow_style=False))
    set_seed(cfg.seed)

    # ----- tokenizer + model -----
    logger.info("Loading tokenizer + model: %s", cfg.base_model)
    # AutoTokenizer downloads SentencePiece (.spm) + tokenizer config. mT5 uses
    # T5TokenizerFast under the hood and supports the <extra_id_*> sentinels.
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    # AutoModelForSeq2SeqLM picks MT5ForConditionalGeneration based on the
    # config — encoder-decoder transformer with span-corruption pretraining.
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model)
    if cfg.gradient_checkpointing:
        # Recompute activations during backward to save VRAM (trades ~25% compute).
        # `use_reentrant=False` is the newer, recommended path in PyTorch 2+.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ----- compute span-corruption token budgets -----
    tokens_per_example, target_length = _compute_input_and_target_lengths(
        inputs_length=cfg.max_seq_length,
        noise_density=cfg.mlm_probability,
        mean_span_length=cfg.mean_span_length,
    )
    logger.info(
        "Span budget: consume %d tokens/example → encoder input %d, decoder target %d",
        tokens_per_example, cfg.max_seq_length, target_length,
    )

    # ----- dataset: stream → tokenize → chunk -----
    # `load_dataset('text', data_files=...)` lazily reads the file line-by-line.
    raw = load_dataset(
        "text",
        data_files=cfg.corpus_path,
        cache_dir=cfg.cache_dir,
    )
    logger.info("Raw dataset: %s", raw)

    def tokenize_fn(batch: dict) -> dict:
        # `add_special_tokens=False` — we add our own EOS in span corruption logic.
        out = tokenizer(batch["text"], add_special_tokens=False)
        return out

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=2,            # 2 workers — T4 boxes have 2 CPU cores
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Concatenate all tokens then chunk into fixed-size examples. This is
    # standard practice for span-corruption pretraining — no padding waste.
    def group_texts(examples: dict) -> dict:
        concatenated = sum(examples["input_ids"], [])
        total_length = (len(concatenated) // tokens_per_example) * tokens_per_example
        chunks = [
            concatenated[i : i + tokens_per_example]
            for i in range(0, total_length, tokens_per_example)
        ]
        return {"input_ids": chunks}

    chunked = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=2,
        desc="Grouping",
    )
    logger.info("Chunked dataset: %s", chunked)

    # Train/validation split
    split = chunked["train"].train_test_split(
        test_size=cfg.validation_fraction,
        seed=cfg.seed,
    )
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info("Train: %d examples | Eval: %d examples",
                len(train_ds), len(eval_ds))

    # ----- collator + training args -----
    collator = T5SpanCorruptionCollator(
        tokenizer=tokenizer,
        noise_density=cfg.mlm_probability,
        mean_span_length=cfg.mean_span_length,
        input_length=cfg.max_seq_length,
        target_length=target_length,
        seed=cfg.seed,
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=False,        # resume from existing checkpoints
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        lr_scheduler_type="linear",
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16 and torch.cuda.is_available(),  # disable fp16 on CPU
        gradient_checkpointing=cfg.gradient_checkpointing,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id or None,
        seed=cfg.seed,
        report_to=["tensorboard"],
        # T5 outputs are seq2seq — Trainer needs to know
        predict_with_generate=False,        # we only need loss for pretraining
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # ----- train (auto-resume from latest checkpoint in output_dir) -----
    checkpoint_dir = Path(cfg.output_dir)
    has_checkpoints = checkpoint_dir.exists() and any(
        p.name.startswith("checkpoint-") for p in checkpoint_dir.iterdir()
    )
    logger.info("Starting training (resume=%s) ...", has_checkpoints)
    trainer.train(resume_from_checkpoint=has_checkpoints)

    # ----- final save + optional Hub push -----
    trainer.save_model(cfg.output_dir)       # writes final model + tokenizer
    if cfg.push_to_hub and cfg.hub_model_id:
        logger.info("Pushing to HuggingFace Hub: %s", cfg.hub_model_id)
        trainer.push_to_hub()

    logger.info("Done.")


# ---------- CLI ----------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/pretrain.yaml",
                   help="Path to YAML config (see configs/pretrain.yaml).")
    # Any field in PretrainConfig can be overridden on the command line.
    for field_name, field_def in PretrainConfig.__dataclass_fields__.items():
        kind = field_def.type if isinstance(field_def.type, type) else str
        # Convert bool fields to --flag/--no-flag style for argparse hygiene
        if kind is bool:
            p.add_argument(f"--{field_name}", dest=field_name, type=lambda x: x.lower() == "true",
                           default=None,
                           help=f"override {field_name} (true/false)")
        else:
            p.add_argument(f"--{field_name}", default=None,
                           help=f"override {field_name}")
    return p


def _cli(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = PretrainConfig.from_yaml(args.config) if Path(args.config).exists() else PretrainConfig()
    # Apply CLI overrides — preserve dataclass field types
    for field_name, field_def in PretrainConfig.__dataclass_fields__.items():
        val = getattr(args, field_name, None)
        if val is None:
            continue
        if isinstance(val, str):
            field_type = field_def.type if isinstance(field_def.type, type) else str
            try:
                val = field_type(val)
            except Exception:
                pass
        setattr(cfg, field_name, val)
    main(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
