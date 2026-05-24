"""Generic loader for HuggingFace-hosted Shona text datasets.

One class — `HFDatasetSource` — handles every entry in `configs/config.yaml`
under `sources` that has a `repo` field. It supports:

  * Single text fields (`text_field: "text"`)
  * Token-list fields (`text_field: "tokens"` where `row[field]` is `list[str]`)
  * Multi-field concatenation (`text_field: ["headline", "text"]`)
  * Combined splits (`split: "train+validation+test"`)
  * Streaming for large datasets that shouldn't fit in memory

Each row is written as one or more lines in the output file. Paragraph
breaks inside a row (runs of newlines) are split into separate lines; this
gives the cleaning pipeline reasonable units to work with later.

Idempotent — skips download if the output file already exists.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Internal newline runs collapse into a single "\n" — used to split a row's
# text into multiple output lines (one per paragraph).
_NEWLINE_RUN = re.compile(r"\n+")


@dataclass
class HFSourceSpec:
    """Typed view of a `configs/config.yaml` source entry.

    Attributes:
        name:        Logical name (the YAML key, e.g. "wikipedia").
        repo:        HuggingFace dataset repo id.
        config:      Dataset config / subset name (e.g. "sna_Latn"). Can be None.
        split:       Split spec. Supports `+` syntax (e.g. "train+validation").
        text_field:  Where to find the text in each row. See module docstring.
        license:     SPDX-ish license string for the dataset card.
        tier:        Cost tier — "small" / "medium" / "large". Used by callers
                     to decide what to download in a given step.
        output:      Project-relative path where one-line-per-paragraph text
                     will be written.
        held_out:    True if this is an eval set — do NOT include in training.
        streaming:   Use streaming mode (don't cache the full dataset locally).
    """

    name: str
    repo: str
    config: str | None
    split: str
    text_field: str | list[str]
    license: str
    tier: str
    output: str
    held_out: bool = False
    streaming: bool = False

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "HFSourceSpec":
        """Build from a YAML-loaded dict. Tolerant of missing optional fields."""
        return cls(
            name=name,
            repo=d["repo"],
            config=d.get("config"),
            split=d["split"],
            text_field=d["text_field"],
            license=d["license"],
            tier=d["tier"],
            output=d["output"],
            held_out=bool(d.get("held_out", False)),
            streaming=bool(d.get("streaming", False)),
        )


class HFDatasetSource:
    """Download one HuggingFace Shona dataset into a one-line-per-paragraph text file.

    Usage:
        spec = HFSourceSpec.from_dict("wikipedia", cfg["sources"]["wikipedia"])
        src = HFDatasetSource(spec, project_root=Path("."))
        src.download()
    """

    def __init__(self, spec: HFSourceSpec, project_root: Path) -> None:
        self.spec = spec
        self.project_root = Path(project_root).resolve()
        self.output_path = (self.project_root / spec.output).resolve()
        self.manifest_path = self.output_path.with_suffix(".manifest.json")

    def download(self, force: bool = False) -> Path:
        """Load the dataset, extract text, write one paragraph per line.

        Args:
            force: Re-download even if output file already exists.

        Returns:
            Resolved path of the written corpus file.
        """
        if self.output_path.exists() and not force:
            existing_lines = sum(1 for _ in self.output_path.open(encoding="utf-8"))
            logger.info(
                "[%s] output already exists at %s (%d lines) — skipping. Pass force=True to refresh.",
                self.spec.name,
                self.output_path,
                existing_lines,
            )
            return self.output_path

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[%s] load_dataset(repo=%s, config=%s, split=%s, streaming=%s)",
            self.spec.name,
            self.spec.repo,
            self.spec.config,
            self.spec.split,
            self.spec.streaming,
        )

        # `load_dataset` returns either a Dataset (in-memory) or IterableDataset
        # (streaming). Both are iterable — we treat them the same downstream.
        ds = load_dataset(
            self.spec.repo,
            self.spec.config,
            split=self.spec.split,
            streaming=self.spec.streaming,
        )

        n_rows = 0
        n_lines = 0
        n_chars = 0

        # tqdm doesn't know length for streaming datasets — disable bar size hint
        bar = tqdm(ds, desc=self.spec.name, unit=" rows", total=None if self.spec.streaming else len(ds))

        # Atomic-write pattern: write to .tmp first, rename only on full success.
        # This way a crashed/killed download won't leave a half-written file that
        # the idempotency check would mistake for a complete download.
        tmp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for row in bar:
                n_rows += 1
                for paragraph in self._extract_paragraphs(row):
                    fh.write(paragraph + "\n")
                    n_lines += 1
                    n_chars += len(paragraph)
        tmp_path.replace(self.output_path)

        logger.info(
            "[%s] wrote %d rows → %d lines (%.1f MB) at %s",
            self.spec.name,
            n_rows,
            n_lines,
            n_chars / 1024 / 1024,
            self.output_path,
        )

        self._write_manifest(n_rows=n_rows, n_lines=n_lines, n_chars=n_chars)
        return self.output_path

    def _extract_paragraphs(self, row: dict) -> Iterable[str]:
        """Yield one stripped paragraph per `\\n+`-separated chunk in the row.

        Handles three text_field shapes:

        * **string**, `row[field]` is `str`           → split on `\\n+`
        * **string**, `row[field]` is `list[str]`     → join tokens with `' '` (NER datasets)
        * **list[str]**, fields concat'd with newline → split each on `\\n+`
        """
        tf = self.spec.text_field
        raw: str

        if isinstance(tf, str):
            value = row.get(tf)
            if value is None:
                return
            if isinstance(value, list):
                # NER-style: a list of token strings → reconstruct a sentence
                raw = " ".join(str(t) for t in value)
            else:
                raw = str(value)
        elif isinstance(tf, list):
            parts = []
            for field in tf:
                v = row.get(field)
                if v is None:
                    continue
                parts.append(str(v) if not isinstance(v, list) else " ".join(map(str, v)))
            raw = "\n".join(parts)
        else:
            raise TypeError(f"Unsupported text_field type: {type(tf)!r}")

        # Split into paragraphs and yield each non-empty stripped chunk
        for chunk in _NEWLINE_RUN.split(raw):
            stripped = chunk.strip()
            if stripped:
                yield stripped

    def _write_manifest(self, *, n_rows: int, n_lines: int, n_chars: int) -> None:
        """Record provenance + counts next to the corpus file."""
        manifest = {
            "name": self.spec.name,
            "repo": self.spec.repo,
            "config": self.spec.config,
            "split": self.spec.split,
            "license": self.spec.license,
            "held_out": self.spec.held_out,
            "n_rows": n_rows,
            "n_lines": n_lines,
            "n_chars": n_chars,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_specs(cfg: dict, only_tiers: Iterable[str] | None = None) -> list[HFSourceSpec]:
    """Pull every HF-loadable source from the config.

    Args:
        cfg: Full project config (parsed YAML).
        only_tiers: Optional filter — e.g. `{"small"}` to skip the big web crawls.

    Returns:
        List of `HFSourceSpec` for sources whose YAML block has a `repo` key.
    """
    specs = []
    for name, entry in (cfg.get("sources") or {}).items():
        if not isinstance(entry, dict) or "repo" not in entry:
            continue  # Bible (no `repo`) and JW300 (different source) skipped here
        if only_tiers is not None and entry.get("tier") not in only_tiers:
            continue
        specs.append(HFSourceSpec.from_dict(name, entry))
    return specs
