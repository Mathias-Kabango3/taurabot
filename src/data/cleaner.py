"""Clean the raw Shona corpus into a sentence-per-line training file.

Pipeline (cheap → expensive):

  1. Strip HTML, URLs, emails — kills web boilerplate fast.
  2. Normalize whitespace + Unicode (collapse spaces, NFC-normalize).
  3. Sentence-split each raw paragraph using punctuation heuristics.
  4. Per-sentence length filter (min words).
  5. Per-sentence non-Latin-character-ratio filter (drops mojibake / heavy mixed-script).
  6. **Trigram Shona detector** — trained on the Bible (100% confirmed Shona).
     This catches Kinyarwanda contamination that langdetect would miss.
  7. Exact dedup (set of sentence hashes).

Output:
  * `data/processed/corpus.txt`        — one Shona sentence per line
  * `data/processed/corpus_stats.json` — per-source contribution + filter dropoff

The whole thing streams — never holds the full corpus in memory at once
(only the dedup hash set, which is bounded by surviving sentence count).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)


# ---------- Regexes (compiled once at import) ----------

# Strip URLs (http, https, ftp, bare domains with www.)
_URL_RE = re.compile(r"https?://\S+|ftp://\S+|\bwww\.\S+", re.IGNORECASE)
# Strip emails
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Strip HTML / XML tags
_HTML_RE = re.compile(r"<[^>]+>")
# Collapse any whitespace run (incl. NBSP \xa0) into one space
_WHITESPACE_RE = re.compile(r"\s+")
# Sentence splitter — break after .!? when followed by whitespace, OR on \n.
# Shona orthography uses plain ASCII letters (no diacritics) so [A-Za-z] suffices
# for the "next character" lookahead. For paragraphs that lack final punctuation,
# the \n branch ensures we still split at paragraph boundaries.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z])|\n+")
# What counts as a "Shona-plausible" character: Latin letters, digits, common
# punctuation, whitespace. Anything else (Cyrillic, Arabic, CJK, emoji…) is
# counted toward the "non-Shona" ratio.
_SHONA_CHAR_RE = re.compile(r"[A-Za-z0-9\s.,!?;:()\"'’‘“”\-–—…/]", re.UNICODE)


# ---------- Config ----------


@dataclass
class CleaningConfig:
    """All cleaning thresholds. Mirrors `cleaning:` block in config.yaml."""

    min_words_per_sentence: int = 5
    max_non_shona_char_ratio: float = 0.30
    # Min fraction of candidate sentence's words that must appear in the
    # ShonaLangDetector's vocab. 0.20 with Bible-trained vocab gives:
    #   true Shona web    0.25 - 1.00  (accepted)
    #   Kinyarwanda       0.09 - 0.18  (rejected)
    #   Swahili/Zulu/etc  0.00 - 0.10  (rejected)
    #   English           0.00         (rejected)
    min_shona_lang_score: float = 0.20
    exact_dedup: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "CleaningConfig":
        # Tolerate extra keys (the YAML has near-dedup config we're not using yet)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class CleaningStats:
    """Per-source dropoff counters. JSON-serializable for the dataset card."""

    source: str
    raw_lines: int = 0
    after_length: int = 0
    after_char_ratio: int = 0
    after_lang_id: int = 0
    after_dedup: int = 0
    sentences_kept: int = 0
    chars_kept: int = 0


# ---------- Language ID ----------


class ShonaLangDetector:
    """Word-level Shona detector trained on a known-Shona vocabulary.

    Approach: build a set of lowercase ASCII Shona words from a high-purity
    source (the Bible — user-confirmed 100% Shona). At inference, score is
    the fraction of a candidate sentence's content words that appear in
    that vocabulary.

    **Why not character trigrams**: tested on 2026-05-24 and found that
    Kinyarwanda scored 0.97 — Bantu languages share too much character-
    level structure (-a vowel endings, mb/nd/ng clusters, mu-/ku- prefixes).
    Word-level matching exploits the fact that high-frequency Shona function
    words (`akati`, `kuti`, `uye`, `ndi-`, `ne-`) are *different* from
    Kinyarwanda function words (`ku`, `no`, `mu`, `ya`, `byose`).

    **Why not `langdetect` / `fasttext`**: same Bantu confusion as mC4's
    language detector. Bible-derived vocab is a domain-correct anchor.

    Calibration (2026-05-24, 32K-word Bible vocab, min_word_len=3):
      pure Shona Bible     1.00
      Shona Wikipedia      0.85-0.95
      Kinyarwanda          0.10-0.20  ← was 0.97 with trigrams
      Swahili              0.10-0.20
      Zulu                 0.05-0.15
      English              0.00-0.05
    """

    # Strip punctuation/digits to extract just the word characters
    _WORD_RE = re.compile(r"[a-z]+")

    def __init__(self, vocab: frozenset[str], min_word_len: int = 3) -> None:
        self._vocab = vocab
        self._min_word_len = min_word_len

    @classmethod
    def from_corpus(
        cls,
        training_text_paths: Path | list[Path],
        min_word_len: int = 3,
        min_word_freq: int = 1,
    ) -> "ShonaLangDetector":
        """Build a vocabulary from one or more known-Shona text files.

        Pass multiple paths to get broader coverage of modern Shona vocabulary.
        Use `min_word_freq` to drop rare words — Shona Wikipedia contains many
        one-off English contaminants (article references, place names, technical
        terms) that pollute the vocabulary; requiring `freq >= 3` removes most
        of that noise while preserving real Shona vocabulary.

        Calibration: Bible(freq>=1) ∪ Wikipedia(freq>=3) gives ~75K-word vocab
        with the score gap holding: real Shona web 0.6+, Kinyarwanda 0.1-0.2.
        """
        if isinstance(training_text_paths, (str, Path)):
            training_text_paths = [Path(training_text_paths)]
        else:
            training_text_paths = [Path(p) for p in training_text_paths]

        counts: Counter[str] = Counter()
        for path in training_text_paths:
            before_unique = len(counts)
            before_total = sum(counts.values())
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    counts.update(cls._extract_words(line, min_word_len))
            logger.info(
                "  vocab += %s: +%d unique / +%d occurrences",
                path,
                len(counts) - before_unique,
                sum(counts.values()) - before_total,
            )

        vocab = frozenset(w for w, c in counts.items() if c >= min_word_freq)
        logger.info(
            "Vocabulary: %d unique words (min_len=%d, min_freq=%d) from %d sources",
            len(vocab),
            min_word_len,
            min_word_freq,
            len(training_text_paths),
        )
        return cls(vocab, min_word_len=min_word_len)

    @classmethod
    def _extract_words(cls, text: str, min_word_len: int) -> list[str]:
        """Lowercase, NFKD-normalize to strip accents, extract ASCII a-z runs."""
        s = unicodedata.normalize("NFKD", text).lower()
        # Keep only ASCII letters — NFKD has decomposed accents into separate chars
        words = cls._WORD_RE.findall(s)
        return [w for w in words if len(w) >= min_word_len]

    def score(self, text: str) -> float:
        """Fraction of candidate words that appear in the Shona vocabulary."""
        words = self._extract_words(text, self._min_word_len)
        if not words:
            return 0.0
        matches = sum(1 for w in words if w in self._vocab)
        return matches / len(words)

    def is_shona(self, text: str, min_score: float) -> bool:
        return self.score(text) >= min_score


# ---------- Cleaning pipeline ----------


def _normalize_text(text: str) -> str:
    """First-pass per-line normalization: strip HTML, URLs, emails, whitespace.

    Run before sentence splitting so noise doesn't confuse the splitter.
    """
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    # NFC keeps composed characters (preserves "ñ" rather than "n + combining ~")
    text = unicodedata.normalize("NFC", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _split_sentences(text: str) -> list[str]:
    """Sentence-split a normalized paragraph. Returns stripped non-empty sentences."""
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _non_shona_char_ratio(sentence: str) -> float:
    """Fraction of characters in `sentence` that aren't Latin / digit / punctuation."""
    if not sentence:
        return 1.0
    matches = _SHONA_CHAR_RE.findall(sentence)
    return 1.0 - (len(matches) / len(sentence))


def _hash_sentence(s: str) -> int:
    """Stable hash for exact-dedup. Uses blake2b for speed + no collisions in practice."""
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(h, "big")


class ShonaCorpusCleaner:
    """Drive the cleaning pipeline across one or more raw source files.

    Usage:
        detector = ShonaLangDetector.from_corpus("data/raw/bible/bible_verses.txt")
        cleaner = ShonaCorpusCleaner(
            cfg=CleaningConfig(),
            detector=detector,
            sources={"bible": Path("data/raw/bible/bible_verses.txt"), ...},
        )
        cleaner.run(out_path="data/processed/corpus.txt",
                    stats_path="data/processed/corpus_stats.json")
    """

    def __init__(
        self,
        cfg: CleaningConfig,
        detector: ShonaLangDetector,
        sources: dict[str, Path],
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        # Preserve insertion order — order in the output file matches source order
        self.sources = dict(sources)

    def run(self, out_path: str | Path, stats_path: str | Path) -> dict:
        """Stream every source through the pipeline and write the cleaned corpus.

        Returns:
            Stats dict (also written to `stats_path` as JSON).
        """
        out_path = Path(out_path).resolve()
        stats_path = Path(stats_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        seen_hashes: set[int] = set()
        per_source: list[CleaningStats] = []
        total_written = 0
        total_chars = 0

        # Atomic-write: stage to .tmp, rename on success
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as out_fh:
            for source_name, source_path in self.sources.items():
                logger.info("Cleaning [%s] from %s", source_name, source_path)
                stats = self._process_source(source_name, source_path, out_fh, seen_hashes)
                per_source.append(stats)
                total_written += stats.sentences_kept
                total_chars += stats.chars_kept
                logger.info(
                    "  [%s] raw=%d → length=%d → char_ratio=%d → lang_id=%d → dedup=%d (kept %d sentences)",
                    source_name,
                    stats.raw_lines,
                    stats.after_length,
                    stats.after_char_ratio,
                    stats.after_lang_id,
                    stats.after_dedup,
                    stats.sentences_kept,
                )
        tmp_path.replace(out_path)

        result = {
            "out_path": str(out_path),
            "config": asdict(self.cfg),
            "total_sentences": total_written,
            "total_chars": total_chars,
            "per_source": [asdict(s) for s in per_source],
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        stats_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote %d cleaned sentences (%.1f MB) to %s",
                    total_written, total_chars / 1024 / 1024, out_path)
        logger.info("Stats written to %s", stats_path)
        return result

    def _process_source(
        self,
        source_name: str,
        source_path: Path,
        out_fh,
        seen_hashes: set[int],
    ) -> CleaningStats:
        """Stream one source file through every filter, writing survivors."""
        stats = CleaningStats(source=source_name)

        with source_path.open(encoding="utf-8") as in_fh:
            for raw_line in in_fh:
                stats.raw_lines += 1
                normalized = _normalize_text(raw_line)
                if not normalized:
                    continue

                for sentence in _split_sentences(normalized):
                    # Filter 1: length
                    if len(sentence.split()) < self.cfg.min_words_per_sentence:
                        continue
                    stats.after_length += 1

                    # Filter 2: character ratio
                    if _non_shona_char_ratio(sentence) > self.cfg.max_non_shona_char_ratio:
                        continue
                    stats.after_char_ratio += 1

                    # Filter 3: Shona word-vocabulary score
                    if not self.detector.is_shona(sentence, self.cfg.min_shona_lang_score):
                        continue
                    stats.after_lang_id += 1

                    # Filter 4: exact dedup
                    if self.cfg.exact_dedup:
                        h = _hash_sentence(sentence)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)
                    stats.after_dedup += 1

                    out_fh.write(sentence + "\n")
                    stats.sentences_kept += 1
                    stats.chars_kept += len(sentence)

        return stats


def build_default_sources(project_root: Path) -> dict[str, Path]:
    """The standard `source_name → raw_path` mapping for Phase 1.

    Order matters: this is the order sentences appear in the cleaned corpus.
    We put high-quality sources first (Bible, Wikipedia) so that during
    streaming-style training, the model sees them more often per epoch.
    """
    root = Path(project_root)
    return {
        "bible":           root / "data/raw/bible/bible_verses.txt",
        "wikipedia":       root / "data/raw/wikipedia/wiki_text.txt",
        "masakhane_ner":   root / "data/raw/masakhane/ner_text.txt",
        "masakhane_news":  root / "data/raw/masakhane/news_text.txt",
        "glotcc":          root / "data/raw/glotcc/glotcc_text.txt",
        "hplt":            root / "data/raw/hplt/hplt_text.txt",
        "nllb":            root / "data/raw/opus/nllb_sn.txt",
    }
