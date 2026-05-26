"""Conversation-pair loading + instruction formatting for Phase 3 fine-tuning.

The user (fluent Shona speaker) authors `conversation_pairs.json` containing
hand-crafted question/response pairs across 8 topic areas. This module:

  1. Loads + validates the JSON.
  2. Formats each pair into an instruction-style input + target the fine-tuned
     model will learn to produce.

Format (per project plan):
    INPUT:   "Mubvunzo: {context}\\nMhinduro:"
    TARGET:  "{response}"

`Mubvunzo` = "question", `Mhinduro` = "answer" in Shona — the instruction
prefix is itself Shona, keeping the model fully in-language at inference.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# The 8 topic areas from the project plan. The validator warns if pairs use a
# topic not in this list, but doesn't reject — fine-tuning works regardless of
# topic taxonomy. The list is for corpus balance + dataset card reporting.
KNOWN_TOPICS = {
    "greetings",          # mhoro, makadii, introductions
    "family",             # mhuri, relationships
    "food_daily_life",    # chikafu, daily routine
    "health_body",        # utano, general (not medical advice)
    "weather_nature",     # mamiriro ekunze, environment
    "numbers_time",       # nhamba, nguva, dates
    "common_qa",          # general Q&A
    "proverbs_idioms",    # tsumo nemadimikira, culture
}


@dataclass(frozen=True)
class ConversationPair:
    """One Shona Q&A pair.

    `english_translation` is for the user's QA review + dataset card examples;
    it is NOT fed to the model during fine-tuning.
    """

    id: str
    context: str
    response: str
    topic: str
    english_translation: dict[str, str] = field(default_factory=dict)

    def format_input(self) -> str:
        """The model's encoder input. We use the raw Shona context with NO
        prefix so the trained model behaves like a free-form chatbot — the
        user just types Shona and gets Shona back, no `Mubvunzo:` wrapper.

        This is the harder training setup (the input distribution doesn't
        differ from pretraining), so it requires more training steps and
        early-stopping to land on a usable checkpoint."""
        return self.context

    def format_target(self) -> str:
        """The decoder target — raw Shona response, no prefix."""
        return self.response


def load_conversation_pairs(path: str | Path) -> list[ConversationPair]:
    """Load + validate `conversation_pairs.json`.

    Args:
        path: Path to the JSON file. Expected schema: list of objects with keys
            `id`, `context`, `response`, `topic`, `english_translation` (the
            last is a dict with `context` and `response` keys).

    Returns:
        Validated list of `ConversationPair`.

    Raises:
        ValueError: On malformed JSON shape or missing required keys.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list at the top level")

    pairs: list[ConversationPair] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry #{i} is not a JSON object")
        for key in ("id", "context", "response", "topic"):
            if key not in entry:
                raise ValueError(f"Entry #{i} missing required key '{key}'")
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate id '{entry['id']}' (entry #{i})")
        seen_ids.add(entry["id"])
        if not entry["context"].strip() or not entry["response"].strip():
            raise ValueError(f"Entry '{entry['id']}' has empty context or response")
        pairs.append(
            ConversationPair(
                id=str(entry["id"]),
                context=str(entry["context"]).strip(),
                response=str(entry["response"]).strip(),
                topic=str(entry["topic"]).strip(),
                english_translation=entry.get("english_translation") or {},
            )
        )

    logger.info("Loaded %d conversation pairs from %s", len(pairs), path)
    return pairs


def summarize_pairs(pairs: Iterable[ConversationPair]) -> dict:
    """Compute corpus stats for the dataset card / README."""
    pairs = list(pairs)
    by_topic = Counter(p.topic for p in pairs)
    avg_context_words = sum(len(p.context.split()) for p in pairs) / max(len(pairs), 1)
    avg_response_words = sum(len(p.response.split()) for p in pairs) / max(len(pairs), 1)
    unknown_topics = {t for t in by_topic if t not in KNOWN_TOPICS}

    return {
        "total_pairs": len(pairs),
        "by_topic": dict(by_topic),
        "unknown_topics": sorted(unknown_topics),
        "avg_context_words": round(avg_context_words, 1),
        "avg_response_words": round(avg_response_words, 1),
        "missing_topics": sorted(KNOWN_TOPICS - set(by_topic)),
    }


def build_hf_dataset(pairs: list[ConversationPair]):
    """Convert pairs into a HuggingFace `Dataset` ready for tokenization.

    Returns a `Dataset` with columns `input`, `target`, `topic`, `id`. The
    fine-tuning script tokenizes `input` → encoder, `target` → decoder.
    """
    from datasets import Dataset  # local import keeps `from src.data.conversation import ...` lightweight

    return Dataset.from_list([
        {
            "id": p.id,
            "topic": p.topic,
            "input": p.format_input(),
            "target": p.format_target(),
        }
        for p in pairs
    ])
