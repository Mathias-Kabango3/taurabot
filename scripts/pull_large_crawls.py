"""Pull the three large web-crawl sources (mC4, GlotCC-V1, HPLT 2.0).

Sequential, streaming, idempotent. Designed to run in the background — all
progress goes to stdout/stderr (captured by the agent's background runner).

Each source is configured in `configs/config.yaml` under `sources` with
`tier: large`. The atomic-write logic in HFDatasetSource ensures that a
killed run leaves no half-written file behind.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bible_downloader import load_config
from src.data.hf_source import HFDatasetSource, load_specs


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    log = logging.getLogger("pull_large_crawls")

    cfg = load_config(PROJECT_ROOT / "configs/config.yaml")
    specs = load_specs(cfg, only_tiers={"large"})

    log.info("=" * 70)
    log.info("Pulling %d large-tier sources sequentially", len(specs))
    for s in specs:
        log.info("  - %-12s %s [%s]", s.name, s.repo, s.license)
    log.info("=" * 70)

    failures: list[tuple[str, str]] = []
    for spec in specs:
        log.info("")
        log.info("########## %s ##########", spec.name.upper())
        t0 = time.time()
        try:
            HFDatasetSource(spec, PROJECT_ROOT).download()
            log.info("[%s] done in %.1fs", spec.name, time.time() - t0)
        except Exception as e:
            log.error("[%s] FAILED: %s", spec.name, e, exc_info=True)
            failures.append((spec.name, str(e)))

    log.info("")
    log.info("=" * 70)
    if failures:
        log.error("FAILURES (%d):", len(failures))
        for name, msg in failures:
            log.error("  - %s: %s", name, msg)
        return 1
    log.info("All large sources pulled successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
