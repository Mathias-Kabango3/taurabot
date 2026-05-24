"""Download monolingual Shona text from OPUS (https://opus.nlpl.eu).

Most OPUS Shona corpora are tiny (Ubuntu UI strings, wikimedia snippets) or
redundant with sources we already have (bible-uedin overlaps the BDRSC Bible
we already pulled). The one large unique addition is **NLLB** — Facebook's
No Language Left Behind mono corpus — which is CC-derived and complements
GlotCC + HPLT.

**JW300 is intentionally not implemented**: JW (Jehovah's Witnesses) requested
removal of JW300 from OPUS; the mono download URLs have returned 404 since
~2024. The OPUS team retracted public hosting.

Usage:
    dl = OpusMonoDownloader(corpus="NLLB", version="v1", language="sn")
    dl.download_to(Path("data/raw/opus/nllb_sn.txt"))
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Known OPUS Shona mono URLs that returned HTTP 206/200 on 2026-05-24.
# The fourth field is the SPDX license (best-effort — verify in the dataset card).
OPUS_SHONA_SOURCES = {
    "NLLB":        ("v1",         "sn", "CC-BY-SA-4.0"),    # 105 MB compressed
    "bible-uedin": ("v1",         "sn", "CC-BY-SA-4.0"),    # 1 MB — overlaps BDRSC
    "Ubuntu":      ("v14.10",     "sn", "CC-BY-SA-3.0"),    # 3 KB — UI strings
    "wikimedia":   ("v20230407",  "sn", "CC-BY-SA-3.0"),    # 87 KB — Wikipedia snippets
}


class OpusMonoDownloader:
    """Pull a single OPUS monolingual file (one sentence per line)."""

    BASE = "https://object.pouta.csc.fi"

    def __init__(self, corpus: str, version: str, language: str = "sn",
                 license_: str = "CC-BY-SA-4.0") -> None:
        self.corpus = corpus
        self.version = version
        self.language = language
        self.license_ = license_
        self.url = f"{self.BASE}/OPUS-{corpus}/{version}/mono/{language}.txt.gz"

    def download_to(self, out_path: str | Path, force: bool = False) -> Path:
        """Download the .gz file, decompress, return the path to the .txt.

        Atomic: writes to <out_path>.tmp.gz, decompresses, replaces final file.
        """
        out_path = Path(out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not force:
            logger.info("[%s] %s exists — skipping. force=True to refresh.", self.corpus, out_path)
            return out_path

        gz_path = out_path.with_suffix(out_path.suffix + ".gz")
        tmp_gz = gz_path.with_suffix(".gz.tmp")

        logger.info("[%s] downloading %s", self.corpus, self.url)
        headers = {"User-Agent": "TauraBot-Research/0.1 (Shona NLP research)"}
        with requests.get(self.url, headers=headers, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with tmp_gz.open("wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"{self.corpus}.gz"
            ) as bar:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    fh.write(chunk)
                    bar.update(len(chunk))
        tmp_gz.replace(gz_path)

        logger.info("[%s] decompressing %s → %s", self.corpus, gz_path, out_path)
        tmp_txt = out_path.with_suffix(out_path.suffix + ".tmp")
        with gzip.open(gz_path, "rb") as gz_fh, tmp_txt.open("wb") as txt_fh:
            shutil.copyfileobj(gz_fh, txt_fh, length=1024 * 1024)
        tmp_txt.replace(out_path)

        n_lines = sum(1 for _ in out_path.open(encoding="utf-8"))
        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info("[%s] %d lines, %.1f MB", self.corpus, n_lines, size_mb)

        # Manifest
        manifest = {
            "corpus": self.corpus,
            "version": self.version,
            "language": self.language,
            "license": self.license_,
            "source_url": self.url,
            "n_lines": n_lines,
            "size_bytes": out_path.stat().st_size,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        }
        out_path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        return out_path
