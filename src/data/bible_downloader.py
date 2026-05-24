"""Download and parse the eBible.org Shona Bible (BDRSC).

Source: Biblica Bhaibheri Dzvene Rakasununguka MuChiShona Chanhasi 2017
Format: `sna_readaloud.zip` — plain-text chapter files, one verse per line.
License: CC BY-SA 4.0 (© 2005, 2018 Biblica, Inc.)

This module is responsible for retrieving the raw zip, extracting the
chapter files, and concatenating verses into a single corpus file. It
does NOT clean or deduplicate — that's the job of `cleaner.py`.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# eBible's readaloud chapter files are named like:
#   sna_002_GEN_01_read.txt   (Genesis chapter 1)
#   sna_096_REV_22_read.txt   (Revelation chapter 22)
# Format: sna_<book_num_3digit>_<book_abbrev>_<chapter_2digit>_read.txt
_CHAPTER_FILE_RE = re.compile(
    r"^sna_(?P<book_num>\d{3})_(?P<book_abbr>[A-Z0-9]{3})_(?P<chapter>\d{2,3})_read\.txt$"
)


@dataclass
class ChapterRecord:
    """One chapter of the Shona Bible parsed into structured form.

    Attributes:
        book_num:   eBible book index (e.g. "002" = Genesis, "096" = Revelation).
        book_abbr:  3-letter book code (e.g. "GEN", "REV").
        book_name:  Book name as it appears in the Shona text (e.g. "Genesisi").
        chapter:    Chapter number (1-indexed).
        verses:     List of verse strings in canonical order. Verse numbers
                    are NOT preserved (the readaloud format omits them); each
                    string corresponds to one verse of source text.
    """

    book_num: str
    book_abbr: str
    book_name: str
    chapter: int
    verses: list[str]


class BibleDownloader:
    """Download, extract, and parse the eBible.org Shona Bible.

    Usage:
        cfg = yaml.safe_load(open("configs/config.yaml"))
        dl = BibleDownloader(cfg["sources"]["bible"], cfg["paths"]["raw_dir"])
        dl.download()
        records = list(dl.parse_all())
        dl.write_verses_corpus(records, cfg["sources"]["bible"]["verses_file"])
    """

    def __init__(self, bible_cfg: dict, raw_dir: str) -> None:
        """Initialize from the `sources.bible` block of config.yaml.

        Args:
            bible_cfg: Sub-dict with keys `zip_url`, `output`, `name`,
                `license`, `attribution`.
            raw_dir:   Root raw-data directory (e.g. "data/raw").
        """
        self.zip_url: str = bible_cfg["zip_url"]
        self.name: str = bible_cfg["name"]
        self.license: str = bible_cfg["license"]
        self.attribution: str = bible_cfg["attribution"]

        # Resolve all paths under the project root. The config stores them
        # relative; we expand to absolute paths once here so callers don't
        # have to think about CWD.
        self.output_dir: Path = Path(bible_cfg["output"]).resolve()
        self.zip_path: Path = self.output_dir / "sna_readaloud.zip"
        self.extract_dir: Path = self.output_dir / "sna_readaloud"
        self.manifest_path: Path = self.output_dir / "manifest.json"
        self._raw_dir: Path = Path(raw_dir).resolve()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- download ----------

    def download(self, force: bool = False, timeout_seconds: int = 60) -> Path:
        """Fetch the BDRSC zip from eBible.org if not already on disk.

        Args:
            force: Re-download even if the zip already exists.
            timeout_seconds: Per-chunk read timeout for the HTTP request.

        Returns:
            Path to the local zip file.

        Raises:
            requests.HTTPError: If the download returns a non-2xx status.
        """
        if self.zip_path.exists() and not force:
            logger.info("Zip already present at %s — skipping download.", self.zip_path)
            return self.zip_path

        logger.info("Downloading %s → %s", self.zip_url, self.zip_path)
        # Use a polite User-Agent — eBible.org returns 403 to default urllib UA.
        headers = {"User-Agent": "TauraBot-Research/0.1 (Shona NLP research)"}
        with requests.get(
            self.zip_url, headers=headers, stream=True, timeout=timeout_seconds
        ) as resp:
            resp.raise_for_status()
            total_bytes = int(resp.headers.get("content-length", 0))
            with self.zip_path.open("wb") as fh, tqdm(
                total=total_bytes, unit="B", unit_scale=True, desc="bible.zip"
            ) as bar:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    fh.write(chunk)
                    bar.update(len(chunk))

        logger.info("Downloaded %d bytes.", self.zip_path.stat().st_size)
        self._write_manifest()
        return self.zip_path

    def _write_manifest(self) -> None:
        """Record provenance: URL, retrieval time, byte size, license info.

        The manifest sits next to the zip and is the single source of truth
        for the dataset card later (Phase 1 step "corpus stats").
        """
        manifest = {
            "name": self.name,
            "source_url": self.zip_url,
            "license": self.license,
            "attribution": self.attribution,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "zip_bytes": self.zip_path.stat().st_size,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.debug("Wrote manifest to %s", self.manifest_path)

    # ---------- extract ----------

    def extract(self, force: bool = False) -> Path:
        """Extract the readaloud zip into `output_dir/sna_readaloud/`.

        Args:
            force: Re-extract even if the directory already has files.

        Returns:
            Path to the extracted directory.
        """
        if self.extract_dir.exists() and any(self.extract_dir.iterdir()) and not force:
            logger.info("Already extracted at %s — skipping.", self.extract_dir)
            return self.extract_dir

        if not self.zip_path.exists():
            raise FileNotFoundError(
                f"Zip not found at {self.zip_path}. Call .download() first."
            )

        self.extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.zip_path) as zf:
            zf.extractall(self.extract_dir)
        logger.info("Extracted %d entries to %s", len(list(self.extract_dir.iterdir())), self.extract_dir)
        return self.extract_dir

    # ---------- parse ----------

    def parse_all(self) -> Iterator[ChapterRecord]:
        """Yield ChapterRecord for every Bible chapter, in canonical order.

        Skips the non-chapter files in the zip (copyright page, signature,
        keys, intro splash). Iterates filenames alphabetically — which IS
        canonical order thanks to the zero-padded book and chapter numbers.
        """
        if not self.extract_dir.exists():
            raise FileNotFoundError(
                f"Extract dir missing: {self.extract_dir}. Call .extract() first."
            )

        chapter_files = sorted(self.extract_dir.glob("sna_*_read.txt"))
        logger.info("Parsing %d chapter files", len(chapter_files))

        for path in chapter_files:
            match = _CHAPTER_FILE_RE.match(path.name)
            if not match:
                logger.debug("Skipping non-chapter file: %s", path.name)
                continue
            parts = match.groupdict()
            # sna_000_000_000_read.txt is an English intro/about file packaged
            # alongside the real chapters. It uses sentinel book_num "000"
            # which isn't a real biblical book — drop it.
            if parts["book_num"] == "000":
                logger.debug("Skipping intro/about file: %s", path.name)
                continue
            yield self._parse_chapter(path, parts)

    @staticmethod
    def _parse_chapter(path: Path, name_parts: dict) -> ChapterRecord:
        """Parse one chapter file into a ChapterRecord.

        File layout (eBible readaloud format):
            <book_name>.        ← first line, e.g. "Genesisi."
            <chapter_num>.      ← second line, e.g. "1."
            <verse 1 text>      ← one verse per line
            <verse 2 text>
            ...
        Files are UTF-8 with a leading BOM (﻿) that we strip.
        """
        text = path.read_text(encoding="utf-8-sig")  # utf-8-sig strips the BOM
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]

        if len(lines) < 2:
            logger.warning("Chapter file too short: %s", path.name)
            return ChapterRecord(
                book_num=name_parts["book_num"],
                book_abbr=name_parts["book_abbr"],
                book_name="",
                chapter=int(name_parts["chapter"]),
                verses=[],
            )

        book_name = lines[0].rstrip(".")
        verses = lines[2:]  # skip book name (line 0) and chapter number (line 1)

        return ChapterRecord(
            book_num=name_parts["book_num"],
            book_abbr=name_parts["book_abbr"],
            book_name=book_name,
            chapter=int(name_parts["chapter"]),
            verses=verses,
        )

    # ---------- output ----------

    def write_verses_corpus(self, records: list[ChapterRecord], output_path: str | Path) -> Path:
        """Write all verses to a single text file, one verse per line.

        This is the canonical raw-corpus output for the Bible source —
        downstream cleaning (`cleaner.py`) reads from here.

        Args:
            records: Parsed chapter records.
            output_path: Where to write the corpus file.

        Returns:
            Resolved path of the written file.
        """
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        verse_count = 0
        with out.open("w", encoding="utf-8") as fh:
            for rec in records:
                for verse in rec.verses:
                    fh.write(verse.strip() + "\n")
                    verse_count += 1

        logger.info("Wrote %d verses to %s", verse_count, out)
        return out

    def write_records_jsonl(self, records: list[ChapterRecord], output_path: str | Path) -> Path:
        """Write structured per-chapter records as JSONL (one chapter per line).

        Useful for downstream analysis that needs to know which book/chapter
        a verse came from (e.g. for the corpus-stats notebook).
        """
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        logger.info("Wrote %d chapter records to %s", len(records), out)
        return out


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load the project config YAML."""
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
