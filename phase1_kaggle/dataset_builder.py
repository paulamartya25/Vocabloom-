"""
Phase 1 — Dataset Builder
=========================
Downloads multilingual parallel corpora from HuggingFace Datasets Hub,
cleans and unifies them into a single Parquet dataset, creates train/val/test
splits, writes dataset statistics, and optionally pushes to HuggingFace Hub.

Usage (Kaggle / CLI):
    python dataset_builder.py

Environment variables:
    HF_TOKEN       — HuggingFace write token (required for hub push)
    HF_USERNAME    — HuggingFace username  (required for hub push)
    OUTPUT_DIR     — Output directory      (default: /kaggle/working/multilingual_dataset)
"""

import os
import json
import logging
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ── HuggingFace ────────────────────────────────────────────────────────────────
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from huggingface_hub import HfApi, login

# ── Language detection ─────────────────────────────────────────────────────────
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dataset_builder")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working/multilingual_dataset"))
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_USERNAME  = os.environ.get("HF_USERNAME", "")
HF_REPO_ID   = f"{HF_USERNAME}/multilingual-translation-data" if HF_USERNAME else ""

SHARD_SIZE   = 100_000   # rows per Parquet shard
MIN_CHARS    = 10
MAX_CHARS    = 512
TRAIN_RATIO  = 0.90
VAL_RATIO    = 0.05
# TEST = remainder

# ── Language coverage ─────────────────────────────────────────────────────────
INDIAN_LANGS   = ["hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "ur"]
EUROPEAN_LANGS = ["fr", "de", "es", "it", "pt", "nl", "pl", "ro", "sv", "cs"]
CJK_LANGS      = ["zh", "ja", "ko"]
PIVOT_LANG     = "en"
ALL_LANGS      = [PIVOT_LANG] + INDIAN_LANGS + EUROPEAN_LANGS + CJK_LANGS

# ── Dataset recipes ────────────────────────────────────────────────────────────
# Each recipe: (hf_dataset_name, config_or_None, src_field_path, tgt_field_path,
#               src_lang_key, tgt_lang_key, domain_tag)
# lang_pair strings follow the pattern expected by each dataset.

OPUS100_PAIRS = [
    # Indian
    ("en-hi", "hi"), ("en-bn", "bn"), ("en-ta", "ta"), ("en-te", "te"),
    ("en-mr", "mr"), ("en-gu", "gu"), ("en-kn", "kn"), ("en-ml", "ml"),
    ("en-pa", "pa"), ("en-ur", "ur"),
    # European
    ("en-fr", "fr"), ("en-de", "de"), ("en-es", "es"), ("en-it", "it"),
    ("en-pt", "pt"), ("en-nl", "nl"), ("en-pl", "pl"), ("en-ro", "ro"),
    ("en-sv", "sv"), ("en-cs", "cs"),
    # CJK
    ("en-zh", "zh"), ("en-ja", "ja"), ("en-ko", "ko"),
]

OPUS_BOOKS_PAIRS = [
    ("en-fr", "fr"), ("en-de", "de"), ("en-es", "es"), ("en-it", "it"),
    ("en-pt", "pt"), ("en-nl", "nl"), ("en-ro", "ro"), ("en-cs", "cs"),
    ("en-ru", None),  # skip ru — not in target langs but keep recipe modular
]

TATOEBA_PAIRS = [
    # Indian subset that tatoeba carries
    ("eng-hin", "hi"), ("eng-ben", "bn"), ("eng-tam", "ta"), ("eng-tel", "te"),
    ("eng-mar", "mr"), ("eng-urd", "ur"),
    # CJK
    ("eng-cmn", "zh"), ("eng-jpn", "ja"), ("eng-kor", "ko"),
    # European
    ("eng-fra", "fr"), ("eng-deu", "de"), ("eng-spa", "es"), ("eng-ita", "it"),
    ("eng-por", "pt"), ("eng-nld", "nl"), ("eng-pol", "pl"),
    ("eng-ron", "ro"), ("eng-swe", "sv"), ("eng-ces", "cs"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _text_hash(src: str, tgt: str) -> str:
    return hashlib.md5(f"{src}|||{tgt}".encode("utf-8")).hexdigest()


def _is_valid_text(text: str, min_c: int = MIN_CHARS, max_c: int = MAX_CHARS) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    return min_c <= len(t) <= max_c


def _detect_lang(text: str, expected_lang: str) -> bool:
    """Return True if detected language matches expected (or detection unavailable)."""
    if not LANGDETECT_AVAILABLE:
        return True
    # Only validate well-resourced languages to avoid false negatives on scripts
    VALIDATABLE = {"en", "fr", "de", "es", "it", "pt", "nl", "pl", "ro", "sv",
                   "cs", "hi", "bn", "ta", "te", "mr", "gu", "ml", "ko", "zh",
                   "ja", "ur"}
    if expected_lang not in VALIDATABLE:
        return True
    try:
        detected = detect(text)
        # langdetect returns "zh-cn" / "zh-tw" — normalise
        if detected.startswith("zh"):
            detected = "zh"
        return detected == expected_lang
    except LangDetectException:
        return True


def clean_text(text: str) -> str:
    """Light normalisation: collapse whitespace, strip leading/trailing spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_row(src: str, tgt: str, src_lang: str, tgt_lang: str, domain: str,
               seen: set) -> Optional[Dict]:
    src = clean_text(src)
    tgt = clean_text(tgt)
    if not _is_valid_text(src) or not _is_valid_text(tgt):
        return None
    h = _text_hash(src, tgt)
    if h in seen:
        return None
    seen.add(h)
    return {
        "source_text":  src,
        "target_text":  tgt,
        "source_lang":  src_lang,
        "target_lang":  tgt_lang,
        "domain":       domain,
        "pair":         f"{src_lang}-{tgt_lang}",
    }


def load_opus100(seen: set) -> List[Dict]:
    rows = []
    for lang_pair, tgt_lang in tqdm(OPUS100_PAIRS, desc="opus-100"):
        try:
            ds = load_dataset(
                "Helsinki-NLP/opus-100",
                lang_pair,
                split="train",
                trust_remote_code=True,
            )
            for ex in ds:
                tr = ex.get("translation", {})
                src_text = tr.get("en", "")
                tgt_text = tr.get(tgt_lang, "")
                row = _build_row(src_text, tgt_text, "en", tgt_lang, "opus100", seen)
                if row:
                    rows.append(row)
        except Exception as exc:
            log.warning(f"opus-100 [{lang_pair}] failed: {exc}")
    log.info(f"opus-100 → {len(rows):,} rows")
    return rows


def load_opus_books(seen: set) -> List[Dict]:
    rows = []
    target_tgt_langs = {"fr", "de", "es", "it", "pt", "nl", "ro", "cs"}
    for lang_pair, tgt_lang in tqdm(OPUS_BOOKS_PAIRS, desc="opus_books"):
        if tgt_lang not in target_tgt_langs:
            continue
        try:
            ds = load_dataset(
                "opus_books",
                lang_pair,
                split="train",
                trust_remote_code=True,
            )
            for ex in ds:
                tr = ex.get("translation", {})
                # opus_books keys may be "en"/"fr" etc.
                src_text = tr.get("en", "")
                tgt_text = tr.get(tgt_lang, "")
                row = _build_row(src_text, tgt_text, "en", tgt_lang, "books", seen)
                if row:
                    rows.append(row)
        except Exception as exc:
            log.warning(f"opus_books [{lang_pair}] failed: {exc}")
    log.info(f"opus_books → {len(rows):,} rows")
    return rows


def load_tatoeba(seen: set) -> List[Dict]:
    rows = []
    for lang_pair, tgt_lang in tqdm(TATOEBA_PAIRS, desc="tatoeba_mt"):
        try:
            ds = load_dataset(
                "Helsinki-NLP/tatoeba_mt",
                lang_pair,
                split="test",
                trust_remote_code=True,
            )
            for ex in ds:
                src_text = ex.get("sourceString", "") or ex.get("src", "")
                tgt_text = ex.get("targetString", "") or ex.get("trg", "")
                row = _build_row(src_text, tgt_text, "en", tgt_lang, "tatoeba", seen)
                if row:
                    rows.append(row)
        except Exception as exc:
            log.warning(f"tatoeba_mt [{lang_pair}] failed: {exc}")
    log.info(f"tatoeba_mt → {len(rows):,} rows")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Language-detection validation pass
# ═══════════════════════════════════════════════════════════════════════════════

def validate_language_detection(df: pd.DataFrame, sample_frac: float = 0.05) -> pd.DataFrame:
    """
    Run langdetect on a random sample of rows.  Rows that FAIL detection on
    the sample are NOT removed — only logged — to avoid over-filtering low-
    resource languages where langdetect is unreliable.
    """
    if not LANGDETECT_AVAILABLE:
        log.info("langdetect not available — skipping validation.")
        return df

    sample = df.sample(frac=sample_frac, random_state=42)
    fail_counts: Dict[str, int] = {}

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="lang-detect validate"):
        for field, lang_col in [("source_text", "source_lang"), ("target_text", "target_lang")]:
            lang = row[lang_col]
            if not _detect_lang(row[field], lang):
                fail_counts[lang] = fail_counts.get(lang, 0) + 1

    if fail_counts:
        log.warning(f"Lang-detect mismatches in sample (not removed): {fail_counts}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Parquet sharded writer
# ═══════════════════════════════════════════════════════════════════════════════

def save_split_parquet(df: pd.DataFrame, split_name: str, out_dir: Path) -> List[Path]:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    n_shards = max(1, (len(df) + SHARD_SIZE - 1) // SHARD_SIZE)
    shard_paths = []
    for shard_idx in range(n_shards):
        start = shard_idx * SHARD_SIZE
        end   = min(start + SHARD_SIZE, len(df))
        shard = table.slice(start, end - start)
        shard_path = split_dir / f"shard_{shard_idx:05d}.parquet"
        pq.write_table(shard, shard_path, compression="snappy")
        shard_paths.append(shard_path)
        log.info(f"  Wrote {shard_path.name}  ({end-start:,} rows)")
    return shard_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_statistics(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict:
    total = len(train) + len(val) + len(test)
    stats = {
        "total_rows":        total,
        "train_rows":        len(train),
        "val_rows":          len(val),
        "test_rows":         len(test),
        "language_pairs":    {},
        "domain_distribution": {},
        "source_len_stats":  {},
        "target_len_stats":  {},
    }

    full = pd.concat([train, val, test], ignore_index=True)

    # Per language-pair counts
    for pair, grp in full.groupby("pair"):
        stats["language_pairs"][pair] = int(len(grp))

    # Domain distribution
    for dom, grp in full.groupby("domain"):
        stats["domain_distribution"][dom] = int(len(grp))

    # Length statistics
    for col, key in [("source_text", "source_len_stats"), ("target_text", "target_len_stats")]:
        lengths = full[col].str.len()
        stats[key] = {
            "mean":   float(lengths.mean()),
            "median": float(lengths.median()),
            "std":    float(lengths.std()),
            "min":    int(lengths.min()),
            "max":    int(lengths.max()),
            "p25":    float(lengths.quantile(0.25)),
            "p75":    float(lengths.quantile(0.75)),
            "p95":    float(lengths.quantile(0.95)),
        }

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace Hub push
# ═══════════════════════════════════════════════════════════════════════════════

def push_to_hub(out_dir: Path, repo_id: str, token: str) -> None:
    if not token or not repo_id:
        log.warning("HF_TOKEN / HF_USERNAME not set — skipping hub push.")
        return

    log.info(f"Logging in to HuggingFace Hub as {HF_USERNAME} …")
    login(token=token)
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        log.info(f"Hub repo: {repo_id}")
    except Exception as exc:
        log.error(f"Could not create repo {repo_id}: {exc}")
        return

    # Upload all Parquet shards + stats JSON
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.suffix in {".parquet", ".json"}:
            rel = path.relative_to(out_dir)
            log.info(f"  Uploading {rel} …")
            try:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=str(rel).replace("\\", "/"),
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )
            except Exception as exc:
                log.warning(f"  Upload failed for {rel}: {exc}")

    log.info("Hub push complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {OUTPUT_DIR}")

    # ── 1. Collect data ───────────────────────────────────────────────────────
    seen: set = set()
    all_rows: List[Dict] = []

    all_rows.extend(load_opus100(seen))
    all_rows.extend(load_opus_books(seen))
    all_rows.extend(load_tatoeba(seen))

    if not all_rows:
        log.error("No rows collected — aborting.")
        sys.exit(1)

    log.info(f"Total collected rows (before cleaning): {len(all_rows):,}")

    # ── 2. Build DataFrame & clean ────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    # Drop rows with any null in critical columns
    df.dropna(subset=["source_text", "target_text", "source_lang", "target_lang"], inplace=True)

    # Length filter (already done per row but re-apply on DataFrame level)
    src_len = df["source_text"].str.len()
    tgt_len = df["target_text"].str.len()
    df = df[(src_len >= MIN_CHARS) & (src_len <= MAX_CHARS) &
            (tgt_len >= MIN_CHARS) & (tgt_len <= MAX_CHARS)].copy()

    # Dedup by hash of (source_text, target_text)
    df["_hash"] = df["source_text"] + "|||" + df["target_text"]
    df.drop_duplicates(subset="_hash", inplace=True)
    df.drop(columns="_hash", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(f"Rows after cleaning: {len(df):,}")

    # ── 3. Language-detection validation (sample) ─────────────────────────────
    df = validate_language_detection(df, sample_frac=0.02)

    # ── 4. Shuffle & split ────────────────────────────────────────────────────
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train : n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val :].copy()

    log.info(f"Split sizes — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    # ── 5. Save Parquet shards ────────────────────────────────────────────────
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log.info(f"Writing {split_name} split …")
        save_split_parquet(split_df, split_name, OUTPUT_DIR)

    # ── 6. Dataset statistics ─────────────────────────────────────────────────
    stats = compute_statistics(train_df, val_df, test_df)
    stats["elapsed_seconds"] = round(time.time() - t0, 2)
    stats_path = OUTPUT_DIR / "dataset_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info(f"Statistics saved → {stats_path}")
    log.info(json.dumps(stats, indent=2))

    # ── 7. Push to HuggingFace Hub ────────────────────────────────────────────
    push_to_hub(OUTPUT_DIR, HF_REPO_ID, HF_TOKEN)

    elapsed = time.time() - t0
    log.info(f"Dataset builder finished in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
