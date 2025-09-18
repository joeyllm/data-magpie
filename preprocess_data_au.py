# preprocess_data_au.py
from __future__ import annotations
import os, gc, random, multiprocessing, tempfile, shutil, logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ================== CONFIG (env overrides) ==================
RAW_DIR        = Path("data/fineweb")   # pre-downloaded parquet files
FILTERED_DIR   = Path("data/fineweb_au")  # where AU-filtered shards go
MAX_RAW_FILES  = 1       # limit raw files to scan/filter
MAX_AU_FILES   = 1       # limit AU-filtered files to load
TEXT_COL       = "text"
URL_COL        = "url"
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.3"
BLOCK_SIZE     = 2048
BATCH_SIZE     = 2
NUM_WORKERS    = 8
SEED           = 1337
KEEP_LOG       = bool(int(os.environ.get("MAGPIE_KEEP_LOG", "1")))
# ============================================================

FILTERED_DIR.mkdir(parents=True, exist_ok=True)
if KEEP_LOG:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [PID %(process)d] [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(FILTERED_DIR / "au_pipeline.log"), logging.StreamHandler()],
    )
else:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

def list_parquet_files(root: Path, limit: int) -> List[Path]:
    files = sorted(root.rglob("*.parquet"))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No parquet files under: {root.resolve()}")
    return files

# -------------------- AU FILTER (LOCAL) ---------------------
def _extract_domain(url: str) -> str:
    try:
        u = urlparse(url)
        if not u.netloc:
            return ""
        host = u.netloc.lower().replace("www.", "")
        return host.split(":")[0]
    except Exception:
        return ""

def _extract_path(url: str) -> str:
    try:
        return urlparse(url).path.lower()
    except Exception:
        return ""

def au_filter_file(local_path: Path, raw_root: Path, out_root: Path) -> Optional[Path]:
    """
    Reads a local parquet, filters for AU-ish URLs, writes to mirrored subdir under out_root.
    Returns the output path if any rows survive, else None.
    """
    # Mirror RAW_DIR/<sub/dirs>/file.parquet -> FILTERED_DIR/<sub/dirs>/file_au.parquet
    rel = local_path.relative_to(raw_root)
    out_subdir = out_root / rel.parent
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / rel.name.replace(".parquet", "_au.parquet")

    if out_path.exists():
        logging.info(f"↩︎ Skipping (exists): {out_path}")
        return out_path

    # Work in a temp file to avoid partial writes
    tmp_out = Path(tempfile.mkstemp(suffix=".parquet", prefix="tmp_au_")[1])

    try:
        lf = pl.scan_parquet(str(local_path))

        lf = lf.filter(pl.col(URL_COL).is_not_null() & pl.col(URL_COL).str.contains(r"\."))

        lf = lf.with_columns([
            pl.col(URL_COL).map_elements(_extract_domain, return_dtype=pl.Utf8).alias("domain"),
            pl.col(URL_COL).map_elements(_extract_path,  return_dtype=pl.Utf8).alias("path"),
        ])

        df = lf.collect(engine="streaming")

        df_au = df.filter(
            (pl.col("domain").str.ends_with(".au")) |
            (pl.col("path").str.contains(r"(^|/)(au|en-au)(/|$)", literal=False)) |
            (pl.col("path").str.contains(r"australia", literal=False))
        )

        if df_au.height == 0:
            logging.info(f"⚠️ No AU rows in {rel}")
            try: tmp_out.unlink(missing_ok=True)
            except Exception: pass
            return None

        df_au.write_parquet(tmp_out)
        shutil.move(str(tmp_out), str(out_path))
        logging.info(f"✅ Saved {df_au.height} AU rows → {out_path}")
        return out_path

    except Exception as e:
        logging.error(f"❌ Error filtering {rel}: {e}")
        try: tmp_out.unlink(missing_ok=True)
        except Exception: pass
        return None

# ---------------- CLEAN → TOKENIZE → PACK -------------------
def load_and_clean_many(paths: List[Path]) -> pl.DataFrame:
    lf = pl.scan_parquet([str(p) for p in paths])
    lf = (
        lf
        .filter(pl.col(TEXT_COL).is_not_null())
        .with_columns(pl.col(text_col).cast(pl.Utf8).str.strip_chars().alias(text_col))
        .filter(pl.col(TEXT_COL).str.len_chars() >= 50)
        .filter(pl.col(TEXT_COL).str.len_chars() <= 20000)
        .select([pl.col(TEXT_COL)])
        .unique(maintain_order=True)
    )
    return lf.collect(engine="streaming")

def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.unk_token
    return tok

def tokenize_texts(tokenizer, texts: List[str]) -> List[List[int]]:
    enc = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
    return enc["input_ids"]

class PackedDataset(Dataset):
    def __init__(self, all_ids: List[List[int]], block_size: int, eos_id: Optional[int]):
        flat: List[int] = []
        for ids in all_ids:
            if ids:
                flat.extend(ids)
                if eos_id is not None:
                    flat.append(eos_id)
        usable = (len(flat) // block_size) * block_size
        self.data = torch.tensor(flat[:usable], dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        s = idx * self.block_size
        e = s + self.block_size
        x = self.data[s:e]
        return {"input_ids": x.clone(), "attention_mask": torch.ones_like(x), "labels": x.clone()}

# ------------------------------ MAIN -----------------------
def main():
    random.seed(SEED)

    # 1) AU-filter the local shards
    raw_paths = list_parquet_files(RAW_DIR, MAX_RAW_FILES)
    logging.info(f"📁 AU filtering {len(raw_paths)} raw files from {RAW_DIR} → {FILTERED_DIR}")
    limited_procs = max(1, multiprocessing.cpu_count() - 4)
    BATCH = 10

    # Batch + Pool for memory stability
    written: List[Path] = []
    for i in range(0, len(raw_paths), BATCH):
        batch = raw_paths[i:i+BATCH]
        with multiprocessing.Pool(processes=limited_procs) as pool:
            results = list(tqdm(pool.imap_unordered(
                lambda p: au_filter_file(p, RAW_DIR, FILTERED_DIR), batch),
                total=len(batch), desc=f"Batch {i//BATCH + 1}"
            ))
        written.extend([p for p in results if p is not None])
        gc.collect()

    if not written:
        raise RuntimeError("No AU-filtered files produced. Check URL column name and filters.")

    # 2) Load up to N AU-filtered files and build dataset
    au_paths = sorted([p for p in FILTERED_DIR.rglob("*_au.parquet")])
    if MAX_AU_FILES > 0:
        au_paths = au_paths[:MAX_AU_FILES]
    logging.info(f"🧾 Using {len(au_paths)} AU-filtered files for tokenization.")
    for p in au_paths: logging.info(f"  • {p}")

    df = load_and_clean_many(au_paths)
    texts = df[TEXT_COL].to_list()
    logging.info(f"🧹 Cleaned texts: {len(texts):,}")

    tok = build_tokenizer(TOKENIZER_NAME)
    ids = tokenize_texts(tok, texts)
    random.shuffle(ids)

    ds = PackedDataset(ids, BLOCK_SIZE, getattr(tok, "eos_token_id", None))
    logging.info(f"📦 Packed blocks: {len(ds):,} (block_size={BLOCK_SIZE}, tokens={len(ds)*BLOCK_SIZE:,})")

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    batch = next(iter(dl))
    for k, v in batch.items():
        logging.info(f"{k}: {tuple(v.shape)}")

if __name__ == "__main__":
    main()
