# au_filter_local.py
from __future__ import annotations
import os, gc, logging, multiprocessing, tempfile, shutil
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import polars as pl
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ========== CONFIG (env overrides allowed) ==========
INPUT_DIR   = Path("data/fineweb")      # folder with pre-downloaded parquet files
OUTPUT_DIR  = Path("data/fineweb_au")   # where AU-filtered files go
MAX_FILES   = 100                       # 0 to process all
URL_COL     = "url"                     # column containing URLs
LOG_TO_FILE = True 
# ====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

handlers = [logging.StreamHandler()]
if LOG_TO_FILE:
    handlers.insert(0, logging.FileHandler(OUTPUT_DIR / "au_filter.log"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PID %(process)d] [%(levelname)s] %(message)s",
    handlers=handlers
)

def list_local_parquet(root: Path, limit: int) -> List[Path]:
    files = sorted(root.rglob("*.parquet"))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No parquet files under: {root.resolve()}")
    return files

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

def process_file(local_path: Path) -> Optional[Path]:
    """
    Read a local parquet, filter for AU-ish URLs, and write to mirrored subdir under OUTPUT_DIR.
    Returns output path if rows survive, else None.
    """
    rel = local_path.relative_to(INPUT_DIR)
    out_subdir = OUTPUT_DIR / rel.parent
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_path = out_subdir / rel.name.replace(".parquet", "_au.parquet")

    if out_path.exists():
        logging.info(f"↩︎ Skipping (exists): {out_path}")
        return out_path

    tmp_out = Path(tempfile.mkstemp(suffix=".parquet", prefix="tmp_au_")[1])

    try:
        # Scan locally (no downloads)
        lf = pl.scan_parquet(str(local_path))

        # Basic URL sanity and single-pass parse -> domain/path columns
        lf = lf.filter(pl.col(URL_COL).is_not_null() & pl.col(URL_COL).str.contains(r"\."))
        lf = lf.with_columns([
            pl.col(URL_COL).map_elements(_extract_domain, return_dtype=pl.Utf8).alias("domain"),
            pl.col(URL_COL).map_elements(_extract_path,  return_dtype=pl.Utf8).alias("path"),
        ])

        df = lf.collect(streaming=True)

        # AU heuristics (domain TLD and common path markers)
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

def main():
    # 1) Find local parquet shards
    parquet_files = list_local_parquet(INPUT_DIR, MAX_FILES)
    logging.info(f"📁 Found {len(parquet_files)} local parquet file(s) under {INPUT_DIR}")

    # 2) Filter with multiprocessing in small batches (stability)
    limited_procs = max(1, multiprocessing.cpu_count() - 4)
    BATCH_SIZE = 10

    written = []
    for i in range(0, len(parquet_files), BATCH_SIZE):
        batch = parquet_files[i:i + BATCH_SIZE]
        with Pool(processes=limited_procs) as pool:
            results = list(tqdm(pool.imap_unordered(process_file, batch), total=len(batch), desc=f"Batch {i//BATCH_SIZE+1}"))
        written.extend([p for p in results if p is not None])
        gc.collect()

    logging.info(f"🏁 Done. AU-filtered files written: {len(written)}")

if __name__ == "__main__":
    main()
