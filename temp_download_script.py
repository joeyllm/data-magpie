# download_fineweb_samples.py
from __future__ import annotations
import os, logging, tempfile, shutil
from pathlib import Path
from typing import List
from tqdm import tqdm

import fsspec
from huggingface_hub import list_repo_files

# -------- Config (override via env) --------
REPO_ID     = "HuggingFaceFW/fineweb"
# Use "sample/100BT/" for small test shards, or "data/" for full set
SUBFOLDER   = "data/"
MAX_FILES   = 2
OUTPUT_DIR  = Path("data/fineweb")
# -------------------------------------------

BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"

def list_parquet_paths(repo_id: str, subfolder: str) -> List[str]:
    # Get all files under the dataset repo, then filter to the chosen subfolder
    all_files = list_repo_files(repo_id, repo_type="dataset")
    return [p for p in sorted(all_files) if p.startswith(subfolder) and p.endswith(".parquet")]

def download_one(rel_path: str, out_root: Path) -> Path:
    """
    Download a single parquet file (without loading it to memory) using fsspec streaming.
    Returns the local path.
    """
    remote_url = BASE_URL + rel_path
    dst = out_root / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    # Write to a temp path first for atomicity
    tmp = Path(tempfile.mkstemp(suffix=".parquet", prefix="tmp_fw_")[1])
    try:
        with fsspec.open(remote_url, "rb") as rf, open(tmp, "wb") as lf:
            shutil.copyfileobj(rf, lf, length=1024 * 1024)  # 1MB chunks
        shutil.move(str(tmp), str(dst))
        return dst
    except Exception:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        raise

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = list_parquet_paths(REPO_ID, SUBFOLDER)
    if MAX_FILES > 0:
        paths = paths[:MAX_FILES]

    if not paths:
        raise SystemExit(f"No parquet files found under '{SUBFOLDER}' in {REPO_ID}.")

    logging.info(f"📦 Repo: {REPO_ID}")
    logging.info(f"📁 Subfolder: {SUBFOLDER}  |  Files to download: {len(paths)}")
    logging.info(f"📂 Output dir: {OUTPUT_DIR.resolve()}")

    for p in tqdm(paths, desc="Downloading"):
        local = download_one(p, OUTPUT_DIR)
        logging.info(f"✓ {p}  →  {local}")

    logging.info("✅ Done.")

if __name__ == "__main__":
    main()
