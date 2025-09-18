# preprocess_data.py
from __future__ import annotations
import os, random
from pathlib import Path
from typing import List, Optional

import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------- Configs ----------
INPUT_DIR      = Path("data/fineweb_au")
MAX_FILES      = 1                    # 0 for all files
TEXT_COL       = "text"
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.3"
BLOCK_SIZE     = 2048
BATCH_SIZE     = 2
NUM_WORKERS    = 8
SEED           = 42
# -----------------------------------------------------

def pick_parquet_files(root: Path, max_files: int) -> List[str]:
    files = sorted([str(p) for p in root.rglob("*.parquet")])
    if not files:
        raise FileNotFoundError(f"No parquet files under: {root.resolve()}")
    return files[:max_files] if max_files > 0 else files

def load_and_clean_many(paths: List[str], text_col: str = TEXT_COL) -> pl.DataFrame:
    # Polars can scan multiple files at once
    lf = pl.scan_parquet(paths)

    lf = (
        lf
        .filter(pl.col(text_col).is_not_null())
        .with_columns(pl.col(text_col).cast(pl.Utf8).str.strip_chars().alias(text_col))
        .filter(pl.col(text_col).str.len_chars() >= 50)
        .filter(pl.col(text_col).str.len_chars() <= 20000)
        .select([pl.col(text_col)])
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

def main():
    random.seed(SEED)
    paths = pick_parquet_files(INPUT_DIR, MAX_FILES)
    print(f"📁 Loading up to {MAX_FILES} files from {INPUT_DIR} ({len(paths)} selected)")
    for p in paths: print("  •", p)

    df = load_and_clean_many(paths, TEXT_COL)
    texts = df[TEXT_COL].to_list()
    print(f"🧹 Cleaned texts: {len(texts):,}")

    tok = build_tokenizer(TOKENIZER_NAME)
    ids = tokenize_texts(tok, texts)
    random.shuffle(ids)

    ds = PackedDataset(ids, BLOCK_SIZE, getattr(tok, "eos_token_id", None))
    print(f"📦 Packed blocks: {len(ds):,}  (block_size={BLOCK_SIZE}, total_tokens={len(ds)*BLOCK_SIZE:,})")

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    batch = next(iter(dl))
    for k, v in batch.items():
        print(k, tuple(v.shape))

if __name__ == "__main__":
    main()
