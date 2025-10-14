#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
huggingface_tar_to_standard.py

Automatically extracts all Hugging Face .tar sequences into `train/npy/<seq_name>/`
and merges all per-frame .txt annotations into `train/mot/<seq_name>.txt`.

Features:
- Handles both 'train' and 'test' splits if present.
- Extracts each `.tar` into <split>/npy/<seq_name>/
- Merges per-frame .txt files into <split>/mot/<seq_name>.txt
- Deletes per-frame .txt files after merging.
- Keeps `.npy` and `.tar` files.
- Displays progress bars for both extraction and conversion.

Usage:
  python dataset/huggingface_tar_to_standard.py --root /path/to/ROOT
"""

import argparse
import tarfile
from pathlib import Path
import re
from typing import List, Tuple
from tqdm import tqdm

FRAME_BASENAME_RE = re.compile(r"^(\d+)\.(npy|txt)$")  # e.g., 000123.npy / 000123.txt


# --------------------------- Argument Parsing ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Hugging Face tar files into npy/<seq_name> and merge txt annotations."
    )
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing 'train' and/or 'test' folders with .tar files.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logs.",
    )
    return p.parse_args()


# --------------------------- Helper Functions ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def extract_tar_to_npy(tar_path: Path, npy_root: Path, quiet: bool):
    """Extract one .tar file into npy_root/<seq_name>/"""
    seq_name = tar_path.stem
    seq_out_dir = npy_root / seq_name
    ensure_dir(seq_out_dir)
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(seq_out_dir)
        if not quiet:
            print(f"[EXTRACT] {tar_path.name} -> {seq_out_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to extract {tar_path}: {e}")
    return seq_out_dir


def detect_frame_files(seq_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Return sorted (npy_files, txt_files) under seq_dir."""
    npys, txts = [], []
    for p in seq_dir.iterdir():
        if not p.is_file():
            continue
        m = FRAME_BASENAME_RE.match(p.name)
        if not m:
            continue
        ext = m.group(2)
        if ext == "npy":
            npys.append(p)
        elif ext == "txt":
            txts.append(p)

    def frame_key(path: Path) -> int:
        m = FRAME_BASENAME_RE.match(path.name)
        return int(m.group(1)) if m else 10**9

    npys.sort(key=frame_key)
    txts.sort(key=frame_key)
    return npys, txts


def concat_txts_to_mot(txt_files: List[Path], out_mot_path: Path) -> int:
    """Concatenate per-frame .txt files directly, ensuring newline separation between frames."""
    ensure_dir(out_mot_path.parent)
    total = 0
    with out_mot_path.open("w", encoding="utf-8") as fout:
        for t in txt_files:
            with t.open("r", encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    fout.write(line.rstrip() + "\n")
                    total += 1
    return total



def delete_files(files: List[Path]):
    for f in files:
        try:
            f.unlink()
        except Exception:
            pass


def process_sequence(seq_dir: Path, mot_out_dir: Path):
    seq_name = seq_dir.name
    _, txt_files = detect_frame_files(seq_dir)
    if not txt_files:
        return 0

    out_mot_path = mot_out_dir / f"{seq_name}.txt"
    written = concat_txts_to_mot(txt_files, out_mot_path)
    if written > 0:
        delete_files(txt_files)
    return written


def process_split(root: Path, split: str, quiet: bool):
    split_dir = root / split
    if not split_dir.exists():
        if not quiet:
            print(f"[INFO] Split directory not found, skipping: {split_dir}")
        return

    tar_files = sorted(split_dir.glob("*.tar"))
    if not tar_files:
        if not quiet:
            print(f"[INFO] No tar files found under {split_dir}")
        return

    npy_root = split_dir / "npy"
    mot_root = split_dir / "mot"
    ensure_dir(npy_root)
    ensure_dir(mot_root)

    # Step 1: Extract all .tar files
    if not quiet:
        print(f"[{split.upper()}] Extracting {len(tar_files)} tar files...")
    for tar_file in tqdm(tar_files, desc=f"Extracting {split}", unit="tar"):
        extract_tar_to_npy(tar_file, npy_root, quiet)

    # Step 2: Convert extracted sequences to MOT files
    seq_dirs = [p for p in npy_root.iterdir() if p.is_dir()]
    if not seq_dirs:
        if not quiet:
            print(f"[INFO] No sequences found after extraction in {npy_root}")
        return

    if not quiet:
        print(f"[{split.upper()}] Converting {len(seq_dirs)} sequences...")
    for seq_dir in tqdm(seq_dirs, desc=f"Converting {split}", unit="seq"):
        written = process_sequence(seq_dir, mot_root)
        if not quiet and written > 0:
            print(f"[OK] {seq_dir.name}: {written} lines written.")


# --------------------------- Main ---------------------------

def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"[ERROR] Root path not found: {root}")

    quiet = args.quiet
    if not quiet:
        print(f"[ROOT] {root}")

    for split in ("train", "test"):
        process_split(root, split, quiet)

    if not quiet:
        print("[DONE] Extraction and conversion completed.")


if __name__ == "__main__":
    main()
