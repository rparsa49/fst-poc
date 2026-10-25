#!/usr/bin/env python3
"""
characterize_subset.py
======================

Usage: python characterize_subset.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = "data/fitzpatrick17k.csv"
IMAGES_DIR = "data/images"


def main() -> None:
    if not Path(CSV).exists():
        sys.exit(f"[error] {CSV} not found")
    df = pd.read_csv(CSV)
    df = df[df["fitzpatrick_scale"].isin([1, 2, 3, 4, 5, 6])].copy()
    df["fst"] = df["fitzpatrick_scale"].astype(int)
    df["fst_group"] = df["fst"].map(
        {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}
    )
    df["host"] = df["url"].str.extract(r"https?://([^/]+)")
    df["malignant"] = (df["three_partition_label"] == "malignant").astype(int)

    # Which md5s are actually downloaded?
    have = set()
    for p in Path(IMAGES_DIR).glob("*.jpg"):
        if p.stat().st_size > 1000:
            have.add(p.stem)
    df["downloaded"] = df["md5hash"].isin(have)

    print(f"Metadata rows (FST 1-6): {len(df):,}")
    print(f"Images downloaded:       {df['downloaded'].sum():,}  "
          f"({df['downloaded'].mean() * 100:.1f}%)")
    print()

    print("=== Host composition (full metadata) ===")
    host_full = df["host"].value_counts()
    for h, n in host_full.items():
        print(f"  {h:>40s}: {n:,}")
    print()

    sub = df[df["downloaded"]].copy()
    print(f"=== FST distribution among downloaded ({len(sub):,} images) ===")
    grouped = sub["fst_group"].value_counts().reindex(["I-II", "III-IV", "V-VI"]).fillna(0).astype(int)
    total = grouped.sum()
    for g, n in grouped.items():
        print(f"  {g:>8s}: {n:>5,}  ({n / max(total, 1) * 100:5.1f}%)")
    if grouped.get("V-VI", 0) > 0:
        ratio = grouped["I-II"] / grouped["V-VI"]
        print(f"  I-II : V-VI ratio: {ratio:.2f}x")
    print()

    print("=== Per-level FST ===")
    per = sub["fst"].value_counts().sort_index()
    for fst, n in per.items():
        print(f"  FST {fst}: {n:,}")
    print()

    print("=== Malignancy by FST group ===")
    mal = sub.groupby("fst_group")["malignant"].agg(["sum", "count", "mean"])
    mal.columns = ["n_malignant", "n_total", "frac_malignant"]
    print(mal.to_string())
    print()

    print("=== Disease coverage (nine_partition_label x FST group) ===")
    joint = pd.crosstab(sub["nine_partition_label"], sub["fst_group"])
    print(joint.to_string())
    print()
    n_empty = (joint == 0).sum().sum()
    n_cells = joint.shape[0] * joint.shape[1]
    print(f"Empty cells: {n_empty} / {n_cells} ({n_empty / n_cells * 100:.1f}%)")


if __name__ == "__main__":
    main()
