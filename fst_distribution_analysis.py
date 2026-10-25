#!/usr/bin/env python3
"""
Fitzpatrick17k FST Distribution Analysis
=============================================================

Downloads the Fitzpatrick17k metadata CSV,
computes the Fitzpatrick Skin Type (FST) distribution overall and grouped
into the I-II / III-IV / V-VI bins used in the project's forecasting
framework, and writes:

  results/fst_distribution.csv          -- counts per FST level
  results/fst_grouped_distribution.csv  -- counts per FST group (I-II, etc.)
  results/fst_summary.md                -- markdown summary table
  results/fst_distribution.png          -- bar chart

Project: Forecasting demographic performance gaps in dermatological AI
Author:  Roya W. Parsa  (f007z0x@dartmouth.edu)
Claude used to comment :)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Data source
# -----------------------------------------------------------------------------
# Official Fitzpatrick17k metadata
METADATA_URL = (
    "https://raw.githubusercontent.com/"
    "mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
)
METADATA_FILENAME = "fitzpatrick17k.csv"

# FST grouping used in the forecasting framework.
# Unknown / -1 values are handled separately.
FST_GROUPS = {
    1: "I-II",
    2: "I-II",
    3: "III-IV",
    4: "III-IV",
    5: "V-VI",
    6: "V-VI",
}
GROUP_ORDER = ["I-II", "III-IV", "V-VI", "Unknown"]


# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
def download_metadata(data_dir: Path) -> Path:
    """Download the Fitzpatrick17k metadata CSV if not already cached."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / METADATA_FILENAME

    if target.exists() and target.stat().st_size > 0:
        print(f"[cache] Using existing metadata at {target}")
        return target

    print(f"[download] Fetching {METADATA_URL}")
    try:
        resp = requests.get(METADATA_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        sys.exit(
            f"[error] Failed to download metadata: {e}\n"
            "If the compute node has no internet access, run this script on "
            "the Discovery login node (andes/polaris) first to cache the file, "
            "then re-submit the SLURM job."
        )

    target.write_bytes(resp.content)
    print(f"[download] Wrote {len(resp.content):,} bytes to {target}")
    return target


# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
def load_metadata(csv_path: Path) -> pd.DataFrame:
    """Load the metadata and identify the FST column (resilient to renames)."""
    df = pd.read_csv(csv_path)
    print(f"[load] {len(df):,} rows, columns: {list(df.columns)}")

    # The canonical column is 'fitzpatrick_scale'. Older / forked copies have
    # used 'fitzpatrick' or 'fitzpatrick_type'. Find whichever is present.
    candidates = ["fitzpatrick_scale", "fitzpatrick", "fitzpatrick_type"]
    fst_col = next((c for c in candidates if c in df.columns), None)
    if fst_col is None:
        sys.exit(
            f"[error] No Fitzpatrick column found. Tried: {candidates}. "
            f"Got columns: {list(df.columns)}"
        )
    print(f"[load] Using FST column: '{fst_col}'")
    df = df.rename(columns={fst_col: "fst"})
    return df


def compute_distributions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-level distribution, grouped distribution)."""
    # Per-level: counts for 1..6 plus any unknown (-1, NaN, etc.)
    fst_clean = df["fst"].where(df["fst"].isin([1, 2, 3, 4, 5, 6]), other=np.nan)
    per_level = (
        fst_clean.value_counts(dropna=False)
        .rename_axis("fst")
        .reset_index(name="count")
        .sort_values("fst", na_position="last")
        .reset_index(drop=True)
    )
    per_level["fst"] = per_level["fst"].apply(
        lambda x: int(x) if pd.notna(x) else "Unknown"
    )
    per_level["share"] = per_level["count"] / per_level["count"].sum()

    # Grouped
    def to_group(v):
        if pd.isna(v) or v not in FST_GROUPS:
            return "Unknown"
        return FST_GROUPS[int(v)]

    df = df.assign(fst_group=df["fst"].apply(to_group))
    grouped = (
        df["fst_group"]
        .value_counts()
        .reindex(GROUP_ORDER, fill_value=0)
        .rename_axis("fst_group")
        .reset_index(name="count")
    )
    grouped["share"] = grouped["count"] / grouped["count"].sum()
    return per_level, grouped


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
def write_markdown_summary(
    per_level: pd.DataFrame, grouped: pd.DataFrame, out_path: Path
) -> None:
    total = int(grouped["count"].sum())

    lines = []
    lines.append("# Fitzpatrick17k FST Distribution -- PoC Output\n")
    lines.append(f"**Total images:** {total:,}\n")

    lines.append("## Grouped distribution (forecasting framework bins)\n")
    lines.append("| FST group | Count | Share |")
    lines.append("|---|---:|---:|")
    for _, r in grouped.iterrows():
        lines.append(
            f"| {r['fst_group']} | {int(r['count']):,} | {r['share']*100:.1f}% |"
        )

    lines.append("\n## Per-level distribution\n")
    lines.append("| FST | Count | Share |")
    lines.append("|---|---:|---:|")
    for _, r in per_level.iterrows():
        lines.append(f"| {r['fst']} | {int(r['count']):,} | {r['share']*100:.1f}% |")

    # Key statistic: I-II vs V-VI ratio
    i_ii = int(grouped.loc[grouped["fst_group"] == "I-II", "count"].iloc[0])
    v_vi = int(grouped.loc[grouped["fst_group"] == "V-VI", "count"].iloc[0])
    if v_vi > 0:
        ratio = i_ii / v_vi
        lines.append(
            f"\n**I-II : V-VI ratio = {ratio:.2f}x** "
            f"({i_ii:,} lightest-skin images vs. {v_vi:,} darkest-skin images). "
            "This imbalance is the empirical motivation for the forecasting "
            "framework: if a curator can predict downstream performance gaps "
            "from this kind of training-set property alone, demographic "
            "auditing becomes a pre-release obligation."
        )

    out_path.write_text("\n".join(lines))
    print(f"[write] {out_path}")


def write_plot(grouped: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#f5d6c0", "#c98a5d", "#5a3a2a", "#bdbdbd"]
    bars = ax.bar(grouped["fst_group"], grouped["count"], color=colors)
    ax.set_xlabel("Fitzpatrick Skin Type group")
    ax.set_ylabel("Number of images")
    ax.set_title("Fitzpatrick17k: image counts by FST group")
    for b, c in zip(bars, grouped["count"]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{int(c):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[write] {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = download_metadata(args.data_dir)
    df = load_metadata(csv_path)
    per_level, grouped = compute_distributions(df)

    print("\n=== Per-level FST distribution ===")
    print(per_level.to_string(index=False))
    print("\n=== Grouped FST distribution ===")
    print(grouped.to_string(index=False))

    per_level.to_csv(args.out_dir / "fst_distribution.csv", index=False)
    grouped.to_csv(args.out_dir / "fst_grouped_distribution.csv", index=False)
    write_markdown_summary(per_level, grouped, args.out_dir / "fst_summary.md")
    write_plot(grouped, args.out_dir / "fst_distribution.png")

    print("\n[done] PoC outputs written to", args.out_dir.resolve())


if __name__ == "__main__":
    main()
