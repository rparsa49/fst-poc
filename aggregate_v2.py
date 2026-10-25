#!/usr/bin/env python3
"""
aggregate_v2.py -- Pilot results aggregation

Outputs in artifacts/aggregate/:
    headline.json              -- key numbers for the poster
    scenario_results.csv       -- per-scenario signals + realized gaps
    univariate_correlations.csv -- Spearman per signal, both targets
    binary_aurocs.csv          -- top-quartile AUROC per signal
    fig_hero_auroc_bars.png    -- THE poster figure
    fig_signals_scatter.png    -- 6-panel scatter, winners highlighted
    fig_sparse_forecast.png    -- predicted vs observed (phi2+phi5)

Usage:  python aggregate_v2.py
Claude used for commenting :)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold

SCENARIO_DIR = Path("artifacts/scenario_results")
OUT_DIR = Path("artifacts/aggregate")
SIGNALS = [
    "phi1_imbalance_ratio",
    "phi2_class_cond_ratio",
    "phi3_empty_cells_frac",
    "phi4_embedding_dist",
    "phi5_lab_kl",
    "phi6_probe_acc_gap",
]
SIGNAL_LABELS = {
    "phi1_imbalance_ratio":  "φ₁  FST imbalance ratio",
    "phi2_class_cond_ratio": "φ₂  class-conditional ratio",
    "phi3_empty_cells_frac": "φ₃  empty cells (FST × disease)",
    "phi4_embedding_dist":   "φ₄  embedding distance",
    "phi5_lab_kl":           "φ₅  LAB color KL divergence",
    "phi6_probe_acc_gap":    "φ₆  probe accuracy gap",
}
SIGNAL_KIND = {
    "phi1_imbalance_ratio":  "size",
    "phi2_class_cond_ratio": "dist",
    "phi3_empty_cells_frac": "size",
    "phi4_embedding_dist":   "dist",
    "phi5_lab_kl":           "dist",
    "phi6_probe_acc_gap":    "probe",
}
SEED = 17

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_results() -> pd.DataFrame:
    rows = []
    skipped = 0
    for p in sorted(SCENARIO_DIR.glob("scenario_*.json")):
        with open(p) as f:
            r = json.load(f)
        if r.get("status") != "ok":
            skipped += 1
            continue
        row = {"id": r["scenario"]["id"],
               "delta_auc": r["delta_auc"],
               "delta_sens": r["delta_sens"]}
        row.update(r["signals"])
        rows.append(row)
    if not rows:
        sys.exit(f"[error] No scenario JSONs in {SCENARIO_DIR}")
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    print(f"Loaded {len(df)} scenarios (skipped {skipped} with status != ok)")
    return df


def univariate_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SIGNALS:
        sub = df.dropna(subset=[s])
        for target in ["delta_auc", "delta_sens"]:
            rho, p = spearmanr(sub[s], sub[target])
            rows.append({"signal": s, "target": target,
                         "spearman_rho": float(rho), "p_value": float(p),
                         "n": int(len(sub))})
    return pd.DataFrame(rows)


def binary_aurocs(df: pd.DataFrame, target: str = "delta_auc",
                  quantile: float = 0.75) -> pd.DataFrame:
    """Top-quantile binary classification: can the signal rank scenarios
    into 'high-gap' vs the rest? Reports both raw direction and the
    discriminative magnitude (max of auc, 1-auc)."""
    threshold = np.quantile(df[target], quantile)
    y_bin = (df[target] > threshold).astype(int).values
    rows = []
    for s in SIGNALS:
        sub = df.dropna(subset=[s])
        y_sub = (sub[target] > threshold).astype(int).values
        if y_sub.sum() == 0 or y_sub.sum() == len(y_sub):
            rows.append({"signal": s, "auroc": np.nan,
                         "auroc_disc": np.nan, "direction": "n/a",
                         "n": int(len(sub))})
            continue
        auc = roc_auc_score(y_sub, sub[s].values)
        disc = max(auc, 1 - auc)
        direction = "positive" if auc >= 0.5 else "negative"
        rows.append({"signal": s, "auroc": float(auc),
                     "auroc_disc": float(disc), "direction": direction,
                     "threshold": float(threshold), "n": int(len(sub))})
    return pd.DataFrame(rows)


def cv_forecast(df: pd.DataFrame, features: list[str], target: str = "delta_auc"
                ) -> dict:
    y = df[target].values
    X = df[features].values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    if len(X) < 10:
        return {"n": int(len(X)), "note": "n<10"}
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    preds = np.zeros_like(y)
    for tr, te in kf.split(X):
        preds[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
    return {
        "n": int(len(y)),
        "features": features,
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
        "baseline_mae": float(np.mean(np.abs(y - y.mean()))),
        "predictions": preds.tolist(),
        "observed": y.tolist(),
    }


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def fig_hero_auroc_bars(bin_df: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart: discriminative AUROC per signal. The hero figure."""
    d = bin_df.copy().sort_values("auroc_disc", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = []
    for s in d["signal"]:
        kind = SIGNAL_KIND[s]
        if d.loc[d["signal"] == s, "auroc_disc"].iloc[0] >= 0.70:
            colors.append("#1f6feb")
        elif kind == "dist":
            colors.append("#7aa6da") 
        elif kind == "probe":
            colors.append("#bfbfbf")
        else:
            colors.append("#d8b4a0")

    labels = [SIGNAL_LABELS[s] for s in d["signal"]]
    ax.barh(labels, d["auroc_disc"], color=colors, edgecolor="white")
    ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--", label="random (0.5)")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Discriminative AUROC  (top-quartile ΔAUC vs rest)")
    ax.set_title("Pre-release signals as screening predictors", pad=10)

    for i, v in enumerate(d["auroc_disc"]):
        if not np.isnan(v):
            ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=11)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_path}")


def fig_signals_scatter(df: pd.DataFrame, uni_df: pd.DataFrame,
                        out_path: Path) -> None:
    """6-panel scatter, highlighting winners (phi2, phi5)."""
    winners = {"phi2_class_cond_ratio", "phi5_lab_kl"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, s in zip(axes.flat, SIGNALS):
        sub = df.dropna(subset=[s])
        is_winner = s in winners
        col = "#1f6feb" if is_winner else "#888888"
        ax.scatter(sub[s], sub["delta_auc"], alpha=0.8, s=60,
                   color=col, edgecolor="white", linewidth=0.5)
        if len(sub) >= 5 and not is_winner:
            pass 
        if is_winner and len(sub) >= 5:

            z = np.polyfit(sub[s], sub["delta_auc"], 1)
            xs = np.linspace(sub[s].min(), sub[s].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color="#1f6feb",
                    linewidth=1.5, alpha=0.6)
        rho_row = uni_df[(uni_df["signal"] == s) & (uni_df["target"] == "delta_auc")]
        rho = rho_row["spearman_rho"].iloc[0] if len(rho_row) else np.nan
        title = SIGNAL_LABELS[s] + f"\nρ = {rho:.2f}"
        if is_winner:
            title = "★ " + title
        ax.set_title(title, color=("#1f6feb" if is_winner else "black"))
        ax.set_xlabel("")
        ax.set_ylabel("ΔAUC (I-II minus V-VI)" if ax in axes[:, 0] else "")
        ax.axhline(0, color="gray", linewidth=0.5)
    fig.suptitle("Training-set signals vs realized ΔAUC across 20 curator scenarios",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_path}")


def fig_sparse_forecast(forecast: dict, out_path: Path) -> None:
    if "predictions" not in forecast:
        print("[skip] sparse forecast — not enough data")
        return
    y = np.array(forecast["observed"])
    p = np.array(forecast["predictions"])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y, p, alpha=0.8, s=70, color="#1f6feb",
               edgecolor="white", linewidth=0.5)
    lo = min(y.min(), p.min()) - 0.05
    hi = max(y.max(), p.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.axvline(0, color="gray", linewidth=0.4)
    ax.set_xlabel("Observed ΔAUC")
    ax.set_ylabel("Forecast ΔAUC (5-fold CV)")
    ax.set_title(f"Sparse forecast: φ₂ + φ₅\n"
                 f"MAE = {forecast['mae']:.3f}  (baseline {forecast['baseline_mae']:.3f}),  "
                 f"R² = {forecast['r2']:.2f}")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    df.to_csv(OUT_DIR / "scenario_results.csv", index=False)

    # Descriptive
    print("\n=== Realized gap distribution ===")
    print(f"ΔAUC : mean={df['delta_auc'].mean():.3f}  sd={df['delta_auc'].std():.3f}  "
          f"min={df['delta_auc'].min():.3f}  max={df['delta_auc'].max():.3f}")
    print(f"      scenarios with V-VI ≥ I-II: {(df['delta_auc'] <= 0).sum()} / {len(df)}")
    print(f"ΔSens: mean={df['delta_sens'].mean():.3f}  sd={df['delta_sens'].std():.3f}")

    # Univariate Spearman
    uni = univariate_correlations(df)
    uni.to_csv(OUT_DIR / "univariate_correlations.csv", index=False)
    print("\n=== Univariate Spearman (ΔAUC) ===")
    print(uni[uni["target"] == "delta_auc"][["signal", "spearman_rho",
                                              "p_value"]].to_string(index=False))

    # Binary AUROC -- the headline
    bin_auc = binary_aurocs(df, "delta_auc", 0.75)
    bin_auc.to_csv(OUT_DIR / "binary_aurocs.csv", index=False)
    print("\n=== Binary classification AUROCs (top-quartile ΔAUC) ===")
    print(bin_auc[["signal", "auroc", "auroc_disc", "direction"]].to_string(index=False))

    # Sparse forecast
    sparse = cv_forecast(df, ["phi2_class_cond_ratio", "phi5_lab_kl"], "delta_auc")
    print("\n=== Sparse forecast (φ₂ + φ₅, 5-fold CV) ===")
    print(f"  MAE      = {sparse.get('mae', float('nan')):.4f}")
    print(f"  baseline = {sparse.get('baseline_mae', float('nan')):.4f}")
    print(f"  R²       = {sparse.get('r2', float('nan')):.3f}")

    # Headline numbers for the poster
    best_signal_row = bin_auc.sort_values("auroc_disc", ascending=False).iloc[0]
    headline = {
        "n_scenarios": int(len(df)),
        "delta_auc": {
            "mean": float(df["delta_auc"].mean()),
            "sd": float(df["delta_auc"].std()),
            "range": [float(df["delta_auc"].min()), float(df["delta_auc"].max())],
            "n_inverted": int((df["delta_auc"] <= 0).sum()),
        },
        "best_screening_signal": {
            "signal": best_signal_row["signal"],
            "auroc_disc": float(best_signal_row["auroc_disc"]),
            "direction": best_signal_row["direction"],
        },
        "sparse_forecast": {
            "features": sparse.get("features"),
            "mae": sparse.get("mae"),
            "baseline_mae": sparse.get("baseline_mae"),
            "r2": sparse.get("r2"),
        },
    }
    with open(OUT_DIR / "headline.json", "w") as f:
        json.dump(headline, f, indent=2)
    print("\n=== Poster headline numbers ===")
    print(json.dumps(headline, indent=2))

    # Figures
    print()
    fig_hero_auroc_bars(bin_auc, OUT_DIR / "fig_hero_auroc_bars.png")
    fig_signals_scatter(df, uni, OUT_DIR / "fig_signals_scatter.png")
    fig_sparse_forecast(sparse, OUT_DIR / "fig_sparse_forecast.png")

    print(f"\n[done] All outputs in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
