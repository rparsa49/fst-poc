#!/usr/bin/env python3
"""
Forecasting Demographic Performance Gaps 
================================================================

Subcommands:
    python pipeline.py download_images 
    python pipeline.py extract_features 
    python pipeline.py generate_scenarios
    python pipeline.py run_scenario --id K
    python pipeline.py aggregate

The pipeline is resumable end-to-end: re-running any stage skips work already
done (existing image files, existing feature cache, existing scenario JSONs).

Author: Roya W. Parsa
Claude used to comment :)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
METADATA_CSV = "data/fitzpatrick17k.csv"
IMAGES_DIR = "data/images"
FEATURES_PATH = "artifacts/features.npz"
SCENARIOS_PATH = "artifacts/scenarios.json"
SCENARIO_RESULTS_DIR = "artifacts/scenario_results"
AGGREGATE_DIR = "artifacts/aggregate"
DOWNLOAD_FAILURE_LOG = "logs/download_failures.csv"

IMG_SIZE = 224
FEATURE_DIM = 2048
LAB_BINS = 8

K_PILOT = 20 
GLOBAL_SEED = 17

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_metadata() -> pd.DataFrame:
    if not Path(METADATA_CSV).exists():
        sys.exit(
            f"[error] {METADATA_CSV} not found. Copy from your fst-poc "
            "directory or re-download from the Groh repo."
        )
    df = pd.read_csv(METADATA_CSV)
    df = df[df["fitzpatrick_scale"].isin([1, 2, 3, 4, 5, 6])].copy()
    df["fst"] = df["fitzpatrick_scale"].astype(int)
    df["fst_group"] = df["fst"].map(
        {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}
    )
    df["malignant"] = (df["three_partition_label"] == "malignant").astype(int)
    return df.reset_index(drop=True)


# =============================================================================
# Stage 1: Download images (run on LOGIN node)
# =============================================================================
def cmd_download_images(args) -> None:
    import requests
 
    ensure_dirs(IMAGES_DIR, Path(DOWNLOAD_FAILURE_LOG).parent)
    df = load_metadata()
    before = len(df)
    df = df[df["url"].str.contains("atlasdermatologico", na=False)].reset_index(drop=True)
    log(f"Filtered {before:,} -> {len(df):,} URLs "
        f"(atlasdermatologico only; dermaamin URLs skipped)")
 
    failure_records = []
    reason_counter = Counter()
    timeout_md5s = []
 
    def fetch(row, timeout_sec):
        url = row["url"]
        md5 = row["md5hash"]
        target = Path(IMAGES_DIR) / f"{md5}.jpg"
        if target.exists() and target.stat().st_size > 1000:
            return ("skip", md5, url, "")
        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout_sec)
            if r.status_code != 200:
                return ("fail", md5, url, f"http_{r.status_code}")
            if len(r.content) < 1000:
                return ("fail", md5, url, f"small_{len(r.content)}b")
            ct = r.headers.get("Content-Type", "")
            if "image" not in ct and "octet" not in ct:
                return ("fail", md5, url, f"bad_ct_{ct[:40]}")
            target.write_bytes(r.content)
            return ("ok", md5, url, "")
        except requests.exceptions.Timeout:
            return ("fail", md5, url, "timeout")
        except requests.exceptions.SSLError:
            return ("fail", md5, url, "ssl")
        except requests.exceptions.ConnectionError as e:
            return ("fail", md5, url, f"conn_{str(e)[:40]}")
        except Exception as e:
            return ("fail", md5, url, f"err_{type(e).__name__}")
 
    n_ok = n_skip = n_fail = 0
    log(f"Pass 1: {args.workers} workers, {args.timeout}s timeout")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch, row, args.timeout) for _, row in df.iterrows()]
        for i, fut in enumerate(as_completed(futures), 1):
            status, md5, url, reason = fut.result()
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                reason_counter[reason] += 1
                if reason == "timeout":
                    timeout_md5s.append(md5)
                else:
                    failure_records.append((md5, url, reason))
            if i % 200 == 0:
                top = ", ".join(f"{r}={c}" for r, c in reason_counter.most_common(3))
                log(f"  [{i:>4}/{len(df)}]  ok={n_ok}  skip={n_skip}  "
                    f"fail={n_fail}  ({top})")
 
    log(f"Pass 1 done: ok={n_ok}  skip={n_skip}  fail={n_fail}")
 
    if timeout_md5s and not args.no_retry:
        log(f"Pass 2: retrying {len(timeout_md5s)} timeouts sequentially "
            f"with {args.timeout * 2}s timeout")
        retry_df = df[df["md5hash"].isin(timeout_md5s)].reset_index(drop=True)
        n_retry_ok = 0
        for i, row in retry_df.iterrows():
            status, md5, url, reason = fetch(row, args.timeout * 2)
            if status == "ok":
                n_ok += 1
                n_retry_ok += 1
            elif status != "skip":
                failure_records.append((md5, url, reason))
            if (i + 1) % 100 == 0:
                log(f"  retry [{i + 1}/{len(retry_df)}] ok so far: {n_retry_ok}")
        log(f"Pass 2 done: recovered {n_retry_ok} / {len(timeout_md5s)} timeouts")
 
    if failure_records:
        pd.DataFrame(failure_records, columns=["md5", "url", "reason"]).to_csv(
            DOWNLOAD_FAILURE_LOG, index=False)
        log(f"Failure details: {DOWNLOAD_FAILURE_LOG}")
        log(f"Top failure modes: {dict(reason_counter.most_common(5))}")
 
    available = n_ok + n_skip
    pct = available / max(len(df), 1) * 100
    log(f"FINAL: {available:,} / {len(df):,} atlasdermatologico images "
        f"available on disk ({pct:.1f}%)")

# =============================================================================
# Stage 2: Extract ResNet-50 features + LAB histograms
# =============================================================================
def cmd_extract_features(args) -> None:
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    from PIL import Image
    from skimage import color as skcolor

    ensure_dirs(Path(FEATURES_PATH).parent)
    df = load_metadata()

    existing = []
    for md5 in df["md5hash"]:
        p = Path(IMAGES_DIR) / f"{md5}.jpg"
        if p.exists() and p.stat().st_size > 1000:
            existing.append(md5)
    log(f"{len(existing):,} images on disk (of {len(df):,} expected)")
    if len(existing) < 500:
        sys.exit("[error] Too few images to proceed. Run download_images first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Using device: {device}")

    # ResNet-50 with modern weights API 
    try:
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
        model = tvm.resnet50(weights=weights)
    except AttributeError:
        log("[info] Falling back to legacy pretrained=True API")
        model = tvm.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    features = np.zeros((len(existing), FEATURE_DIM), dtype=np.float32)
    lab_hists = np.zeros((len(existing), LAB_BINS ** 3), dtype=np.float32)
    valid = np.zeros(len(existing), dtype=bool)

    BATCH = 64
    batch_imgs, batch_idx = [], []

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            x = torch.stack(batch_imgs).to(device)
            f = model(x).cpu().numpy()
        for k, idx in enumerate(batch_idx):
            features[idx] = f[k]
            valid[idx] = True
        batch_imgs.clear()
        batch_idx.clear()

    n_failed_open = 0
    for i, md5 in enumerate(existing):
        try:
            img = Image.open(Path(IMAGES_DIR) / f"{md5}.jpg").convert("RGB")
        except Exception:
            n_failed_open += 1
            continue

        # LAB histogram on a downscaled copy
        arr = np.asarray(img.resize((128, 128))) / 255.0
        lab = skcolor.rgb2lab(arr)
        bins = [
            np.linspace(0, 100, LAB_BINS + 1),
            np.linspace(-128, 127, LAB_BINS + 1),
            np.linspace(-128, 127, LAB_BINS + 1),
        ]
        h, _ = np.histogramdd(lab.reshape(-1, 3), bins=bins)
        h = h.flatten()
        lab_hists[i] = h / max(h.sum(), 1)

        batch_imgs.append(tfm(img))
        batch_idx.append(i)
        if len(batch_imgs) >= BATCH:
            flush()
        if (i + 1) % 500 == 0:
            log(f"  [{i + 1:>5}/{len(existing)}]")
    flush()

    # Keep only successfully-processed entries
    md5_keep = [m for i, m in enumerate(existing) if valid[i]]
    log(f"Successfully featurized {len(md5_keep):,} images "
        f"(failed to open: {n_failed_open})")
    np.savez(
        FEATURES_PATH,
        md5=np.array(md5_keep),
        features=features[valid],
        lab_hists=lab_hists[valid],
    )
    log(f"Saved features to {FEATURES_PATH}")


# =============================================================================
# Stage 3: Generate K scenarios (adaptive sizing)
# =============================================================================
def cmd_generate_scenarios(args) -> None:
    """Produce a deterministic scenarios.json. Sizes adapt to the actual pool
    of available images, so this works with full Fitz17k or a partial subset."""
    ensure_dirs(Path(SCENARIOS_PATH).parent)
    rng = np.random.default_rng(GLOBAL_SEED)

    df = load_metadata()
    feat = np.load(FEATURES_PATH, allow_pickle=True)
    have_md5 = set(feat["md5"].tolist())
    df = df[df["md5hash"].isin(have_md5)].reset_index(drop=True)
    log(f"{len(df):,} images available for scenario sampling")

    pool = df.groupby(["fst_group", "malignant"]).size().to_dict()
    log("Available by (fst_group, malignant):")
    for k, v in pool.items():
        log(f"  {k}: {v}")

    n_light = pool.get(("I-II", 0), 0) + pool.get(("I-II", 1), 0)
    n_dark = pool.get(("V-VI", 0), 0) + pool.get(("V-VI", 1), 0)
    log(f"Total light (I-II): {n_light:,}   Total dark (V-VI): {n_dark:,}")

    if n_dark < 50 or n_light < 200:
        log("[warn] Very small pool. Scenarios will be limited; "
            "the forecasting model may have wide confidence intervals.")

    # Adaptive size choices: keep each scenario well within available data
    # so the stratified split still leaves a usable test set per group.
    # We cap any scenario's total N at 60% of the available combined pool.
    max_total = int(0.6 * (n_light + n_dark))
    candidate_sizes = sorted({s for s in [800, 1500, 3000, 6000] if s <= max_total})
    if not candidate_sizes:
        candidate_sizes = [max(400, int(0.5 * (n_light + n_dark)))]
    log(f"Candidate scenario sizes: {candidate_sizes}")

    mal_fracs = [0.20, 0.35, 0.50]
    k = args.k or K_PILOT
    scenarios = []
    for i in range(k):
        dark_share = float(rng.uniform(0.05, 0.50))
        N = int(rng.choice(candidate_sizes))
        mal_frac = float(rng.choice(mal_fracs))

        n_dark_target = int(N * dark_share)
        n_light_target = N - n_dark_target
        if n_dark_target > int(0.7 * n_dark) or n_light_target > int(0.7 * n_light):
            scale = min(
                0.7 * n_dark / max(n_dark_target, 1),
                0.7 * n_light / max(n_light_target, 1),
            )
            N = max(400, int(N * scale))
        scenarios.append({
            "id": i,
            "seed": GLOBAL_SEED + i,
            "N": N,
            "dark_share": dark_share,
            "malignant_frac": mal_frac,
        })

    with open(SCENARIOS_PATH, "w") as f:
        json.dump(scenarios, f, indent=2)
    log(f"Wrote {len(scenarios)} scenarios to {SCENARIOS_PATH}")


# =============================================================================
# Stage 4: Run a single scenario
# =============================================================================
def cmd_run_scenario(args) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    import torchvision.models as tvm
    from PIL import Image
    from sklearn.metrics import roc_auc_score, recall_score
    from sklearn.neighbors import NearestNeighbors

    ensure_dirs(SCENARIO_RESULTS_DIR)
    out_path = Path(SCENARIO_RESULTS_DIR) / f"scenario_{args.id:03d}.json"
    if out_path.exists() and not args.force:
        log(f"Scenario {args.id} already done, skipping (use --force to redo)")
        return

    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)
    spec = scenarios[args.id]
    log(f"Running scenario {args.id}: {spec}")

    df = load_metadata()
    feat = np.load(FEATURES_PATH, allow_pickle=True)
    have_md5 = set(feat["md5"].tolist())
    df = df[df["md5hash"].isin(have_md5)].reset_index(drop=True)

    rng = np.random.default_rng(spec["seed"])
    N = spec["N"]
    n_dark = int(N * spec["dark_share"])
    n_light = N - n_dark
    n_dark_mal = int(n_dark * spec["malignant_frac"])
    n_light_mal = int(n_light * spec["malignant_frac"])

    def pick(group, malignant, n):
        pool = df[(df["fst_group"] == group) & (df["malignant"] == malignant)]
        if len(pool) == 0:
            return pool
        take = min(len(pool), n)
        if take < n * 0.5:
            log(f"  [warn] requested {n} {group}/mal={malignant}, only {take} available")
        return pool.sample(n=take, random_state=int(rng.integers(1e9)))

    sample = pd.concat([
        pick("V-VI", 1, n_dark_mal),
        pick("V-VI", 0, n_dark - n_dark_mal),
        pick("I-II", 1, n_light_mal),
        pick("I-II", 0, n_light - n_light_mal),
    ]).reset_index(drop=True)
    log(f"Sampled {len(sample)} images")

    # Stratified train/probe/test splits
    sample = sample.sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    test_frac = 0.20
    probe_target = 300

    splits = {"train": [], "probe": [], "test": []}
    for g in ["I-II", "V-VI"]:
        sub = sample[sample["fst_group"] == g].reset_index(drop=True)
        n_test = max(20, int(len(sub) * test_frac)) if len(sub) >= 50 else len(sub) // 3
        n_probe = max(10, int(probe_target * (len(sub) / max(len(sample), 1))))
        n_probe = min(n_probe, max(0, len(sub) - n_test - 30))  # leave training data
        splits["test"].append(sub.iloc[:n_test])
        splits["probe"].append(sub.iloc[n_test:n_test + n_probe])
        splits["train"].append(sub.iloc[n_test + n_probe:])

    train_df = pd.concat(splits["train"]).reset_index(drop=True)
    probe_df = pd.concat(splits["probe"]).reset_index(drop=True)
    test_df = pd.concat(splits["test"]).reset_index(drop=True)
    log(f"Split: train={len(train_df)}  probe={len(probe_df)}  test={len(test_df)}")

    if len(train_df) < 100:
        log(f"[skip] training set too small ({len(train_df)}); skipping scenario")
        # Still write a result record so aggregator can see why
        with open(out_path, "w") as f:
            json.dump({"scenario": spec, "status": "skipped_small_train",
                       "n_train": len(train_df)}, f, indent=2)
        return

    # ---- Train EfficientNet-B0 ---------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class ImgDS(Dataset):
        def __init__(self, frame, train=False):
            self.frame = frame.reset_index(drop=True)
            self.train = train
            base = [T.Resize(256)]
            if train:
                base += [T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                         T.RandomHorizontalFlip(),
                         T.ColorJitter(0.2, 0.2, 0.2)]
            else:
                base += [T.CenterCrop(IMG_SIZE)]
            base += [T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
            self.tfm = T.Compose(base)

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, i):
            row = self.frame.iloc[i]
            img = Image.open(Path(IMAGES_DIR) / f"{row['md5hash']}.jpg").convert("RGB")
            return self.tfm(img), int(row["malignant"])

    torch.manual_seed(spec["seed"])
    try:
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = tvm.efficientnet_b0(weights=weights)
    except AttributeError:
        model = tvm.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(device)

    loader = DataLoader(ImgDS(train_df, train=True),
                        batch_size=32, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 15
    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        n_seen = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        log(f"  epoch {epoch + 1}/{EPOCHS}  loss={ep_loss / max(n_seen, 1):.4f}")

    # ---- Evaluate per FST group --------------------------------------------
    def predict(frame):
        model.eval()
        ds = ImgDS(frame, train=False)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        probs, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
                probs.append(p); ys.append(y.numpy())
        return np.concatenate(probs), np.concatenate(ys)

    metrics = {}
    for g in ["I-II", "V-VI"]:
        sub = test_df[test_df["fst_group"] == g]
        if len(sub) < 10 or sub["malignant"].nunique() < 2:
            metrics[g] = {"auc": float("nan"), "sens": float("nan"), "n": int(len(sub))}
            continue
        p, y = predict(sub)
        metrics[g] = {
            "auc": float(roc_auc_score(y, p)),
            "sens": float(recall_score(y, (p > 0.5).astype(int), zero_division=0)),
            "n": int(len(sub)),
        }
    delta_auc = metrics["I-II"]["auc"] - metrics["V-VI"]["auc"]
    delta_sens = metrics["I-II"]["sens"] - metrics["V-VI"]["sens"]
    log(f"  Test AUC: I-II={metrics['I-II']['auc']:.3f}  "
        f"V-VI={metrics['V-VI']['auc']:.3f}  ΔAUC={delta_auc:.3f}")

    # ---- Signals -----------------------------------------------------------
    tr_light = train_df[train_df["fst_group"] == "I-II"]
    tr_dark = train_df[train_df["fst_group"] == "V-VI"]

    phi1 = len(tr_light) / max(len(tr_dark), 1)
    p_mal_dark = tr_dark["malignant"].mean() if len(tr_dark) else 0.0
    p_mal_light = tr_light["malignant"].mean() if len(tr_light) else 1e-9
    phi2 = p_mal_dark / max(p_mal_light, 1e-9)

    joint = pd.crosstab(train_df["fst"], train_df["nine_partition_label"])
    expected_cells = 6 * max(train_df["nine_partition_label"].nunique(), 1)
    phi3 = float((joint == 0).sum().sum()) / max(expected_cells, 1)

    md5_to_idx = {m: i for i, m in enumerate(feat["md5"].tolist())}
    light_idx = [md5_to_idx[m] for m in tr_light["md5hash"] if m in md5_to_idx]
    dark_idx = [md5_to_idx[m] for m in tr_dark["md5hash"] if m in md5_to_idx]

    if len(light_idx) > 5 and len(dark_idx) > 0:
        light_feats = feat["features"][light_idx]
        dark_feats = feat["features"][dark_idx]
        light_n = light_feats / (np.linalg.norm(light_feats, axis=1, keepdims=True) + 1e-9)
        dark_n = dark_feats / (np.linalg.norm(dark_feats, axis=1, keepdims=True) + 1e-9)
        nn_model = NearestNeighbors(n_neighbors=5, metric="cosine").fit(light_n)
        dists, _ = nn_model.kneighbors(dark_n)
        phi4 = float(dists.mean())
    else:
        phi4 = float("nan")

    if len(light_idx) > 0 and len(dark_idx) > 0:
        h_light = feat["lab_hists"][light_idx].mean(axis=0)
        h_dark = feat["lab_hists"][dark_idx].mean(axis=0)
        eps = 1e-9
        h_light = h_light + eps; h_light /= h_light.sum()
        h_dark = h_dark + eps; h_dark /= h_dark.sum()
        phi5 = float(np.sum(h_dark * np.log(h_dark / h_light)))
    else:
        phi5 = float("nan")

    probe_acc = {}
    for g in ["I-II", "V-VI"]:
        sub = probe_df[probe_df["fst_group"] == g]
        if len(sub) < 10:
            probe_acc[g] = float("nan")
            continue
        p, y = predict(sub)
        probe_acc[g] = float(((p > 0.5).astype(int) == y).mean())
    phi6 = probe_acc["I-II"] - probe_acc["V-VI"]

    result = {
        "scenario": spec,
        "status": "ok",
        "n_train": int(len(train_df)),
        "n_probe": int(len(probe_df)),
        "n_test": int(len(test_df)),
        "metrics_test": metrics,
        "delta_auc": float(delta_auc),
        "delta_sens": float(delta_sens),
        "signals": {
            "phi1_imbalance_ratio": float(phi1),
            "phi2_class_cond_ratio": float(phi2),
            "phi3_empty_cells_frac": float(phi3),
            "phi4_embedding_dist": float(phi4),
            "phi5_lab_kl": float(phi5),
            "phi6_probe_acc_gap": float(phi6),
        },
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Wrote {out_path}")


# =============================================================================
# Stage 5: Aggregate
# =============================================================================
def cmd_aggregate(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error, r2_score

    ensure_dirs(AGGREGATE_DIR)

    rows = []
    skipped = 0
    for p in sorted(Path(SCENARIO_RESULTS_DIR).glob("scenario_*.json")):
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
        sys.exit("[error] No usable scenario results.")
    log(f"Aggregating {len(rows)} scenarios (skipped {skipped} with status != ok)")

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    df.to_csv(Path(AGGREGATE_DIR) / "scenario_results.csv", index=False)

    signal_cols = [c for c in df.columns if c.startswith("phi")]
    corr_rows = []
    for col in signal_cols:
        x = df[col].values
        for target in ["delta_auc", "delta_sens"]:
            y = df[target].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                rho, pval = float("nan"), float("nan")
            else:
                rho, pval = spearmanr(x[mask], y[mask])
            corr_rows.append({"signal": col, "target": target,
                              "spearman_rho": float(rho),
                              "p_value": float(pval),
                              "n": int(mask.sum())})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(Path(AGGREGATE_DIR) / "univariate_correlations.csv", index=False)
    log("Univariate Spearman correlations:")
    print(corr_df.to_string(index=False))

    # Multivariate: linear regression with 5-fold CV
    X_cols = ["phi1_imbalance_ratio", "phi2_class_cond_ratio",
              "phi3_empty_cells_frac", "phi4_embedding_dist", "phi5_lab_kl"]
    X = df[X_cols].values
    y = df["delta_auc"].values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]

    forecast_summary: dict = {"n_scenarios": int(len(y))}
    preds = None
    if len(X) >= 10:
        kf = KFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
        preds = np.zeros_like(y)
        for tr, te in kf.split(X):
            reg = LinearRegression().fit(X[tr], y[tr])
            preds[te] = reg.predict(X[te])
        forecast_summary.update({
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
            "baseline_mae_predict_mean": float(np.mean(np.abs(y - y.mean()))),
        })
        log(f"5-fold CV forecast: MAE={forecast_summary['mae']:.4f}  "
            f"R²={forecast_summary['r2']:.3f}  "
            f"(baseline MAE={forecast_summary['baseline_mae_predict_mean']:.4f})")
    else:
        forecast_summary["note"] = "n<10, skipped CV regression"
        log("[warn] Fewer than 10 valid scenarios; skipping multivariate regression")

    with open(Path(AGGREGATE_DIR) / "forecast_summary.json", "w") as f:
        json.dump(forecast_summary, f, indent=2)

    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, signal_cols):
        ax.scatter(df[col], df["delta_auc"], alpha=0.7)
        ax.set_xlabel(col); ax.set_ylabel("ΔAUC (I-II minus V-VI)")
        sub = corr_df[(corr_df["signal"] == col) & (corr_df["target"] == "delta_auc")]
        rho = sub["spearman_rho"].iloc[0] if len(sub) else float("nan")
        ax.set_title(f"{col}\nSpearman ρ={rho:.2f}")
        ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(Path(AGGREGATE_DIR) / "signals_vs_delta_auc.png", dpi=130)
    plt.close(fig)

    if preds is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y, preds, alpha=0.7)
        lo = min(y.min(), preds.min()) - 0.02
        hi = max(y.max(), preds.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        ax.set_xlabel("Observed ΔAUC")
        ax.set_ylabel("Forecast ΔAUC (5-fold CV)")
        ax.set_title(f"Multivariate forecast  MAE={forecast_summary['mae']:.3f}  "
                     f"R²={forecast_summary['r2']:.2f}")
        fig.tight_layout()
        fig.savefig(Path(AGGREGATE_DIR) / "forecast_vs_observed.png", dpi=130)
        plt.close(fig)

    log(f"Aggregation outputs in {AGGREGATE_DIR}")


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download_images")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--timeout", type=int, default=30)
    d.add_argument("--no-retry", action="store_true")

    sub.add_parser("extract_features")

    g = sub.add_parser("generate_scenarios")
    g.add_argument("--k", type=int, default=K_PILOT)

    r = sub.add_parser("run_scenario")
    r.add_argument("--id", type=int, required=True)
    r.add_argument("--force", action="store_true")

    sub.add_parser("aggregate")

    args = parser.parse_args()
    {
        "download_images": cmd_download_images,
        "extract_features": cmd_extract_features,
        "generate_scenarios": cmd_generate_scenarios,
        "run_scenario": cmd_run_scenario,
        "aggregate": cmd_aggregate,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
