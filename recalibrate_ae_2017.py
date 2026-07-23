import json
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from data.column_map import apply_2017_column_map
from data.label_maps import LABEL_MAP_2017
from training.autoencoder import Autoencoder

MODEL_DIR     = Path("models")
PROCESSED_DIR = Path("processed")

with open(PROCESSED_DIR / "feature_list.json") as f:
    feature_cols = json.load(f)

with open(MODEL_DIR / "threshold_ae.json") as f:
    ae_cfg = json.load(f)

scaler = joblib.load(MODEL_DIR / "scaler.pkl")
model  = Autoencoder(input_dim=len(feature_cols), bottleneck_dim=8)
model.load_state_dict(
    torch.load(MODEL_DIR / "autoencoder.pt", map_location="cpu")
)
model.eval()

original_threshold = ae_cfg["threshold"]

# ── Load 2017 data ────────────────────────────────────────────────
import glob
dfs = []
for f in sorted(glob.glob("raw/cic2017/**/*.csv", recursive=True)):
    df = pd.read_csv(f, low_memory=False, encoding_errors="replace")
    df = apply_2017_column_map(df)
    df["Label"] = df["Label"].astype(str).str.strip().map(LABEL_MAP_2017)
    df = df.dropna(subset=["Label"])
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = df.replace([np.inf, -np.inf], np.nan)
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0
df = df.dropna(subset=feature_cols)

# ── Compute 2017-BENIGN reconstruction errors ─────────────────────
benign_2017 = df[df["Label"] == "BENIGN"]
X_ben       = scaler.transform(
    benign_2017[feature_cols].values.astype(np.float32)
)
tensor_ben  = torch.tensor(X_ben, dtype=torch.float32)
ben_errors  = model.reconstruction_error(tensor_ben)

new_threshold = float(np.percentile(ben_errors, 95))

print(f"Original threshold (2018 BENIGN 95th pct) : {original_threshold:.6f}")
print(f"New threshold      (2017 BENIGN 95th pct) : {new_threshold:.6f}")
# Cap outlier errors for reporting (doesn't affect threshold or detection rates)
ben_errors_capped = np.clip(ben_errors, 0, np.percentile(ben_errors, 99.9))
print(f"2017 BENIGN error (capped 99.9th pct):")
print(f"  mean = {ben_errors_capped.mean():.4f}")
print(f"  std  = {ben_errors_capped.std():.4f}")
print(f"  p95  = {np.percentile(ben_errors_capped, 95):.4f}")
print(f"  Raw outlier flows (>p99.9): "
      f"{(ben_errors > np.percentile(ben_errors, 99.9)).sum():,}")

# ── Re-evaluate all 2017 classes with 2017-calibrated threshold ───
print(f"\n=== AE detection rates with 2017-calibrated threshold ===\n")
print(f"  {'Class':<15} {'Mean Err':>10} {'Detect%(orig)':>14} "
      f"{'Detect%(2017cal)':>16}")
print("  " + "-" * 60)

for label in sorted(df["Label"].unique()):
    subset = df[df["Label"] == label]
    if len(subset) > 50000:
        subset = subset.sample(50000, random_state=42)

    X      = scaler.transform(
        subset[feature_cols].values.astype(np.float32)
    )
    tensor = torch.tensor(X, dtype=torch.float32)
    errors = model.reconstruction_error(tensor)

    det_orig = float((errors > original_threshold).mean() * 100)
    det_new  = float((errors > new_threshold).mean() * 100)

    bar = "#" * int(det_new / 2.5)
    print(f"  {label:<15} {errors.mean():>10.6f} "
          f"{det_orig:>13.1f}%  {det_new:>15.1f}%  {bar}")

# Save 2017-calibrated threshold separately — does NOT overwrite
# the deployment threshold (which is set per environment)
result = {
    "threshold_2018_calibrated": original_threshold,
    "threshold_2017_calibrated": new_threshold,
    "finding": (
        "AE threshold requires calibration on local BENIGN traffic. "
        "2017 BENIGN has 70.9% false alarm with 2018 threshold. "
        "Re-calibrated on 2017 BENIGN shows true detection capability. "
        "In production: collect 1-2hrs local benign traffic, "
        "recompute threshold at 95th percentile."
    ),
}
import os
os.makedirs("results", exist_ok=True)
with open("results/ae_threshold_recalibration.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: results/ae_threshold_recalibration.json")
