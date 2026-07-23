import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from data.column_map import apply_2017_column_map
from data.label_maps import LABEL_MAP_2017

MODEL_DIR     = Path("models")
PROCESSED_DIR = Path("processed")

model  = joblib.load(MODEL_DIR / "xgboost_nids.pkl")
le     = joblib.load(MODEL_DIR / "label_encoder.pkl")

with open(PROCESSED_DIR / "feature_list.json") as f:
    feature_cols = json.load(f)

# Load one attack-heavy 2017 file for diagnosis
import glob
csv_files = sorted(glob.glob("raw/cic2017/**/*.csv", recursive=True))

# Load all files and filter to attack traffic
dfs = []
for f in csv_files:
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

# ── 1. What does XGBoost predict for each 2017 attack class? ─────
print("=== XGBoost prediction distribution on 2017 attack flows ===\n")
for label in sorted(df["Label"].unique()):
    if label == "BENIGN":
        continue
    subset = df[df["Label"] == label]
    if len(subset) > 10000:
        subset = subset.sample(10000, random_state=42)

    X     = subset[feature_cols].values.astype(np.float32)
    preds = le.inverse_transform(model.predict(X))
    dist  = pd.Series(preds).value_counts(normalize=True) * 100

    print(f"{label} (n={len(subset):,}):")
    for cls, pct in dist.items():
        bar = "#" * int(pct / 2)
        print(f"  Predicted as {cls:<15} {pct:>6.1f}%  {bar}")
    print()

# ── 2. Top feature importance vs distribution shift ───────────────
print("=== Top XGBoost features vs 2017/2018 distribution shift ===\n")

train_2018 = pd.read_parquet(PROCESSED_DIR / "train.parquet")

importances = model.feature_importances_
top_idx     = np.argsort(importances)[::-1][:15]

print(f"  {'Feature':<40} {'Importance':>10} "
      f"{'2018 mean':>12} {'2017 mean':>12} {'Ratio':>8}")
print("  " + "-" * 86)

for idx in top_idx:
    feat      = feature_cols[idx]
    imp       = importances[idx]
    mean_2018 = float(train_2018[feat].mean()) if feat in train_2018 else 0
    mean_2017 = float(df[feat].mean())
    ratio     = mean_2017 / mean_2018 if mean_2018 != 0 else float("inf")
    flag      = " <-- SHIFT" if abs(ratio - 1) > 0.5 else ""
    print(f"  {feat:<40} {imp:>10.4f} "
          f"{mean_2018:>12.2f} {mean_2017:>12.2f} "
          f"{ratio:>8.2f}{flag}")
