import json
import joblib
import numpy as np
import pandas as pd
from data.column_map import apply_2017_column_map
from data.label_maps import LABEL_MAP_2017

with open("processed/feature_list.json") as f:
    feature_cols = json.load(f)

scaler = joblib.load("models/scaler.pkl")

# Load one 2017 CSV
import glob
f = glob.glob("raw/cic2017/MachineLearningCVE/*.csv")[0]
df = pd.read_csv(f, low_memory=False, encoding_errors="replace")
df.columns = df.columns.str.strip()
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

df.replace([np.inf, -np.inf], np.nan, inplace=True)
X_raw    = df[feature_cols].dropna().values[:10000].astype(np.float32)
X_scaled = scaler.transform(X_raw)

print("=== Feature scale diagnostics ===")
print(f"Raw    - mean: {X_raw.mean():.2f}    std: {X_raw.std():.2f}")
print(f"Scaled - mean: {X_scaled.mean():.4f}  std: {X_scaled.std():.4f}")
print(f"\nExpected after in-distribution scaling: mean~0, std~1")
print(f"If scaled mean/std are far from 0/1, the scaler is the problem.")

# Show worst-shifted features
means  = X_scaled.mean(axis=0)
stds   = X_scaled.std(axis=0)
shifts = np.abs(means)
worst  = np.argsort(shifts)[::-1][:10]
print(f"\nTop 10 most shifted features (should be ~0 for in-distribution):")
for i in worst:
    print(f"  {feature_cols[i]:<40} mean={means[i]:+.2f}  std={stds[i]:.2f}")
