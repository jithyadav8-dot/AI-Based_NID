import json
import joblib
import numpy as np
import pandas as pd

model  = joblib.load("models/xgboost_nids.pkl")
scaler = joblib.load("models/scaler.pkl")
le     = joblib.load("models/label_encoder.pkl")

with open("processed/feature_list.json") as f:
    feature_cols = json.load(f)

test = pd.read_parquet("processed/test.parquet")

# ── Test both versions ────────────────────────────────────────────
X_raw    = test[feature_cols].values.astype("float32")
X_scaled = scaler.transform(X_raw)

print("=== Raw features ===")
print(f"  mean={X_raw.mean():.2f}  std={X_raw.std():.2f}")
pred_raw = model.predict(X_raw)
counts_raw = pd.Series(le.inverse_transform(pred_raw)).value_counts()
print(f"  Prediction distribution:\n{counts_raw.to_string()}")

print("\n=== Scaled features ===")
print(f"  mean={X_scaled.mean():.4f}  std={X_scaled.std():.4f}")
pred_scaled = model.predict(X_scaled)
counts_scaled = pd.Series(le.inverse_transform(pred_scaled)).value_counts()
print(f"  Prediction distribution:\n{counts_scaled.to_string()}")

print("\n=== Confirmation ===")
inf_raw    = (le.inverse_transform(pred_raw) == "Infiltration").sum()
inf_scaled = (le.inverse_transform(pred_scaled) == "Infiltration").sum()
print(f"  Infiltration predictions (raw)    : {inf_raw:,}")
print(f"  Infiltration predictions (scaled) : {inf_scaled:,}")
print(f"\n  The version with Infiltration > 0 is what the script must use.")
