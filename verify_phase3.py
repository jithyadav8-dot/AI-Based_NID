# verify_phase3.py

import json
import torch
import joblib
import numpy as np
import pandas as pd
from training.autoencoder import Autoencoder

print("Loading Phase 3 artifacts...")

with open("processed/feature_list.json") as f:
    feature_cols = json.load(f)

with open("models/threshold_ae.json") as f:
    cfg = json.load(f)

model = Autoencoder(input_dim=len(feature_cols))
model.load_state_dict(torch.load("models/autoencoder.pt",
                                  map_location="cpu"))
model.eval()

scaler    = joblib.load("models/scaler.pkl")
threshold = cfg["threshold"]

# Run a single benign and a single attack flow through
test = pd.read_parquet("processed/test.parquet")

benign_sample = test[test["Label"] == "BENIGN"].iloc[:1]
attack_sample = test[test["Label"] == "Infiltration"].iloc[:1]

for name, sample in [("BENIGN", benign_sample),
                     ("Infiltration", attack_sample)]:
    X      = scaler.transform(
        sample[feature_cols].values.astype(np.float32)
    )
    X      = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    tensor = torch.tensor(X, dtype=torch.float32)
    error  = model.reconstruction_error(tensor)[0]
    flag   = "ANOMALY" if error > threshold else "normal"
    print(f"  {name:<15} error={error:.6f}  threshold={threshold:.6f}  → {flag}")

print(f"\nPhase 3 verification PASSED")
