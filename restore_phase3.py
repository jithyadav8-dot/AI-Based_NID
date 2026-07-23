import json
import torch
import joblib
import numpy as np
import pandas as pd
from training.autoencoder import Autoencoder

with open("processed/feature_list.json") as f:
    feature_cols = json.load(f)

with open("models/threshold_ae.json") as f:
    cfg = json.load(f)

# Reload with bottleneck=8 (the default)
model = Autoencoder(input_dim=len(feature_cols), bottleneck_dim=8)
model.load_state_dict(
    torch.load("models/autoencoder.pt", map_location="cpu")
)
model.eval()

print(f"Bottleneck dim : 8")
print(f"Threshold      : {cfg['threshold']:.6f}")
print(f"BENIGN FA rate : {cfg['benign_false_alarm_rate']*100:.1f}%")
print("Restored to bottleneck=8 successfully.")
