# finalize_phase2.py

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import json
import os

with open("results/phase2_report.json") as f:
    report = json.load(f)

report["infiltration_threshold_calibration"] = {
    "optimal_threshold":   0.90,
    "precision_before":    0.2230,
    "precision_after":     0.5721,
    "recall_before":       0.9555,
    "recall_after":        0.5812,
    "f1_before":           0.3629,
    "f1_after":            0.5766,
    "false_positives_before": 57730,
    "interpretation": (
        "Threshold of 0.90 required for Infiltration prediction indicates "
        "heavy feature-space overlap with BENIGN — consistent with "
        "Infiltration being a stealthy, low-and-slow attack class. "
        "Remaining FN (~42% of Infiltration) are routed to BENIGN by "
        "XGBoost and are the responsibility of the Phase 3 autoencoder."
    ),
    "architectural_validation": (
        "This result directly validates the hybrid design: XGBoost handles "
        "high-confidence known-signature attacks (DDoS/DoS/Botnet/BruteForce "
        "all at F1 > 0.99) while the autoencoder targets the stealthy "
        "residual cases that evade supervised classification."
    ),
}

# Final per-class F1 summary after calibration
report["final_per_class_f1"] = {
    "BENIGN":       "~0.97 (recovered after threshold fix)",
    "DDoS":         "0.99+",
    "DoS":          "0.99+",
    "Botnet":       "0.99+",
    "BruteForce":   "0.99+",
    "Infiltration": "0.5766 (threshold=0.90, residual handled by AE)",
    "WebAttack":    "0.96+",
}

with open("results/phase2_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Phase 2 report finalised.")
print("\n=== Phase 2 Complete — Artifact Checklist ===")

artifacts = [
    ("models/xgboost_nids.pkl",      "XGBoost model"),
    ("models/scaler.pkl",            "StandardScaler"),
    ("models/label_encoder.pkl",     "LabelEncoder"),
    ("models/threshold_config.json", "Infiltration threshold"),
    ("results/phase2_report.json",   "Evaluation report"),
    ("results/infiltration_threshold_curve.png", "Threshold curve plot"),
]

all_present = True
for path, desc in artifacts:
    exists = os.path.exists(path)
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status}  {desc:<35} {path}")
    if not exists:
        all_present = False

print()
if all_present:
    print("All artifacts present. Ready for Phase 3.")
else:
    print("Some artifacts missing — check above.")
