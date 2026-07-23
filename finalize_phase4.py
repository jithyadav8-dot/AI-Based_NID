import json

with open("results/phase4_report.json", encoding="utf-8") as f:
    report = json.load(f)

report["ae_recalibration"] = {
    "threshold_2018": 0.323290,
    "threshold_2017_calibrated": 1.752906,
    "benign_false_alarm_rate": "5.0%",
    "detection_at_2017_threshold": {
        "DDoS":         "63.5%  <- compensates for XGBoost tool gap",
        "DoS":          "66.9%  <- strong combined with XGBoost 0.42",
        "Infiltration": "55.6%  <- AE primary detector (XGBoost 0.00)",
        "BruteForce":   "0.0%   <- covered by XGBoost 0.54",
        "Botnet":       "2.4%   <- covered by XGBoost 0.58",
        "PortScan":     "0.3%   <- coverage gap in both models",
    },
    "benign_outlier_note": (
        "2017 BENIGN mean reconstruction error=70M driven by <0.1% "
        "of flows with extreme feature values amplified by 2018 scaler. "
        "95th percentile (1.75) is the operative threshold value. "
        "5% false alarm rate confirms threshold calibration is correct."
    ),
    "production_deployment_pattern": (
        "Collect 1-2hrs local benign traffic. Compute reconstruction "
        "error on all flows. Set threshold at 95th percentile. "
        "Recompute monthly or after significant network topology changes."
    ),
}

report["combined_system_2017"] = {
    "summary": (
        "Hybrid system provides meaningful coverage across 2017 dataset "
        "despite being trained only on 2018. XGBoost handles Botnet (0.58) "
        "and BruteForce (0.54). AE compensates for XGBoost's DDoS tool gap "
        "with 63.5% detection. Infiltration covered by AE at 55.6% (only "
        "36 samples in 2017, insufficient for XGBoost evaluation). "
        "PortScan is the primary coverage gap - absent from 2018 training "
        "and low AE reconstruction error makes it indistinguishable from "
        "benign at the current threshold."
    ),
    "coverage_gap": "PortScan - requires 2018 dataset augmentation or dedicated rule",
}

with open("results/phase4_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("Phase 4 finalised.")
print("\n=== Complete Phase Summary ===")
print("\nPhase 1 [OK]  43 features, 3.69M rows, CSE-CIC-IDS2018")
print("Phase 2 [OK]  XGBoost raw features, Macro F1=0.88 (2018)")
print("              Infiltration threshold=0.86, BENIGN F1=0.9835")
print("Phase 3 [OK]  Autoencoder 43->32->16->8->16->32->43")
print("              BN=8, threshold=0.323, WebAttack 42.6%, Infiltration 12.1%")
print("Phase 4 [OK]  Cross-dataset validation on CICIDS2017")
print("              XGBoost Macro F1=0.41 (no retraining)")
print("              AE DDoS 63.5%, DoS 66.9%, Infiltration 55.6%")
print("              Key finding: Src Port absent from 2017, DDoS tool gap")
print("              Production pattern: recalibrate AE threshold locally")
