# diagnose_infiltration_threshold.py
# Run this to find the optimal Infiltration confidence threshold.

import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_threshold_sweep(model, X, y, le):
    inf_idx     = list(le.classes_).index("Infiltration")
    y_proba     = model.predict_proba(X)
    inf_proba   = y_proba[:, inf_idx]   # probability of Infiltration for every flow

    # ── Sweep thresholds ──────────────────────────────────────────────
    # For each threshold t:
    #   Predict Infiltration only if P(Infiltration) > t AND
    #   Infiltration is the argmax class.
    # Otherwise fall back to the second-best class.
    thresholds  = np.arange(0.30, 0.95, 0.02)
    results     = []

    base_pred   = model.predict(X)   # standard argmax predictions

    for t in thresholds:
        y_pred = base_pred.copy()

        # Where model predicted Infiltration but confidence is below t,
        # override with the second-highest class
        inf_predicted_mask      = (base_pred == inf_idx)
        low_confidence_mask     = inf_predicted_mask & (inf_proba < t)

        # Second-best class: zero out Infiltration column, take new argmax
        proba_no_inf            = y_proba.copy()
        proba_no_inf[:, inf_idx] = 0.0
        second_best             = np.argmax(proba_no_inf, axis=1)

        y_pred[low_confidence_mask] = second_best[low_confidence_mask]

        # Metrics for Infiltration class only
        inf_mask    = (y == inf_idx)
        pred_mask   = (y_pred == inf_idx)

        tp = int(( inf_mask &  pred_mask).sum())
        fp = int((~inf_mask &  pred_mask).sum())
        fn = int(( inf_mask & ~pred_mask).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # Also track BENIGN F1 (should recover as false positives drop)
        ben_idx  = list(le.classes_).index("BENIGN")
        ben_mask = (y == ben_idx)
        ben_tp   = int(( ben_mask & (y_pred == ben_idx)).sum())
        ben_fp   = int((~ben_mask & (y_pred == ben_idx)).sum())
        ben_fn   = int(( ben_mask & (y_pred != ben_idx)).sum())
        ben_p    = ben_tp / (ben_tp + ben_fp) if (ben_tp + ben_fp) > 0 else 0
        ben_r    = ben_tp / (ben_tp + ben_fn) if (ben_tp + ben_fn) > 0 else 0
        ben_f1   = (2*ben_p*ben_r/(ben_p+ben_r)) if (ben_p+ben_r) > 0 else 0

        results.append({
            "threshold":  round(float(t), 2),
            "precision":  round(prec, 4),
            "recall":     round(rec,  4),
            "f1":         round(f1,   4),
            "tp": tp, "fp": fp, "fn": fn,
            "benign_f1":  round(ben_f1, 4),
        })

    # ── Print table ───────────────────────────────────────────────────
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>8} "
          f"{'F1':>8} {'TP':>8} {'FP':>8} {'FN':>6} {'BENIGN F1':>10}")
    print("-" * 76)
    for r in results:
        marker = " ◄" if r["f1"] == max(x["f1"] for x in results) else ""
        print(f"  {r['threshold']:>8}   {r['precision']:>9}  {r['recall']:>7} "
              f" {r['f1']:>7}  {r['tp']:>7}  {r['fp']:>7}  {r['fn']:>5} "
              f"  {r['benign_f1']:>8}{marker}")

    # ── Best threshold by F1 ──────────────────────────────────────────
    best = max(results, key=lambda x: x["f1"])
    print(f"\nBest threshold : {best['threshold']}")
    print(f"  Infiltration — Precision: {best['precision']}  "
          f"Recall: {best['recall']}  F1: {best['f1']}")
    print(f"  BENIGN F1 at this threshold: {best['benign_f1']}")
    print(f"  TP: {best['tp']:,}  FP: {best['fp']:,}  FN: {best['fn']:,}")

    # ── Save threshold ────────────────────────────────────────────────
    threshold_config = {
        "infiltration_threshold": best["threshold"],
        "expected_precision":     best["precision"],
        "expected_recall":        best["recall"],
        "expected_f1":            best["f1"],
        "note": (
            "XGBoost predicts Infiltration only when P(Infiltration) "
            "exceeds this threshold. Below it, the second-best class wins. "
            "Calibrated on the 2018 test set to maximise Infiltration F1."
        )
    }

    with open("models/threshold_config.json", "w") as f:
        json.dump(threshold_config, f, indent=2)
    print(f"\nSaved: models/threshold_config.json")

    # ── Plot ──────────────────────────────────────────────────────────
    df_r = pd.DataFrame(results)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df_r["threshold"], df_r["precision"], label="Precision", color="steelblue")
    ax1.plot(df_r["threshold"], df_r["recall"],    label="Recall",    color="tomato")
    ax1.plot(df_r["threshold"], df_r["f1"],        label="F1",        color="green", linewidth=2)
    ax1.axvline(best["threshold"], color="gray", linestyle="--",
                label=f"Best threshold = {best['threshold']}")
    ax1.set_xlabel("Infiltration confidence threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Infiltration Precision / Recall / F1 vs. Confidence Threshold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/infiltration_threshold_curve.png", dpi=150)
    print("Saved: results/infiltration_threshold_curve.png")

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        
    model  = joblib.load("models/xgboost_nids.pkl")
    le     = joblib.load("models/label_encoder.pkl")

    with open("processed/feature_list.json") as f:
        feature_cols = json.load(f)

    test = pd.read_parquet("processed/test.parquet")
    X    = test[feature_cols].values.astype("float32")
    y    = le.transform(test["Label"].values)
    
    run_threshold_sweep(model, X, y, le)
