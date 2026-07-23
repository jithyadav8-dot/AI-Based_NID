import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from training.xgboost_trainer import run

if __name__ == "__main__":
    # Ensure Windows console doesn't crash on unicode characters like █ and ×
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        
    run()

    model  = joblib.load("models/xgboost_nids.pkl")
    le     = joblib.load("models/label_encoder.pkl")

    with open("processed/feature_list.json") as f:
        feature_cols = json.load(f)

    test = pd.read_parquet("processed/test.parquet")
    X    = test[feature_cols].values.astype("float32")
    y    = le.transform(test["Label"].values)

    y_pred = model.predict(X)

    # ── 1. Infiltration precision and recall separately ───────────────
    inf_idx = list(le.classes_).index("Infiltration")
    print("\n=== Infiltration breakdown ===")
    inf_mask   = (y == inf_idx)
    pred_mask  = (y_pred == inf_idx)

    tp = int(( inf_mask &  pred_mask).sum())
    fp = int((~inf_mask &  pred_mask).sum())
    fn = int(( inf_mask & ~pred_mask).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"  True Positives  (correctly caught Infiltration) : {tp:,}")
    print(f"  False Positives (benign/other flagged as Infil) : {fp:,}")
    print(f"  False Negatives (Infiltration missed)           : {fn:,}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")

    # ── 2. Where are Infiltration flows actually going? ───────────────
    print("\n=== Where actual Infiltration flows are predicted as ===")
    inf_actual       = y_pred[inf_mask]
    pred_counts      = pd.Series(inf_actual).value_counts()
    for cls_idx, count in pred_counts.items():
        pct = count / inf_mask.sum() * 100
        print(f"  Predicted as {le.classes_[cls_idx]:<15}: {count:,}  ({pct:.1f}%)")

    # ── 3. What is being wrongly predicted as Infiltration? ───────────
    print("\n=== What flows are being falsely flagged as Infiltration ===")
    fp_actual        = y[pred_mask & ~inf_mask]
    fp_counts        = pd.Series(fp_actual).value_counts()
    for cls_idx, count in fp_counts.items():
        pct = count / pred_mask.sum() * 100
        print(f"  Actually {le.classes_[cls_idx]:<15}: {count:,}  ({pct:.1f}%)")

    # ── 4. Feature overlap: Infiltration vs BENIGN ────────────────────
    print("\n=== Mean feature values: Infiltration vs BENIGN ===")
    ben_idx = list(le.classes_).index("BENIGN")
    df_feat = pd.DataFrame(X, columns=feature_cols)
    df_feat["Label"] = le.inverse_transform(y)

    inf_means = df_feat[df_feat["Label"] == "Infiltration"][feature_cols].mean()
    ben_means = df_feat[df_feat["Label"] == "BENIGN"][feature_cols].mean()

    diff = (inf_means - ben_means).abs().sort_values(ascending=False)
    print("  Top 10 features where Infiltration ≠ BENIGN:")
    for feat, delta in diff.head(10).items():
        print(f"    {feat:<40} delta={delta:.4f}")
    print("  Bottom 5 features (almost identical to BENIGN):")
    for feat, delta in diff.tail(5).items():
        print(f"    {feat:<40} delta={delta:.4f}")
