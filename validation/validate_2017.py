# validation/validate_2017.py

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score

from data.column_map import apply_2017_column_map
from data.label_maps import LABEL_MAP_2017
from training.autoencoder import Autoencoder

# ── Paths ─────────────────────────────────────────────────────────
RAW_2017_DIR  = Path("raw/cic2017")
MODEL_DIR     = Path("models")
RESULTS_DIR   = Path("results")
PROCESSED_DIR = Path("processed")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Step 1: Load CICIDS2017 ───────────────────────────────────────

def load_2017(feature_cols: list) -> pd.DataFrame:
    print("[1/5] Loading CICIDS2017...")

    csv_files = sorted(RAW_2017_DIR.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {RAW_2017_DIR}")

    print(f"  Found {len(csv_files)} files:")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(
                f, low_memory=False, encoding_errors="replace"
            )
            df = apply_2017_column_map(df)   # strip whitespace only
            df = df.replace("Infinity", np.inf)
            df = df.replace("infinity", np.inf)
            print(f"    {f.name:<55} {len(df):>8,} rows")
            dfs.append(df)
        except Exception as e:
            print(f"    SKIPPED {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Raw combined shape: {combined.shape}")

    # ── Unify labels ──────────────────────────────────────────────
    combined["Label"] = (
        combined["Label"].astype(str).str.strip().map(LABEL_MAP_2017)
    )
    unmapped = combined["Label"].isna().sum()
    if unmapped:
        print(f"  Dropping {unmapped:,} rows with unmapped labels")
    combined = combined.dropna(subset=["Label"])

    print(f"\n  Label distribution:")
    print(combined["Label"].value_counts().to_string())

    # ── Fill missing features ─────────────────────────────────────
    missing = [c for c in feature_cols if c not in combined.columns]
    if missing:
        print(f"\n  WARNING: {len(missing)} features missing — filling 0:")
        for m in missing:
            print(f"    - {m}")
            combined[m] = 0.0

    # ── Clean ─────────────────────────────────────────────────────
    combined = combined.replace([np.inf, -np.inf], np.nan)
    before   = len(combined)
    combined = combined.dropna(subset=feature_cols)

    # min_seg_size_forward has CICFlowMeter version differences between
    # 2017 and 2018 — 2017 produces large negative values (mean=-2746)
    # vs valid positive values in 2018 (mean=+18.78).
    # Clip to physically valid range: [0, 1460] (max standard Ethernet MSS)
    if "min_seg_size_forward" in combined.columns:
        n_clipped = (combined["min_seg_size_forward"] < 0).sum()
        combined["min_seg_size_forward"] = combined[
            "min_seg_size_forward"
        ].clip(lower=0, upper=1460)
        print(f"  Clipped {n_clipped:,} negative min_seg_size_forward values")

    if "Flow Duration" in combined.columns:
        combined = combined[combined["Flow Duration"] >= 0]
    combined = combined.drop_duplicates(subset=feature_cols)
    print(f"\n  After cleaning: {before:,} → {len(combined):,} rows\n")

    return combined


# ── Step 2: Align and produce two arrays ─────────────────────────

def align_and_scale(df: pd.DataFrame,
                    feature_cols: list) -> tuple:
    """
    Returns two separate arrays:
      X_raw    → XGBoost  (unscaled — tree splits are scale-invariant)
      X_scaled → Autoencoder (2018 scaler applied — AE needs normalisation)

    Scaler is NOT refit on 2017 data. The Phase 2 scaler is applied
    directly so AE reconstruction error is on the same scale as training.
    """
    print("[2/5] Aligning features...")

    scaler   = joblib.load(MODEL_DIR / "scaler.pkl")
    X_raw    = df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    print(f"  Feature count     : {X_raw.shape[1]}")
    print(f"  XGBoost input     : raw unscaled features")
    print(f"  Autoencoder input : 2018-scaler transformed")
    print(f"  X_raw   mean/std  : {X_raw.mean():.2f} / {X_raw.std():.2f}")
    print(f"  X_scaled mean/std : {X_scaled.mean():.4f} / "
          f"{X_scaled.std():.4f}\n")

    return X_raw, X_scaled


# ── Step 3: XGBoost evaluation ────────────────────────────────────

def evaluate_xgboost(X_raw: np.ndarray,
                     df: pd.DataFrame,
                     feature_cols: list) -> tuple:
    print("[3/5] Evaluating XGBoost on CICIDS2017...")

    model  = joblib.load(MODEL_DIR / "xgboost_nids.pkl")
    le     = joblib.load(MODEL_DIR / "label_encoder.pkl")

    with open(MODEL_DIR / "threshold_config.json") as f:
        threshold_cfg = json.load(f)

    inf_idx       = list(le.classes_).index("Infiltration")
    inf_threshold = threshold_cfg["infiltration_threshold"]
    known_classes = set(le.classes_)

    # ── Predict with Infiltration threshold ───────────────────────
    proba     = model.predict_proba(X_raw)     # raw features
    base_pred = np.argmax(proba, axis=1)

    inf_mask       = (base_pred == inf_idx)
    low_confidence = inf_mask & (proba[:, inf_idx] < inf_threshold)

    if low_confidence.any():
        proba_no_inf             = proba.copy()
        proba_no_inf[:, inf_idx] = 0.0
        base_pred[low_confidence] = np.argmax(proba_no_inf, axis=1)[low_confidence]

    y_pred_labels = le.inverse_transform(base_pred)

    # ── Separate PortScan (unseen class) ──────────────────────────
    portscan_mask = (df["Label"] == "PortScan")
    known_mask    = ~portscan_mask

    results = {}

    # ── Evaluate known classes ────────────────────────────────────
    if known_mask.any():
        y_true_known = df.loc[known_mask, "Label"].values
        y_pred_known = y_pred_labels[known_mask]
        eval_classes = sorted(set(y_true_known) & known_classes)

        report_str  = classification_report(
            y_true_known, y_pred_known,
            labels       = eval_classes,
            target_names = eval_classes,
            digits       = 4,
            zero_division = 0,
        )
        report_dict = classification_report(
            y_true_known, y_pred_known,
            labels       = eval_classes,
            target_names = eval_classes,
            output_dict  = True,
            zero_division = 0,
        )

        print("\n" + "="*60)
        print("XGBoost — CICIDS2017 (cross-dataset, no retraining)")
        print("="*60)
        print(report_str)

        f1_per_class = {
            cls: round(report_dict[cls]["f1-score"], 4)
            for cls in eval_classes
        }

        print("Per-class F1 (2017):")
        for cls, f1 in sorted(
            f1_per_class.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(f1 * 40)
            print(f"  {cls:<15} {bar:<40} {f1:.4f}")

        results["known_classes"] = {
            "f1_per_class": f1_per_class,
            "macro_f1":     round(report_dict["macro avg"]["f1-score"], 4),
            "weighted_f1":  round(report_dict["weighted avg"]["f1-score"], 4),
            "classification_report": report_dict,
        }

    # ── PortScan analysis ─────────────────────────────────────────
    if portscan_mask.any():
        ps_preds  = y_pred_labels[portscan_mask]
        ps_counts = pd.Series(ps_preds).value_counts()

        print("\n" + "="*60)
        print("PortScan — UNSEEN CLASS (absent from 2018 training data)")
        print("="*60)
        print(f"  PortScan samples : {portscan_mask.sum():,}")
        print("  Predicted as:")
        for pred_cls, count in ps_counts.items():
            pct = count / portscan_mask.sum() * 100
            print(f"    {pred_cls:<15} {count:>7,}  ({pct:.1f}%)")

        results["portscan_unseen"] = {
            "n_samples":    int(portscan_mask.sum()),
            "predicted_as": {k: int(v) for k, v in ps_counts.items()},
            "note": (
                "PortScan absent from 2018 training set. "
                "Model routes flows to nearest learned pattern."
            ),
        }

    return results, y_pred_labels


# ── Step 4: Autoencoder evaluation ───────────────────────────────

def evaluate_autoencoder(X_scaled: np.ndarray,
                         df: pd.DataFrame) -> dict:
    print("\n[4/5] Evaluating autoencoder on CICIDS2017...")

    with open(MODEL_DIR / "threshold_ae.json") as f:
        ae_cfg = json.load(f)

    with open(PROCESSED_DIR / "feature_list.json") as f:
        feature_cols = json.load(f)

    threshold = ae_cfg["threshold"]
    model     = Autoencoder(input_dim=len(feature_cols), bottleneck_dim=8)
    model.load_state_dict(
        torch.load(MODEL_DIR / "autoencoder.pt", map_location="cpu")
    )
    model.eval()
    model.to(DEVICE)

    phase3_rates = ae_cfg.get("detection_rates", {})
    results      = {}

    print(f"\n  {'Class':<15} {'Mean Err':>10} {'95th pct':>10} "
          f"{'Detect%':>9}  {'vs Phase3':>10}")
    print("  " + "-" * 60)

    labels_array = df["Label"].values

    for label in sorted(df["Label"].unique()):
        mask  = (labels_array == label)
        X_sub = X_scaled[mask]

        if len(X_sub) > 50_000:
            idx   = np.random.RandomState(42).choice(
                len(X_sub), 50_000, replace=False
            )
            X_sub = X_sub[idx]

        tensor   = torch.tensor(
            X_sub.astype(np.float32), dtype=torch.float32
        ).to(DEVICE)
        errors   = model.reconstruction_error(tensor)
        detected = float((errors > threshold).mean() * 100)

        p3_str = phase3_rates.get(label, None)
        if p3_str:
            p3_val    = float(str(p3_str).replace("%", ""))
            delta_str = f"{detected - p3_val:+.1f}%"
        else:
            delta_str = "N/A"

        results[label] = {
            "mean_error":    float(errors.mean()),
            "p95_error":     float(np.percentile(errors, 95)),
            "detection_pct": round(detected, 2),
            "vs_phase3":     delta_str,
            "n_samples":     len(X_sub),
        }

        bar = "█" * int(detected / 2.5)
        print(f"  {label:<15} {errors.mean():>10.6f} "
              f"{np.percentile(errors, 95):>10.6f} "
              f"{detected:>8.1f}%  {delta_str:>10}  {bar}")

    return results


# ── Step 5: Delta report ──────────────────────────────────────────

def print_delta_report(xgb_results: dict):
    print("\n" + "="*60)
    print("Cross-dataset generalisation: Phase2 (2018) vs Phase4 (2017)")
    print("="*60)

    with open(RESULTS_DIR / "phase2_report.json") as f:
        p2 = json.load(f)

    # Handle both possible key names from phase2_report
    p2_report = p2.get("classification_report", {})
    p2_f1 = {
        cls: round(v["f1-score"], 4)
        for cls, v in p2_report.items()
        if cls not in ("accuracy", "macro avg", "weighted avg")
    }

    p4_f1 = xgb_results.get(
        "known_classes", {}
    ).get("f1_per_class", {})

    all_classes = sorted(set(p2_f1) | set(p4_f1))

    print(f"\n  {'Class':<15} {'Phase2 F1':>10} {'Phase4 F1':>10} "
          f"{'Delta':>8}  {'Status'}")
    print("  " + "-" * 58)

    for cls in all_classes:
        p2_val = p2_f1.get(cls)
        p4_val = p4_f1.get(cls)

        if p2_val is None:
            print(f"  {cls:<15} {'N/A':>10} {p4_val:>10.4f} "
                  f"{'—':>8}  new in 2017")
            continue
        if p4_val is None:
            print(f"  {cls:<15} {p2_val:>10.4f} {'N/A':>10} "
                  f"{'—':>8}  unseen in 2017")
            continue

        delta  = p4_val - p2_val
        status = ("✓ stable"     if abs(delta) <= 0.05
                  else "~ minor drop" if -0.15 <= delta < -0.05
                  else "✗ major drop" if delta < -0.15
                  else "↑ improved")
        print(f"  {cls:<15} {p2_val:>10.4f} {p4_val:>10.4f} "
              f"{delta:>+8.4f}  {status}")

    macro = xgb_results.get("known_classes", {}).get("macro_f1", "N/A")
    print(f"\n  Macro F1 on CICIDS2017: {macro}")
    print()


# ── Step 6: Save ──────────────────────────────────────────────────

def save_results(xgb_results: dict, ae_results: dict):
    report = {
        "phase":       "4 — Cross-dataset Validation",
        "train_set":   "CSE-CIC-IDS2018",
        "eval_set":    "CICIDS2017",
        "xgboost":     xgb_results,
        "autoencoder": ae_results,
        "methodology": {
            "xgboost_input":   "raw unscaled features",
            "ae_input":        "2018 scaler applied (not refit on 2017)",
            "no_retraining":   True,
            "portscan_status": "unseen class — absent from 2018 training",
        },
    }
    with open(RESULTS_DIR / "phase4_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: results/phase4_report.json")


# ── Main ──────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("PHASE 4 — Cross-Dataset Validation on CICIDS2017")
    print("=" * 60 + "\n")

    with open(PROCESSED_DIR / "feature_list.json") as f:
        feature_cols = json.load(f)

    df               = load_2017(feature_cols)
    X_raw, X_scaled  = align_and_scale(df, feature_cols)

    xgb_results, _   = evaluate_xgboost(X_raw, df, feature_cols)
    ae_results        = evaluate_autoencoder(X_scaled, df)

    print_delta_report(xgb_results)
    save_results(xgb_results, ae_results)

    macro_f1 = xgb_results.get("known_classes", {}).get("macro_f1", "N/A")

    print("=" * 60)
    print("Phase 4 complete.")
    print(f"  Macro F1 on CICIDS2017 : {macro_f1}")
    print(f"  Artifact: results/phase4_report.json")
    print("=" * 60)
