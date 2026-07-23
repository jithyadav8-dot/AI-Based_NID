import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PROCESSED_DIR = Path("processed")
MODEL_DIR     = Path("models")
RESULTS_DIR   = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    print("[1/5] Loading processed data...")
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    with open(PROCESSED_DIR / "feature_list.json") as f:
        feature_cols = json.load(f)
    with open(PROCESSED_DIR / "label_classes.json") as f:
        label_classes = json.load(f)

    print(f"  Train : {len(train):,} rows")
    print(f"  Test  : {len(test):,} rows")
    print(f"  Classes: {label_classes}\n")
    return train, test, feature_cols, label_classes


def fit_and_save_scaler(train, feature_cols):
    """
    Fits and saves scaler for autoencoder use only.
    NOT applied to XGBoost inputs.
    """
    print("[2/5] Fitting StandardScaler (autoencoder use only)...")
    scaler  = StandardScaler()
    X_train = train[feature_cols].values.astype(np.float32)
    scaler.fit(X_train)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"  Saved: models/scaler.pkl  (autoencoder use only)\n")
    return scaler


def encode_labels(train, test, label_classes):
    print("[3/5] Encoding labels...")
    le = LabelEncoder()
    le.classes_ = np.array(label_classes)

    y_train = le.transform(train["Label"].values)
    y_test  = le.transform(test["Label"].values)

    joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
    print(f"  Class → integer mapping:")
    for i, cls in enumerate(le.classes_):
        count = int((y_train == i).sum())
        print(f"    {i}: {cls:<15} ({count:,} training samples)")
    print(f"  Saved: models/label_encoder.pkl\n")
    return y_train, y_test, le


def train_model(train, test, feature_cols, y_train, y_test, n_classes):
    print("[4/5] Training XGBoost on raw (unscaled) features...")

    # Raw features — no scaler applied
    # Trees use threshold comparisons on actual values — scale invariant
    # Scaling bakes in 2018 distributions, collapsing 2017 generalisation
    X_train = train[feature_cols].values.astype(np.float32)
    X_test  = test[feature_cols].values.astype(np.float32)

    sample_weights = compute_sample_weight(
        class_weight="balanced", y=y_train
    )

    model = xgb.XGBClassifier(
        n_estimators          = 500,
        max_depth             = 8,
        learning_rate         = 0.1,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 5,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        tree_method           = "hist",
        device                = "cuda",
        objective             = "multi:softprob",
        num_class             = n_classes,
        eval_metric           = "mlogloss",
        early_stopping_rounds = 30,
        n_jobs                = -1,
        random_state          = 42,
        verbosity             = 1,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        sample_weight = sample_weights,
        eval_set      = [(X_test, y_test)],
        verbose       = 50,
    )
    print(f"\n  Training complete in {time.time()-t0:.1f}s")
    print(f"  Best iteration : {model.best_iteration}")
    print(f"  Best mlogloss  : {model.best_score:.6f}")

    joblib.dump(model, MODEL_DIR / "xgboost_nids.pkl")
    print(f"  Saved: models/xgboost_nids.pkl\n")
    return model, X_test


def evaluate(model, X_test, y_test, le):
    print("[5/5] Evaluating on 2018 test set...")

    y_pred = model.predict(X_test)

    report_str  = classification_report(
        y_test, y_pred, target_names=le.classes_, digits=4
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=le.classes_,
        output_dict=True
    )

    print("\n" + "="*60)
    print("Phase 2 — XGBoost on raw features (2018 test set)")
    print("="*60)
    print(report_str)

    f1_per_class = f1_score(y_test, y_pred, average=None)
    print("Per-class F1:")
    for cls, f1 in zip(le.classes_, f1_per_class):
        bar = "█" * int(f1 * 40)
        print(f"  {cls:<15} {bar:<40} {f1:.4f}")

    # Infiltration threshold sweep — reuse existing logic
    from diagnose_infiltration_threshold import run_threshold_sweep
    run_threshold_sweep(model, X_test, y_test, le)

    results = {
        "phase":                   "2 — XGBoost Training (raw features)",
        "scaling":                 "None — XGBoost receives raw features",
        "scaler_note":             "Scaler saved for autoencoder use only",
        "best_iteration":          int(model.best_iteration),
        "best_mlogloss":           float(model.best_score),
        "classification_report":   report_dict,
        "per_class_f1": {
            cls: round(float(f1), 4)
            for cls, f1 in zip(le.classes_, f1_per_class)
        },
    }

    with open(RESULTS_DIR / "phase2_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/phase2_report.json")
    return results


def run():
    print("=" * 60)
    print("PHASE 2 — XGBoost Training (raw features, no scaling)")
    print("=" * 60 + "\n")

    train, test, feature_cols, label_classes = load_data()

    # Fit scaler for autoencoder — not used here
    fit_and_save_scaler(train, feature_cols)

    y_train, y_test, le = encode_labels(train, test, label_classes)

    # Train on raw features
    model, X_test = train_model(
        train, test, feature_cols,
        y_train, y_test,
        n_classes=len(label_classes),
    )

    results = evaluate(model, X_test, y_test, le)

    print("\n" + "=" * 60)
    print("Phase 2 complete.")
    print(f"  Macro F1  : {results['classification_report']['macro avg']['f1-score']:.4f}")
    print("=" * 60)
