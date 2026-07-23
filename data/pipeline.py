# Data pipeline scripts
# data/pipeline.py

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm

from data.column_map import apply_2018_column_map
from data.label_maps import LABEL_MAP_2018, UNIFIED_LABELS

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ── Constants ─────────────────────────────────────────────────────

# Columns that are never features: metadata, labels, identifiers
NON_FEATURE_COLS = {
    "Label", "label", "Timestamp", "timestamp",
    "Flow ID", "Source IP", "Destination IP",
    "Src IP", "Dst IP", "src_ip", "dst_ip",
}

# ── Step 1: Load ──────────────────────────────────────────────────

def load_2018(raw_dir: str) -> pd.DataFrame:
    """
    Loads all CSVs from the 2018 directory.
    Handles corrupted files, encoding issues, and the 'Infinity'
    string that appears in some 2018 numeric columns.
    Cleans and unifies labels per chunk to minimize memory usage.
    """
    raw_dir = Path(raw_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"Found {len(csv_files)} CSV files in {raw_dir}\n")

    dfs = []
    for f in tqdm(csv_files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(
                f,
                low_memory   = False,
                encoding     = "utf-8",
                encoding_errors = "replace",   # don't crash on bad bytes
            )

            df = apply_2018_column_map(df)

            # Drop non-feature columns early to save massive amounts of memory
            cols_to_drop = [c for c in df.columns if c in NON_FEATURE_COLS and c.lower() != "label"]
            df = df.drop(columns=cols_to_drop, errors="ignore")

            # Replace the literal string 'Infinity' that appears in some files
            df = df.replace(["Infinity", "infinity"], np.inf)

            # Force feature columns to numeric, coercing errors to NaN
            for c in df.columns:
                if c.lower() != "label" and c != "_source_file":
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # Downcast float64 to float32 to halve memory usage
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].astype('float32')

            # Use category for string column to save memory
            df["_source_file"] = f.name
            df["_source_file"] = df["_source_file"].astype("category")

            # Clean per-chunk
            df = unify_labels(df, LABEL_MAP_2018, quiet=True)
            df = clean(df, quiet=True)

            dfs.append(df)

        except Exception as e:
            print(f"  Skipping {f.name}: {e}")

    if not dfs:
        raise RuntimeError("All CSV files failed to load.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nRaw combined shape after chunked cleaning: {combined.shape}")
    return combined


# ── Step 2: Unify labels ──────────────────────────────────────────

def unify_labels(df: pd.DataFrame, label_map: dict, quiet: bool = False) -> pd.DataFrame:
    """
    Maps raw dataset label strings to the unified 8-class scheme.
    Drops rows whose label doesn't appear in the map (unknown labels).
    """
    original_len = len(df)

    df["Label"] = df["Label"].astype(str).str.strip().map(label_map)

    unmapped = df["Label"].isna().sum()
    if unmapped > 0 and not quiet:
        print(f"  Dropping {unmapped} rows with unmapped labels")

    df = df.dropna(subset=["Label"])
    if not quiet:
        print(f"  Label unification: {original_len} -> {len(df)} rows")
        print(f"  Label distribution:\n{df['Label'].value_counts().to_string()}\n")
    return df


# ── Step 3: Clean ─────────────────────────────────────────────────

def clean(df: pd.DataFrame, quiet: bool = False) -> pd.DataFrame:
    """
    Removes rows that would corrupt model training:
      - Infinity values (replace with NaN, then drop)
      - NaN in any feature column
      - Negative flow durations (data collection artefact)
      - Exact duplicate rows
    """
    original_len = len(df)

    # Feature columns only (not Label or metadata)
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and c != "_source_file"
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Replace inf/-inf with NaN only in numeric columns to save memory
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in any feature column
    before_nan = len(df)
    df = df.dropna(subset=feature_cols)
    if not quiet:
        print(f"  Dropped {before_nan - len(df)} rows with NaN/Inf values")

    # Drop negative flow durations
    if "Flow Duration" in df.columns:
        before_dur = len(df)
        df = df[df["Flow Duration"] >= 0]
        if not quiet:
            print(f"  Dropped {before_dur - len(df)} rows with negative Flow Duration")

    # Drop exact duplicates
    before_dup = len(df)
    df = df.drop_duplicates(subset=feature_cols)
    if not quiet:
        print(f"  Dropped {before_dup - len(df)} duplicate rows")

    if not quiet:
        print(f"  Clean: {original_len} -> {len(df)} rows\n")
    return df


# ── Step 4: Feature selection ─────────────────────────────────────

def select_features(df: pd.DataFrame,
                    variance_threshold: float = 0.01,
                    correlation_threshold: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    """
    Removes features that add no predictive signal:
      1. Non-numeric columns (can't be used by XGBoost/autoencoder)
      2. Zero / near-zero variance features (constant or near-constant)
      3. Highly correlated feature pairs (keep one, drop the other)

    Returns the filtered DataFrame and the final feature list.
    The feature list is saved as feature_list.json — this is the
    contract all subsequent phases depend on.
    """
    # Numeric feature columns only
    candidate_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and c != "_source_file"
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[candidate_cols]

    # ── Step 4a: Variance threshold ───────────────────────────────
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(X)
    low_var = [c for c, keep in zip(candidate_cols, selector.get_support()) if not keep]
    if low_var:
        print(f"  Dropping {len(low_var)} low-variance features: {low_var}")
    candidate_cols = [c for c in candidate_cols if c not in low_var]
    X = df[candidate_cols]

    # ── Step 4b: Correlation filter ───────────────────────────────
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper.columns:
        if any(upper[col] > correlation_threshold):
            to_drop.add(col)

    if to_drop:
        print(f"  Dropping {len(to_drop)} highly correlated features")

    final_cols = [c for c in candidate_cols if c not in to_drop]
    print(f"  Final feature count: {len(final_cols)}\n")

    return df[final_cols + ["Label"]], final_cols


# ── Step 5: Balance classes ───────────────────────────────────────

def balance(df: pd.DataFrame, random_state: int = 42,
            max_upsample_ratio: int = 10) -> pd.DataFrame:
    """
    max_upsample_ratio: no class will be upsampled beyond
    (original_count × max_upsample_ratio).
    Prevents 68× replication of WebAttack (877 raw samples).
    """
    from sklearn.utils import resample

    counts         = df["Label"].value_counts()
    attack_counts  = counts.drop("BENIGN", errors="ignore")
    median_attacks = int(attack_counts.median())
    max_attacks    = int(attack_counts.max())
    benign_target  = max_attacks * 3

    print(f"  Balancing strategy:")
    print(f"    BENIGN target     : {benign_target:,} (3x largest attack class)")
    print(f"    Upsample below    : {median_attacks // 2:,} (1/2 median attack class)")
    print(f"    Max upsample ratio: {max_upsample_ratio}x\n")

    balanced_dfs = []
    for label, group in df.groupby("Label"):
        n = len(group)
        if label == "BENIGN":
            target = min(benign_target, n)
            balanced_dfs.append(
                resample(group, replace=False,
                         n_samples=target, random_state=random_state)
            )
        elif n < median_attacks // 2:
            # Cap: never upsample beyond max_upsample_ratio × original count
            uncapped_target = median_attacks // 2
            capped_target   = n * max_upsample_ratio
            target          = min(uncapped_target, capped_target)

            if capped_target < uncapped_target:
                print(f"    [{label}] Capped: "
                      f"{uncapped_target:,} -> {target:,} "
                      f"({n:,} raw x {max_upsample_ratio})")

            balanced_dfs.append(
                resample(group, replace=True,
                         n_samples=target, random_state=random_state)
            )
        else:
            balanced_dfs.append(group)

    balanced = pd.concat(balanced_dfs).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    print(f"\n  Balanced distribution:")
    print(balanced["Label"].value_counts().to_string())
    print()
    return balanced


# ── Step 6: Save ──────────────────────────────────────────────────

def save(df: pd.DataFrame, feature_cols: list[str],
         out_dir: str = "processed/"):
    """
    Saves the processed dataset and all artifacts needed by later phases.

    processed/
      train.parquet         — 80% training split (balanced)
      test.parquet          — 20% test split (unbalanced, realistic distribution)
      feature_list.json     — ordered feature list (the inter-phase contract)
      label_classes.json    — ordered class names matching LabelEncoder output
      stats.json            — dataset statistics for README / dashboard
    """
    from sklearn.model_selection import train_test_split

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_cols].values
    y = df["Label"].values

    # Stratified split: test set preserves natural class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df["Label"] = y_train
    test_df  = pd.DataFrame(X_test,  columns=feature_cols)
    test_df["Label"]  = y_test

    # Save as parquet — ~5-10× smaller and faster to load than CSV
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    test_df.to_parquet(out_dir  / "test.parquet",  index=False)

    # Feature list — used by Phase 2, 3, 4, 5, 6
    with open(out_dir / "feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Label class list — used by Phase 2 LabelEncoder
    label_classes = sorted(df["Label"].unique().tolist())
    with open(out_dir / "label_classes.json", "w") as f:
        json.dump(label_classes, f, indent=2)

    # Stats for README / dashboard
    stats = {
        "total_samples":   len(df),
        "train_samples":   len(train_df),
        "test_samples":    len(test_df),
        "feature_count":   len(feature_cols),
        "label_counts":    df["Label"].value_counts().to_dict(),
        "features":        feature_cols,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved to {out_dir}/")
    print(f"  train.parquet  : {len(train_df):,} rows")
    print(f"  test.parquet   : {len(test_df):,} rows")
    print(f"  feature_list.json: {len(feature_cols)} features")
    print(f"  label_classes.json: {label_classes}")


# ── Main pipeline ─────────────────────────────────────────────────

def run(raw_2018_dir: str = "raw/cic2018",
        out_dir: str      = "processed"):
    """
    Full Phase 1 pipeline. Call this from run_phase1.py.
    """
    print("=" * 60)
    print("PHASE 1 — Data Pipeline")
    print("=" * 60)

    print("\n[1/3] Loading, Unifying, and Cleaning CSE-CIC-IDS2018 chunks...")
    df = load_2018(raw_2018_dir)

    print("\n[2/3] Selecting features...")
    df, feature_cols = select_features(df)

    print("\n[3/3] Balancing classes...")
    df = balance(df)

    print("\nSaving processed dataset...")
    save(df, feature_cols, out_dir)

    print("\n" + "=" * 60)
    print("Phase 1 complete.")
    print(f"  {len(df):,} total rows")
    print(f"  {len(feature_cols)} features")
    print(f"  Outputs in: {out_dir}/")
    print("=" * 60)