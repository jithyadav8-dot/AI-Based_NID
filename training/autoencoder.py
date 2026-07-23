# training/autoencoder.py

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────
PROCESSED_DIR = Path("processed")
MODEL_DIR     = Path("models")
RESULTS_DIR   = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ── Device ────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Autoencoder] Using device: {DEVICE}")


# ── Model ─────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    Symmetric autoencoder for network flow anomaly detection.

    Architecture: 43 → 32 → 16 → 8 → 16 → 32 → 43

    Design choices:
    - Bottleneck of 8 forces the model to learn a compressed
      representation of benign traffic. Attack flows that don't
      fit this representation produce high reconstruction error.
    - Linear output layer (no activation): StandardScaler-normalized
      features can be negative, so sigmoid/tanh output is inappropriate.
    - Dropout 0.1: light regularisation to prevent memorising exact
      training samples. Heavier dropout hurts reconstruction quality.
    - ReLU throughout: fast, stable, avoids vanishing gradients.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, bottleneck_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, input_dim),
            # No activation — output matches StandardScaler space
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> np.ndarray:
        """
        Per-sample MSE reconstruction error.
        Returns a 1D numpy array of shape (n_samples,).
        Used for threshold calibration and live inference.
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            # Mean over feature dimension → one scalar per sample
            errors = torch.mean((x - reconstructed) ** 2, dim=1)
        return errors.cpu().numpy()


# ── Early stopping ────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # Deep copy the best weights
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Load the best weights back into the model."""
        if self.best_state:
            model.load_state_dict(self.best_state)


# ── Step 1: Load benign-only training data ────────────────────────

def load_benign_data(feature_cols: list[str]) -> tuple:
    """
    Loads benign flows from train.parquet and applies the Phase 2 scaler.
    Splits into 80% autoencoder train / 20% threshold calibration val.

    Key: we NEVER retrain the scaler. The same scaler fitted in Phase 2
    is used here so feature scales are identical at live inference.
    """
    print("[1/5] Loading benign training data...")

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")

    # Filter to BENIGN only — autoencoder sees no attack traffic
    benign = train[train["Label"] == "BENIGN"].copy()
    print(f"  Total train rows  : {len(train):,}")
    print(f"  Benign rows       : {len(benign):,}  "
          f"({len(benign)/len(train)*100:.1f}% of training set)")

    X = scaler.transform(benign[feature_cols].values.astype(np.float32))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 80/20 split — val set used only for threshold calibration
    split       = int(len(X) * 0.8)
    idx         = np.random.RandomState(42).permutation(len(X))
    X_train     = X[idx[:split]]
    X_val       = X[idx[split:]]

    print(f"  AE train split    : {len(X_train):,}")
    print(f"  Threshold val split: {len(X_val):,}\n")

    return X_train, X_val


# ── Step 2: Train ─────────────────────────────────────────────────

def train_autoencoder(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    input_dim:      int,
    bottleneck_dim: int   = 8,
    epochs:         int   = 100,
    batch_size:     int   = 2048,
    lr:             float = 1e-3,
    patience:       int   = 10,
) -> tuple[Autoencoder, list, list]:
    """
    Trains the autoencoder on benign-only data.

    batch_size=2048: large batches work well for autoencoders on
    tabular data — more stable gradients, faster epochs.

    num_workers=0: required on Windows to avoid multiprocessing errors
    with DataLoader.
    """
    print("[2/5] Training autoencoder...")
    print(f"  Input dim  : {input_dim}")
    print(f"  Batch size : {batch_size}")
    print(f"  Max epochs : {epochs}")
    print(f"  Patience   : {patience}\n")

    # Build DataLoaders
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor   = torch.tensor(X_val,   dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,    # Windows: must be 0
        pin_memory  = DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = 0,
    )
    model     = Autoencoder(input_dim, bottleneck_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    stopper   = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        train_loss = epoch_loss / len(X_train)

        # ── Validate ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(DEVICE)
                val_loss += criterion(model(batch), batch).item() * len(batch)

        val_loss /= len(X_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"train_loss={train_loss:.6f}  "
                  f"val_loss={val_loss:.6f}  "
                  f"({elapsed:.1f}s)")

        # Early stopping
        if stopper.step(val_loss, model):
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best val_loss={stopper.best_loss:.6f})")
            break

    stopper.restore_best(model)
    print(f"\n  Training complete. Best val_loss: {stopper.best_loss:.6f}")

    # Save model
    torch.save(model.state_dict(), MODEL_DIR / "autoencoder.pt")
    print(f"  Saved: models/autoencoder.pt\n")

    return model, train_losses, val_losses


# ── Step 3: Compute threshold ─────────────────────────────────────

def compute_threshold(model: Autoencoder, X_val: np.ndarray,
                      percentile: float = 95.0) -> float:
    """
    Computes the reconstruction error threshold on benign validation data.

    Uses the 95th percentile: 5% of benign flows will false-alarm.
    This is the correct tradeoff for a NIDS — slightly over-alert
    rather than miss genuine attacks.
    """
    print("[3/5] Computing reconstruction error threshold...")

    val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    errors     = model.reconstruction_error(val_tensor)

    threshold  = float(np.percentile(errors, percentile))

    print(f"  Benign val errors — "
          f"mean={errors.mean():.6f}  "
          f"std={errors.std():.6f}  "
          f"max={errors.max():.6f}")
    print(f"  Threshold ({percentile}th percentile): {threshold:.6f}")
    print(f"  Benign flows above threshold: "
          f"{(errors > threshold).sum():,} / {len(errors):,} "
          f"({(errors > threshold).mean()*100:.1f}% — expected ~5%)\n")

    threshold_config = {
        "threshold":            threshold,
        "percentile":           percentile,
        "benign_error_mean":    float(errors.mean()),
        "benign_error_std":     float(errors.std()),
        "benign_error_max":     float(errors.max()),
        "benign_false_alarm_rate": float((errors > threshold).mean()),
        "note": (
            "Reconstruction error threshold computed on benign-only "
            "validation data. Flows with MSE > threshold are flagged "
            "as anomalous by the autoencoder."
        ),
    }

    with open(MODEL_DIR / "threshold_ae.json", "w") as f:
        json.dump(threshold_config, f, indent=2)
    print(f"  Saved: models/threshold_ae.json\n")

    return threshold


# ── Step 4: Evaluate across all classes ──────────────────────────

def evaluate(model: Autoencoder, threshold: float,
             feature_cols: list[str]):
    """
    Computes reconstruction error for every class in the test set.

    The key question: does the autoencoder assign meaningfully higher
    error to attack classes — especially Infiltration — than to BENIGN?
    This validates the anomaly detection capability.
    """
    print("[4/5] Evaluating reconstruction error by class...")

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    test   = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    results_by_class = {}
    detection_rates  = {}

    print(f"\n  {'Class':<15} {'Mean Error':>12} {'Std':>10} "
          f"{'95th pct':>10} {'Detection%':>12}")
    print("  " + "-" * 62)

    for label in sorted(test["Label"].unique()):
        subset  = test[test["Label"] == label]
        # Cap at 50K samples for speed
        if len(subset) > 50_000:
            subset = subset.sample(50_000, random_state=42)

        X       = scaler.transform(
            subset[feature_cols].values.astype(np.float32)
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        tensor  = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        errors  = model.reconstruction_error(tensor)

        detected     = (errors > threshold).mean() * 100
        results_by_class[label] = {
            "mean":      float(errors.mean()),
            "std":       float(errors.std()),
            "p95":       float(np.percentile(errors, 95)),
            "detection": float(detected),
            "n_samples": len(subset),
        }
        detection_rates[label] = float(detected)

        # Visual bar for detection rate
        bar = "█" * int(detected / 2.5)
        print(f"  {label:<15} {errors.mean():>12.6f} {errors.std():>10.6f} "
              f"{np.percentile(errors, 95):>10.6f} "
              f"{detected:>10.1f}%  {bar}")

    print()

    # Save results
    report = {
        "phase":             "3 — Autoencoder Training",
        "threshold":         threshold,
        "by_class":          results_by_class,
        "detection_rates":   detection_rates,
        "architecture": {
            "layers":        "43→32→16→8→16→32→43",
            "activation":    "ReLU",
            "dropout":       0.1,
            "bottleneck_dim": 8,
            "loss":          "MSE",
            "trained_on":    "BENIGN only",
        },
    }

    with open(RESULTS_DIR / "phase3_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: results/phase3_report.json\n")

    return results_by_class


# ── Step 5: Plot ──────────────────────────────────────────────────

def plot_results(model: Autoencoder, threshold: float,
                 feature_cols: list[str],
                 train_losses: list, val_losses: list):
    """
    Two plots saved to results/:
      1. Training loss curve
      2. Reconstruction error distribution by class (with threshold line)
    """
    print("[5/5] Generating plots...")

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    test   = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Plot 1: Training curve ─────────────────────────────────────
    ax = axes[0]
    ax.plot(train_losses, label="Train loss", color="steelblue")
    ax.plot(val_losses,   label="Val loss",   color="tomato")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Plot 2: Error distribution by class ───────────────────────
    ax  = axes[1]
    colors = {
        "BENIGN":       "steelblue",
        "DDoS":         "tomato",
        "DoS":          "orange",
        "Botnet":       "purple",
        "BruteForce":   "green",
        "Infiltration": "crimson",
        "WebAttack":    "brown",
    }

    for label in sorted(test["Label"].unique()):
        subset = test[test["Label"] == label]
        if len(subset) > 10_000:
            subset = subset.sample(10_000, random_state=42)

        X      = scaler.transform(
            subset[feature_cols].values.astype(np.float32)
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        errors = model.reconstruction_error(tensor)

        # Clip for readability
        errors_clipped = np.clip(errors, 0, np.percentile(errors, 99))
        color = colors.get(label, "gray")
        lw    = 2.5 if label in ("BENIGN", "Infiltration") else 1.5
        ax.hist(errors_clipped, bins=80, alpha=0.5, density=True,
                label=label, color=color, linewidth=lw)

    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error by Traffic Class")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase3_error_distribution.png", dpi=150)
    print(f"  Saved: results/phase3_error_distribution.png\n")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────

def run():
    # Fix all sources of randomness
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    print("=" * 60)
    print("PHASE 3 — PyTorch Autoencoder Training")
    print("=" * 60 + "\n")

    with open(PROCESSED_DIR / "feature_list.json") as f:
        feature_cols = json.load(f)

    input_dim = len(feature_cols)

    X_train, X_val = load_benign_data(feature_cols)

    model, train_losses, val_losses = train_autoencoder(
        X_train, X_val,
        input_dim      = input_dim,
        bottleneck_dim = 8,
        epochs         = 100,
        batch_size     = 2048,
        lr             = 1e-3,
        patience       = 10,
    )

    threshold = compute_threshold(model, X_val, percentile=95.0)

    results   = evaluate(model, threshold, feature_cols)

    plot_results(model, threshold, feature_cols, train_losses, val_losses)

    print("=" * 60)
    print("Phase 3 complete.")
    print(f"  Threshold      : {threshold:.6f}")
    print(f"  Infiltration detection rate: "
          f"{results.get('Infiltration', {}).get('detection', 0):.1f}%")
    print(f"  Artifacts:")
    print(f"    models/autoencoder.pt")
    print(f"    models/threshold_ae.json")
    print(f"    results/phase3_report.json")
    print(f"    results/phase3_error_distribution.png")
    print("=" * 60)

if __name__ == "__main__":
    run()
