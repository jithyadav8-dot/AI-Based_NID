# inference/predict.py
# Shared prediction function used by Phase 2 evaluation,
# Phase 4 validation, and Phase 6 live inference.

import json
import numpy as np
import joblib
import torch
from training.autoencoder import Autoencoder

class NIDSPredictor:
    """
    Wraps XGBoost with per-class confidence thresholds.
    Currently applies a calibrated threshold for Infiltration only.
    Easily extensible to other classes if Phase 4 reveals more issues.
    """

    def __init__(self, model_dir: str = "models"):
        self.xgb_model  = joblib.load(f"{model_dir}/xgboost_nids.pkl")
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        self.le     = joblib.load(f"{model_dir}/label_encoder.pkl")

        with open(f"{model_dir}/threshold_config.json") as f:
            cfg = json.load(f)

        self.inf_idx   = list(self.le.classes_).index("Infiltration")
        self.inf_threshold = cfg["infiltration_threshold"]

        # Load feature list to know the input dimension for AE
        with open("processed/feature_list.json") as f:
            feature_cols = json.load(f)

        self.ae_model = Autoencoder(input_dim=len(feature_cols), bottleneck_dim=8)
        self.ae_model.load_state_dict(torch.load(f"{model_dir}/autoencoder.pt", map_location="cpu"))
        self.ae_model.eval()

        print(f"[NIDSPredictor] Loaded. Infiltration threshold: "
              f"{self.inf_threshold}")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (predicted_labels, probabilities).
        Applies Infiltration confidence threshold before returning labels.

        X: unscaled feature array, shape (n_samples, n_features)
        """
        proba      = self.xgb_model.predict_proba(X)
        base_pred  = np.argmax(proba, axis=1)

        # Override low-confidence Infiltration predictions
        inf_predicted   = (base_pred == self.inf_idx)
        inf_proba       = proba[:, self.inf_idx]
        low_confidence  = inf_predicted & (inf_proba < self.inf_threshold)

        if low_confidence.any():
            # Fall back to second-best class
            proba_no_inf = proba.copy()
            proba_no_inf[:, self.inf_idx] = 0.0
            second_best  = np.argmax(proba_no_inf, axis=1)
            base_pred[low_confidence] = second_best[low_confidence]

        labels = self.le.inverse_transform(base_pred)
        return labels, proba

    def predict_single(self, raw_features: np.ndarray) -> dict:
        """
        raw_features: 1D array from flow assembler (unscaled).

        XGBoost  → raw features directly
        AE       → scaler.transform(raw_features) first
        """
        X = raw_features.reshape(1, -1).astype(np.float32)

        # XGBoost: raw
        proba     = self.xgb_model.predict_proba(X)
        base_pred = np.argmax(proba, axis=1)

        # Infiltration threshold
        inf_proba = proba[0, self.inf_idx]
        if base_pred[0] == self.inf_idx and inf_proba < self.inf_threshold:
            proba_no_inf              = proba.copy()
            proba_no_inf[0, self.inf_idx] = 0.0
            base_pred[0]              = np.argmax(proba_no_inf)

        predicted_class = self.le.inverse_transform(base_pred)[0]

        # Autoencoder: scaled
        X_scaled = self.scaler.transform(X)
        ae_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        ae_error  = self.ae_model.reconstruction_error(ae_tensor)[0]

        return {
            "predicted_class": predicted_class,
            "confidence":      round(float(proba[0].max()), 4),
            "class_probs":     {
                cls: round(float(p), 4)
                for cls, p in zip(self.le.classes_, proba[0])
            },
            "ae_error":        round(float(ae_error), 6),
        }
