import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

import config

from training.autoencoder import Autoencoder

MODEL_DIR = Path("models")


@dataclass
class Alert:
    """
    Output of one inference pass on a single flow window.
    Passed to the FastAPI alert store and dashboard in Phase 7.
    """
    timestamp:         float
    src_ip:            str
    dst_ip:            str
    src_port:          int
    dst_port:          int
    protocol:          str          # "TCP" / "UDP" / "ICMP" / "OTHER"
    n_packets:         int

    # XGBoost output
    xgb_class:         str          # predicted attack class
    xgb_confidence:    float        # max class probability
    xgb_is_attack:     bool         # True if not BENIGN

    # Autoencoder output
    ae_error:          float        # reconstruction MSE
    ae_threshold:      float        # 95th pct threshold
    ae_is_anomaly:     bool         # True if error > threshold

    # Fused output
    severity:          float        # 0–100
    level:             str          # CRITICAL / HIGH / MEDIUM / LOW / NORMAL
    label:             str          # human-readable final label

    @property
    def is_threat(self) -> bool:
        return self.level in ("CRITICAL", "HIGH")


class NIDSPredictor:
    """
    Loads all Phase 2 + Phase 3 artifacts and scores live flow windows.

    XGBoost receives raw unscaled features (scale-invariant).
    Autoencoder receives scaler-transformed features.
    Alert fusion combines both outputs into a severity score 0-100.
    """

    PROTO_MAP = {6: "TCP", 17: "UDP", 1: "ICMP"}

    def __init__(self):
        print("[NIDSPredictor] Loading models...")

        # ── XGBoost ───────────────────────────────────────────────
        self.xgb_model  = joblib.load(MODEL_DIR / "xgboost_nids.pkl")
        self.scaler     = joblib.load(MODEL_DIR / "scaler.pkl")
        self.le         = joblib.load(MODEL_DIR / "label_encoder.pkl")

        with open(MODEL_DIR / "threshold_config.json") as f:
            xgb_cfg = json.load(f)

        self.inf_idx       = list(self.le.classes_).index("Infiltration")
        self.inf_threshold = xgb_cfg["infiltration_threshold"]

        # ── Autoencoder ───────────────────────────────────────────
        with open(MODEL_DIR / "threshold_ae.json") as f:
            ae_cfg = json.load(f)

        self.ae_threshold = ae_cfg["threshold"]

        with open("processed/feature_list.json") as f:
            feature_cols = json.load(f)

        self.ae_model = Autoencoder(
            input_dim      = len(feature_cols),
            bottleneck_dim = 8,
        )
        self.ae_model.load_state_dict(
            torch.load(MODEL_DIR / "autoencoder.pt", map_location="cpu")
        )
        self.ae_model.eval()

        # ── Stats ─────────────────────────────────────────────────
        self._n_scored   = 0
        self._n_alerts   = 0
        self._latency_ms = []

        print(f"[NIDSPredictor] Ready.")
        print(f"  XGBoost classes     : {list(self.le.classes_)}")
        print(f"  Infiltration thresh : {self.inf_threshold}")
        print(f"  AE threshold        : {self.ae_threshold:.6f}")

    # ── Public API ────────────────────────────────────────────────

    def score(self, window: dict) -> Alert:
        """
        Score one flow window.
        window: dict from WindowAssembler.on_window_ready callback.
        Returns an Alert dataclass.
        """
        t0       = time.perf_counter()
        features = window["features"]    # float32 numpy array, shape (42,)

        xgb_class, xgb_conf, xgb_proba = self._xgb_predict(features)
        ae_error, ae_anomaly            = self._ae_predict(
            features,
            protocol = window["protocol"],
            dst_port = window["dst_port"],
        )
        severity, level                 = self._fuse(
            xgb_class, xgb_conf, xgb_proba, ae_error, ae_anomaly
        )
        label = self._label(xgb_class, ae_anomaly, level)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._latency_ms.append(latency_ms)
        self._n_scored += 1
        if level not in ("NORMAL", "LOW"):
            self._n_alerts += 1

        return Alert(
            timestamp      = window["timestamp"],
            src_ip         = window["src_ip"],
            dst_ip         = window["dst_ip"],
            src_port       = window["src_port"],
            dst_port       = window["dst_port"],
            protocol       = self.PROTO_MAP.get(window["protocol"], "OTHER"),
            n_packets      = window["n_packets"],
            xgb_class      = xgb_class,
            xgb_confidence = xgb_conf,
            xgb_is_attack  = (xgb_class != "BENIGN"),
            ae_error       = ae_error,
            ae_threshold   = self.ae_threshold,
            ae_is_anomaly  = ae_anomaly,
            severity       = severity,
            level          = level,
            label          = label,
        )

    @property
    def stats(self) -> dict:
        return {
            "windows_scored": self._n_scored,
            "alerts_raised":  self._n_alerts,
            "avg_latency_ms": round(
                float(np.mean(self._latency_ms)), 3
            ) if self._latency_ms else 0,
        }

    # ── XGBoost inference ─────────────────────────────────────────

    def _xgb_predict(self, features: np.ndarray) -> tuple:
        """
        Runs XGBoost on raw (unscaled) features.
        Applies Infiltration confidence threshold.
        Returns (class_name, confidence, full_proba_array).
        """
        X     = features.reshape(1, -1)
        proba = self.xgb_model.predict_proba(X)[0]    # shape (n_classes,)
        pred  = int(np.argmax(proba))

        # Infiltration threshold: only predict Infiltration if confident enough
        if pred == self.inf_idx and proba[self.inf_idx] < self.inf_threshold:
            proba_no_inf        = proba.copy()
            proba_no_inf[self.inf_idx] = 0.0
            pred                = int(np.argmax(proba_no_inf))

        cls  = self.le.inverse_transform([pred])[0]
        conf = float(proba[pred])
        return cls, conf, proba

    # ── Autoencoder inference ─────────────────────────────────────

    def _ae_predict(self, features: np.ndarray,
                    protocol: int = 0,
                    dst_port: int = 0) -> tuple:

        # Skip AE for protocols known to produce false positives:
        # QUIC (UDP/443), WireGuard (UDP/51820), IPSec (UDP/4500),
        # DNS-over-TCP (TCP/53) — all post-2018 or statistically unusual
        # but confirmed benign in practice
        if protocol == 17 and dst_port in config.AE_BYPASS_UDP:
            return 0.0, False
        if protocol == 6 and dst_port in config.AE_BYPASS_TCP:
            return 0.0, False

        X_scaled = self.scaler.transform(features.reshape(1, -1))
        X_scaled = np.clip(X_scaled, -10, 10)
        tensor   = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            error = float(self.ae_model.reconstruction_error(tensor)[0])

        return error, (error > self.ae_threshold)

    # ── Alert fusion ──────────────────────────────────────────────

    def _fuse(
        self,
        xgb_class:  str,
        xgb_conf:   float,
        xgb_proba:  np.ndarray,
        ae_error:   float,
        ae_anomaly: bool,
    ) -> tuple:
        """
        Combines XGBoost and AE outputs into a severity score 0-100
        and a named alert level.

        Scoring:
          XGBoost contributes 0-60 pts — only when predicting attack.
          AE contributes 0-40 pts — scales with error / threshold ratio.

        Level thresholds:
          CRITICAL : XGBoost attack + AE anomaly (both agree)
          HIGH     : XGBoost attack with high confidence
          MEDIUM   : AE anomaly only (XGBoost sees benign)
          LOW      : Weak signal from either model
          NORMAL   : No signal
        """
        xgb_is_attack = (xgb_class != "BENIGN")

        # XGBoost score: 0-60 pts
        if xgb_is_attack:
            xgb_score = xgb_conf * 60.0
        else:
            xgb_score = 0.0

        # AE score: 0-40 pts — capped at 2× threshold
        ae_ratio  = min(ae_error / max(self.ae_threshold, 1e-9), 2.0)
        ae_score  = ae_ratio * 20.0   # 0–40

        severity = min(xgb_score + ae_score, 100.0)

        # Level determination
        if xgb_is_attack and xgb_conf >= 0.80 and ae_anomaly:
            level = "CRITICAL"   # both models agree with high confidence
        elif xgb_is_attack and xgb_conf >= 0.60:
            level = "HIGH"       # XGBoost confident, known signature
        elif xgb_is_attack and xgb_conf >= 0.30:
            level = "MEDIUM"     # XGBoost lower confidence
        elif ae_anomaly:
            level = "MEDIUM"     # AE anomaly, XGBoost sees benign
        elif ae_score > 20:
            level = "LOW"        # AE elevated but below threshold
        else:
            level = "NORMAL"

        return round(severity, 2), level

    def _label(self, xgb_class: str, ae_anomaly: bool, level: str) -> str:
        """Human-readable label for the alert."""
        if xgb_class != "BENIGN":
            return xgb_class
        if ae_anomaly:
            return "Anomaly (Unknown)"
        return "Benign"
