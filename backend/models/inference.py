"""
Low-Latency Inference & Explainability Service (<200ms target) for Hybrid XGBoost-LSTM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import joblib
import numpy as np
import shap
import torch
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

from backend.models.features import (
    FEATURE_DISPLAY_NAMES,
    SEQ_LEN,
    TABULAR_FEATURE_NAMES,
    TS_FEATURE_NAMES,
    compute_static_baseline_risk,
    extract_tabular_features,
    extract_timeseries_window,
)
from backend.models.lstm_net import VitalsLSTM
from backend.twin.schema import TwinVector

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


class RiskFactorContribution(BaseModel):
    """Single SHAP / temporal feature contribution to the 30-day MACE prediction."""

    feature: str
    display_name: str
    value: float
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]
    magnitude: float


class RiskPrediction(BaseModel):
    """Output of the Hybrid XGBoost-LSTM 30-day MACE Risk Service."""

    risk_score_pct: float = Field(..., ge=0.0, le=100.0)
    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    xgb_prob: float
    lstm_prob: float
    baseline_static_pct: float
    latency_ms: float
    top_factors: List[RiskFactorContribution]


class HybridMacePredictor:
    """
    Singleton-friendly inference wrapper loading XGBoost, PyTorch VitalsLSTM,
    StandardScaler, LogisticRegression stacking meta-model, and SHAP TreeExplainer.
    """

    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or ARTIFACTS_DIR
        self._loaded = False
        self.xgb_model: Optional[XGBClassifier] = None
        self.lstm_model: Optional[VitalsLSTM] = None
        self.seq_scaler = None
        self.meta_model = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.evaluation_metrics: dict = {}

    def load(self) -> None:
        """Load trained model artifacts from disk (or train if missing)."""
        if self._loaded:
            return

        xgb_path = self.artifacts_dir / "xgb_model.json"
        lstm_path = self.artifacts_dir / "lstm_model.pt"
        scaler_path = self.artifacts_dir / "seq_scaler.joblib"
        meta_path = self.artifacts_dir / "meta_model.joblib"
        metrics_path = self.artifacts_dir / "evaluation_metrics.json"

        if not (xgb_path.exists() and lstm_path.exists() and scaler_path.exists() and meta_path.exists()):
            from backend.models.train import train_and_evaluate

            train_and_evaluate()

        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(str(xgb_path))

        self.lstm_model = VitalsLSTM(input_size=len(TS_FEATURE_NAMES), hidden_size=48, num_layers=2, dropout=0.2)
        self.lstm_model.load_state_dict(torch.load(lstm_path, map_location="cpu", weights_only=True))
        self.lstm_model.eval()

        self.seq_scaler = joblib.load(scaler_path)
        self.meta_model = joblib.load(meta_path)
        self.explainer = shap.TreeExplainer(self.xgb_model)

        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.evaluation_metrics = json.load(f)

        self._loaded = True

    def _classify_risk_level(self, pct: float) -> Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]:
        if pct >= 55.0:
            return "CRITICAL"
        if pct >= 30.0:
            return "HIGH"
        if pct >= 12.0:
            return "MODERATE"
        return "LOW"

    def predict(
        self,
        current_vector: TwinVector,
        history: Optional[Sequence[TwinVector]] = None,
        mace_risk_multiplier: float = 1.0,
        include_shap: bool = True,
        top_k: int = 6,
    ) -> RiskPrediction:
        """
        Run hybrid XGBoost-LSTM inference on a patient's current TwinVector
        and optional longitudinal history window.
        """
        self.load()
        t0 = time.perf_counter()

        seq_vecs = list(history) if history else [current_vector]
        if seq_vecs[-1] != current_vector:
            seq_vecs.append(current_vector)

        x_tab = extract_tabular_features(current_vector).reshape(1, -1)
        x_seq = extract_timeseries_window(seq_vecs, seq_len=SEQ_LEN)
        T, F = x_seq.shape
        x_seq_scaled = self.seq_scaler.transform(x_seq.reshape(-1, F)).reshape(1, T, F)

        xgb_prob = float(self.xgb_model.predict_proba(x_tab)[0, 1])
        with torch.no_grad():
            lstm_logit = self.lstm_model(torch.tensor(x_seq_scaled, dtype=torch.float32))
            lstm_prob = float(torch.sigmoid(lstm_logit).item())

        eps = 1e-5
        px = float(np.clip(xgb_prob, eps, 1.0 - eps))
        pl = float(np.clip(lstm_prob, eps, 1.0 - eps))
        meta_x = np.array([[np.log(px / (1.0 - px)), np.log(pl / (1.0 - pl)), px * pl]], dtype=np.float64)
        hybrid_prob = float(self.meta_model.predict_proba(meta_x)[0, 1])

        # Apply antithrombotic risk multiplier (e.g., Aspirin)
        adjusted_prob = float(np.clip(hybrid_prob * mace_risk_multiplier, 0.001, 0.995))
        risk_pct = round(adjusted_prob * 100.0, 2)
        baseline_pct = round(compute_static_baseline_risk(current_vector) * 100.0, 2)

        top_factors: List[RiskFactorContribution] = []
        if include_shap and self.explainer is not None:
            shap_vals = self.explainer.shap_values(x_tab)
            if isinstance(shap_vals, list):
                shap_arr = np.asarray(shap_vals[-1])[0]
            else:
                shap_arr = np.asarray(shap_vals)[0]

            ranked_idx = np.argsort(np.abs(shap_arr))[::-1][:top_k]
            for idx in ranked_idx:
                fname = TABULAR_FEATURE_NAMES[int(idx)]
                sval = float(shap_arr[int(idx)])
                top_factors.append(
                    RiskFactorContribution(
                        feature=fname,
                        display_name=FEATURE_DISPLAY_NAMES.get(fname, fname),
                        value=round(float(x_tab[0, int(idx)]), 3),
                        shap_value=round(sval, 4),
                        direction="increases_risk" if sval >= 0 else "decreases_risk",
                        magnitude=round(abs(sval), 4),
                    )
                )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return RiskPrediction(
            risk_score_pct=risk_pct,
            risk_level=self._classify_risk_level(risk_pct),
            xgb_prob=round(xgb_prob, 4),
            lstm_prob=round(lstm_prob, 4),
            baseline_static_pct=baseline_pct,
            latency_ms=latency_ms,
            top_factors=top_factors,
        )
