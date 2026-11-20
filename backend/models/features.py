"""
Feature Engineering & Static Baseline Risk Calculator for Hybrid XGBoost-LSTM.

Defines:
1. TABULAR_FEATURE_NAMES: 26 static/tabular & derived hemodynamic features for XGBoost.
2. TS_FEATURE_NAMES: 8 longitudinal vitals/biomarker channels over SEQ_LEN=12 steps for LSTM.
3. compute_static_baseline_risk: Framingham/ASCVD-style static risk calculator
   used as the clinical benchmark for false-positive rate comparison.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from backend.twin.schema import TwinVector

SEQ_LEN = 12

# 8 time-series channels fed into the LSTM branch
TS_FEATURE_NAMES: List[str] = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "ejection_fraction",
    "cardiac_output",
    "o2_saturation",
    "bnp",
    "troponin",
]

# 26 tabular + derived features fed into the XGBoost branch
TABULAR_FEATURE_NAMES: List[str] = [
    "age",
    "sex_male",
    "bmi",
    "smoking_code",
    "diabetes",
    "family_history",
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "ejection_fraction",
    "cardiac_output",
    "o2_saturation",
    "ldl",
    "hdl",
    "triglycerides",
    "bnp",
    "troponin",
    "potassium",
    "creatinine",
    "egfr",
    # Derived clinical interaction features
    "map_mmhg",
    "pulse_pressure",
    "shock_index",
    "ldl_hdl_ratio",
    "log_bnp",
    "log_troponin",
]

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "age": "Age (years)",
    "sex_male": "Biological Sex (Male)",
    "bmi": "Body Mass Index",
    "smoking_code": "Smoking Exposure",
    "diabetes": "Diabetes Mellitus",
    "family_history": "Family History of CAD",
    "heart_rate": "Heart Rate (bpm)",
    "systolic_bp": "Systolic BP (mmHg)",
    "diastolic_bp": "Diastolic BP (mmHg)",
    "ejection_fraction": "Ejection Fraction (%)",
    "cardiac_output": "Cardiac Output (L/min)",
    "o2_saturation": "SpO2 (%)",
    "ldl": "LDL Cholesterol (mg/dL)",
    "hdl": "HDL Cholesterol (mg/dL)",
    "triglycerides": "Triglycerides (mg/dL)",
    "bnp": "BNP (pg/mL)",
    "troponin": "Troponin I (ng/mL)",
    "potassium": "Serum Potassium (mEq/L)",
    "creatinine": "Creatinine (mg/dL)",
    "egfr": "eGFR (mL/min/1.73m2)",
    "map_mmhg": "Mean Arterial Pressure",
    "pulse_pressure": "Pulse Pressure",
    "shock_index": "Shock Index (HR/SBP)",
    "ldl_hdl_ratio": "LDL/HDL Ratio",
    "log_bnp": "Log BNP Burden",
    "log_troponin": "Log Troponin Burden",
}


def _smoking_to_code(status: str) -> float:
    return {"never": 0.0, "former": 1.0, "current": 2.0}.get(status, 0.0)


def extract_tabular_features(vector: TwinVector) -> np.ndarray:
    """
    Convert a TwinVector into a 1D numpy float32 array matching TABULAR_FEATURE_NAMES.
    """
    map_mmhg = (vector.systolic_bp + 2.0 * vector.diastolic_bp) / 3.0
    pulse_pressure = max(10.0, vector.systolic_bp - vector.diastolic_bp)
    shock_index = vector.heart_rate / max(50.0, vector.systolic_bp)
    ldl_hdl_ratio = vector.ldl / max(15.0, vector.hdl)
    log_bnp = math.log1p(max(0.0, vector.bnp))
    log_troponin = math.log1p(max(0.0, vector.troponin * 1000.0))

    row = [
        float(vector.age),
        1.0 if vector.sex == "M" else 0.0,
        float(vector.bmi),
        _smoking_to_code(vector.smoking_status),
        1.0 if vector.diabetes_status else 0.0,
        1.0 if vector.family_history else 0.0,
        float(vector.heart_rate),
        float(vector.systolic_bp),
        float(vector.diastolic_bp),
        float(vector.ejection_fraction),
        float(vector.cardiac_output),
        float(vector.o2_saturation),
        float(vector.ldl),
        float(vector.hdl),
        float(vector.triglycerides),
        float(vector.bnp),
        float(vector.troponin),
        float(vector.potassium),
        float(vector.creatinine),
        float(vector.egfr),
        float(map_mmhg),
        float(pulse_pressure),
        float(shock_index),
        float(ldl_hdl_ratio),
        float(log_bnp),
        float(log_troponin),
    ]
    return np.asarray(row, dtype=np.float32)


def extract_timeseries_window(history: Sequence[TwinVector], seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Build a (seq_len, len(TS_FEATURE_NAMES)) float32 matrix from a sequence of
    TwinVectors (oldest -> newest). Left-pads with the earliest available state
    if fewer than seq_len snapshots exist.
    """
    if not history:
        raise ValueError("history must contain at least one TwinVector")

    rows = []
    for vec in history[-seq_len:]:
        d = vec.model_dump()
        rows.append([float(d[name]) for name in TS_FEATURE_NAMES])

    while len(rows) < seq_len:
        rows.insert(0, rows[0])

    return np.asarray(rows, dtype=np.float32)


def compute_static_baseline_risk(vector: TwinVector) -> float:
    """
    Static Framingham / ASCVD-style point-in-time risk calculator (returns probability in [0, 1]).
    Uses only traditional static risk factors (age, sex, SBP, total/LDL/HDL chol,
    smoking, diabetes) without temporal biomarker trajectories or hemodynamic fusion.
    Used as the baseline comparator in model evaluation.
    """
    z = -6.1
    z += 0.055 * (vector.age - 45.0)
    z += 0.35 if vector.sex == "M" else 0.0
    z += 0.022 * max(0.0, vector.systolic_bp - 115.0)
    z += 0.008 * max(0.0, vector.ldl - 100.0)
    z -= 0.015 * (vector.hdl - 45.0)
    z += 0.55 if vector.smoking_status == "current" else (0.18 if vector.smoking_status == "former" else 0.0)
    z += 0.65 if vector.diabetes_status else 0.0
    z += 0.30 if vector.family_history else 0.0
    z += 0.025 * max(0.0, vector.bmi - 26.0)
    prob = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, z))))
    return float(prob)
