"""
State-Vector Fusion Engine (/twin).

Fuses asynchronous lab results, bedside/wearable vitals, and patient demographics
into a single versioned TwinState object:
1. Unit normalization to canonical clinical units (defined in shared/twin_schema.json).
2. Missing-value imputation using last-known-good state or sex/age-aware clinical defaults.
3. Physiological clamping to plausible bounds.
4. Derivation of secondary hemodynamic & renal indices (eGFR via CKD-EPI 2021,
   Cardiac Output from EF, HR, and BSA/BMI).
5. Clinical threshold evaluation (normal / caution / critical).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional, Tuple

from .schema import (
    TWIN_SCHEMA,
    FieldStatus,
    TwinIngestPayload,
    TwinState,
    TwinVector,
)

# Canonical unit conversion factors to target canonical units in twin_schema.json
_UNIT_CONVERSIONS: Dict[Tuple[str, str], float] = {
    # Pressure: kPa -> mmHg
    ("systolic_bp", "kPa"): 7.50062,
    ("diastolic_bp", "kPa"): 7.50062,
    # Lipids: mmol/L -> mg/dL
    ("ldl", "mmol/L"): 38.67,
    ("hdl", "mmol/L"): 38.67,
    ("triglycerides", "mmol/L"): 88.57,
    # Creatinine: umol/L -> mg/dL
    ("creatinine", "umol/L"): 1.0 / 88.4,
    # Troponin: ng/L or ug/L -> ng/mL
    ("troponin", "ng/L"): 0.001,
    ("troponin", "ug/L"): 1.0,
    # BNP: ng/L -> pg/mL (1:1)
    ("bnp", "ng/L"): 1.0,
    # Potassium: mmol/L -> mEq/L (1:1 for monovalent K+)
    ("potassium", "mmol/L"): 1.0,
}


def normalize_unit(field: str, value: float, unit: Optional[str]) -> float:
    """Convert an incoming measurement to its canonical schema unit."""
    if unit is None:
        return value
    canonical = TWIN_SCHEMA["fields"].get(field, {}).get("unit")
    if unit == canonical:
        return value
    factor = _UNIT_CONVERSIONS.get((field, unit))
    if factor is None:
        return value
    return value * factor


def clamp_field(field: str, value: float) -> float:
    """Clamp a numeric field to its physiologically plausible range."""
    meta = TWIN_SCHEMA["fields"].get(field, {})
    lo = meta.get("physiological_min")
    hi = meta.get("physiological_max")
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return value


def compute_egfr_ckd_epi_2021(creatinine_mg_dl: float, age: int, sex: Literal["M", "F"]) -> float:
    """
    2021 CKD-EPI Creatinine Equation (race-free) for estimated GFR (mL/min/1.73m2).
    """
    scr = max(0.3, creatinine_mg_dl)
    kappa = 0.7 if sex == "F" else 0.9
    alpha = -0.241 if sex == "F" else -0.302
    sex_factor = 1.012 if sex == "F" else 1.0
    ratio = scr / kappa
    egfr = 142.0 * (min(ratio, 1.0) ** alpha) * (max(ratio, 1.0) ** -1.200) * (0.9938 ** age) * sex_factor
    return round(clamp_field("egfr", egfr), 1)


def compute_cardiac_output(ejection_fraction: float, heart_rate: float, bmi: float, sex: Literal["M", "F"]) -> float:
    """
    Reduced-order hemodynamic derivation of Cardiac Output (L/min).
    Estimates left ventricular end-diastolic volume (EDV) scaled by body habitus (BMI)
    and biological sex, then computes Stroke Volume = EDV * (EF / 100) and
    Cardiac Output = (SV * HR) / 1000.
    """
    base_edv_ml = 128.0 if sex == "M" else 108.0
    habitus_scale = max(0.82, min(1.28, 1.0 + 0.012 * (bmi - 25.0)))
    edv_ml = base_edv_ml * habitus_scale
    stroke_volume_ml = edv_ml * (ejection_fraction / 100.0)
    co_l_min = (stroke_volume_ml * heart_rate) / 1000.0
    return round(clamp_field("cardiac_output", co_l_min), 2)


def evaluate_field_status(field: str, value: float) -> FieldStatus:
    """
    Classify a numeric field into normal / caution / critical based on
    thresholds in shared/twin_schema.json.
    """
    meta = TWIN_SCHEMA["fields"].get(field, {})
    normal_min = meta.get("normal_min")
    normal_max = meta.get("normal_max")
    caution_low = meta.get("caution_low")
    caution_high = meta.get("caution_high")
    critical_low = meta.get("critical_low")
    critical_high = meta.get("critical_high")

    if normal_min is None or normal_max is None:
        return FieldStatus.NORMAL

    if (critical_low is not None and value <= critical_low and critical_low < normal_min) or (
        critical_high is not None and value >= critical_high and critical_high > normal_max
    ):
        return FieldStatus.CRITICAL

    if normal_min <= value <= normal_max:
        return FieldStatus.NORMAL

    if (caution_low is not None and value < normal_min and value >= caution_low) or (
        caution_high is not None and value > normal_max and value <= caution_high
    ):
        return FieldStatus.CAUTION

    return FieldStatus.CRITICAL


def evaluate_all_statuses(vector: TwinVector) -> Dict[str, FieldStatus]:
    """Evaluate clinical status across all monitored numeric fields."""
    numeric_fields = [
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
        "bmi",
    ]
    data = vector.model_dump()
    return {f: evaluate_field_status(f, float(data[f])) for f in numeric_fields}


class TwinFusionEngine:
    """Fuses raw clinical inputs + optional prior TwinState into a new versioned TwinState."""

    @staticmethod
    def fuse(payload: TwinIngestPayload, prior_state: Optional[TwinState] = None) -> TwinState:
        fields_meta = TWIN_SCHEMA["fields"]
        prior_dict: Dict[str, Any] = prior_state.vector.model_dump() if prior_state else {}
        provenance: Dict[str, Literal["measured", "imputed", "derived"]] = {}

        resolved: Dict[str, Any] = {}
        payload_dict = payload.model_dump()

        # 1. Resolve demographics first (needed for derived eGFR & CO)
        for demo_field in ("age", "sex", "bmi", "smoking_status", "diabetes_status", "family_history"):
            val = payload_dict.get(demo_field)
            if val is not None:
                if demo_field == "bmi":
                    val = round(clamp_field("bmi", float(val)), 2)
                resolved[demo_field] = val
                provenance[demo_field] = "measured"
            elif demo_field in prior_dict:
                resolved[demo_field] = prior_dict[demo_field]
                provenance[demo_field] = "imputed"
            else:
                resolved[demo_field] = fields_meta[demo_field]["default"]
                provenance[demo_field] = "imputed"

        # 2. Resolve direct vitals & labs with unit normalization and clamping
        direct_fields = (
            "heart_rate",
            "systolic_bp",
            "diastolic_bp",
            "ejection_fraction",
            "o2_saturation",
            "ldl",
            "hdl",
            "triglycerides",
            "bnp",
            "troponin",
            "potassium",
            "creatinine",
        )
        for field in direct_fields:
            raw_val = payload_dict.get(field)
            if raw_val is not None:
                unit = payload.units.get(field)
                norm_val = normalize_unit(field, float(raw_val), unit)
                clamped = clamp_field(field, norm_val)
                resolved[field] = round(clamped, 3 if field == "troponin" else 2)
                provenance[field] = "measured"
            elif field in prior_dict:
                resolved[field] = prior_dict[field]
                provenance[field] = "imputed"
            else:
                # Sex-aware HDL adjustment if using default
                default_val = float(fields_meta[field]["default"])
                if field == "hdl" and resolved["sex"] == "F":
                    default_val = 54.0
                resolved[field] = default_val
                provenance[field] = "imputed"

        # Ensure diastolic BP remains physiologically below systolic BP
        if resolved["diastolic_bp"] >= resolved["systolic_bp"] - 10.0:
            resolved["diastolic_bp"] = round(max(35.0, resolved["systolic_bp"] - 20.0), 1)

        # 3. Resolve or derive eGFR
        if payload_dict.get("egfr") is not None:
            resolved["egfr"] = round(clamp_field("egfr", float(payload_dict["egfr"])), 1)
            provenance["egfr"] = "measured"
        else:
            resolved["egfr"] = compute_egfr_ckd_epi_2021(
                creatinine_mg_dl=resolved["creatinine"],
                age=int(resolved["age"]),
                sex=resolved["sex"],
            )
            provenance["egfr"] = "derived"

        # 4. Resolve or derive Cardiac Output
        if payload_dict.get("cardiac_output") is not None:
            resolved["cardiac_output"] = round(clamp_field("cardiac_output", float(payload_dict["cardiac_output"])), 2)
            provenance["cardiac_output"] = "measured"
        else:
            resolved["cardiac_output"] = compute_cardiac_output(
                ejection_fraction=resolved["ejection_fraction"],
                heart_rate=resolved["heart_rate"],
                bmi=resolved["bmi"],
                sex=resolved["sex"],
            )
            provenance["cardiac_output"] = "derived"

        vector = TwinVector(**resolved)
        statuses = evaluate_all_statuses(vector)
        next_version = (prior_state.state_version + 1) if prior_state else 1

        return TwinState(
            schema_version=TWIN_SCHEMA["schema_version"],
            state_version=next_version,
            patient_id=payload.patient_id,
            timestamp=payload.timestamp or datetime.now(timezone.utc),
            source=payload.source,
            refresh_interval_seconds=TWIN_SCHEMA["refresh_interval_seconds"],
            vector=vector,
            statuses=statuses,
            provenance=provenance,
        )
