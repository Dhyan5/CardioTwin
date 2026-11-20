"""
Shared Digital Twin State Vector Schema and Pydantic Models.

Loads `shared/twin_schema.json` so that unit definitions, physiological bounds,
and clinical alert thresholds remain strictly synchronized between the FastAPI
backend and the React + TypeScript frontend.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "shared" / "twin_schema.json"


def load_shared_schema() -> Dict[str, Any]:
    """Load the canonical shared schema JSON file."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


TWIN_SCHEMA: Dict[str, Any] = load_shared_schema()


class FieldStatus(str, Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    CRITICAL = "critical"


class TwinVector(BaseModel):
    """Unified reduced-order cardiovascular state vector."""

    # Vitals & Hemodynamics
    heart_rate: float = Field(..., description="Heart rate (bpm)")
    systolic_bp: float = Field(..., description="Systolic blood pressure (mmHg)")
    diastolic_bp: float = Field(..., description="Diastolic blood pressure (mmHg)")
    ejection_fraction: float = Field(..., description="Left ventricular ejection fraction (%)")
    cardiac_output: float = Field(..., description="Derived cardiac output (L/min)")
    o2_saturation: float = Field(..., description="Oxygen saturation SpO2 (%)")

    # Labs & Biomarkers
    ldl: float = Field(..., description="LDL cholesterol (mg/dL)")
    hdl: float = Field(..., description="HDL cholesterol (mg/dL)")
    triglycerides: float = Field(..., description="Triglycerides (mg/dL)")
    bnp: float = Field(..., description="B-type natriuretic peptide (pg/mL)")
    troponin: float = Field(..., description="Cardiac troponin I (ng/mL)")
    potassium: float = Field(..., description="Serum potassium K+ (mEq/L)")
    creatinine: float = Field(..., description="Serum creatinine (mg/dL)")
    egfr: float = Field(..., description="Estimated GFR (mL/min/1.73m2)")

    # Demographics & Risk Factors
    age: int = Field(..., ge=18, le=110)
    sex: Literal["M", "F"]
    bmi: float = Field(..., description="Body mass index (kg/m2)")
    smoking_status: Literal["never", "former", "current"]
    diabetes_status: bool
    family_history: bool


class TwinIngestPayload(BaseModel):
    """
    Raw incoming patient data payload. Any clinical field may be omitted
    (imputed by the fusion engine) and optional `units` overrides are supported
    for automatic unit normalization.
    """

    patient_id: str = Field(..., min_length=1, max_length=128)
    source: str = Field(default="ehr_stream", description="Data source (wearable, ehr_stream, lab_panel, manual)")
    timestamp: Optional[datetime] = None

    # Optional vitals & hemodynamics
    heart_rate: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    ejection_fraction: Optional[float] = None
    cardiac_output: Optional[float] = None
    o2_saturation: Optional[float] = None

    # Optional labs & biomarkers
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    triglycerides: Optional[float] = None
    bnp: Optional[float] = None
    troponin: Optional[float] = None
    potassium: Optional[float] = None
    creatinine: Optional[float] = None
    egfr: Optional[float] = None

    # Optional demographics
    age: Optional[int] = None
    sex: Optional[Literal["M", "F"]] = None
    bmi: Optional[float] = None
    smoking_status: Optional[Literal["never", "former", "current"]] = None
    diabetes_status: Optional[bool] = None
    family_history: Optional[bool] = None

    # Optional unit map, e.g. {"ldl": "mmol/L", "creatinine": "umol/L"}
    units: Dict[str, str] = Field(default_factory=dict)


class TwinState(BaseModel):
    """Versioned Digital Twin State Snapshot."""

    schema_version: str = Field(default=TWIN_SCHEMA["schema_version"])
    state_version: int = Field(default=1, ge=1)
    patient_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "fusion_engine"
    refresh_interval_seconds: int = Field(default=TWIN_SCHEMA["refresh_interval_seconds"])
    vector: TwinVector
    statuses: Dict[str, FieldStatus]
    provenance: Dict[str, Literal["measured", "imputed", "derived"]]
