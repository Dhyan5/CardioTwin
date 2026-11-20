"""
FastAPI REST API (/api) for CardioTwin Digital Twin Platform.

Ties together:
- /twin: State-vector fusion engine + 5-minute refresh scheduler
- /models: Hybrid XGBoost-LSTM 30-day MACE prediction + SHAP explainability
- /pharmaco: 8-class pharmacodynamics & interaction engine
- /whatif: Weekly longitudinal simulation & plan comparison
- /db: PostgreSQL/SQLAlchemy TwinState snapshot persistence
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.api.seed import seed_demo_patients
from backend.db.database import (
    PatientRecord,
    SessionLocal,
    TwinStateSnapshot,
    get_db,
    get_latest_snapshot,
    get_recent_snapshots,
    init_db,
)
from backend.models.inference import HybridMacePredictor, RiskPrediction
from backend.pharmaco.engine import MedicationPlan, PharmacoAssessment, apply_medication_plan
from backend.twin.fusion import TwinFusionEngine
from backend.twin.schema import TWIN_SCHEMA, TwinIngestPayload, TwinState, TwinVector
from backend.whatif.simulator import NamedMedicationPlan, WhatIfComparisonResponse, WhatIfSimulator

predictor = HybridMacePredictor()
simulator = WhatIfSimulator(predictor=predictor)
_refresh_task: Optional[asyncio.Task] = None


async def _five_minute_twin_refresh_loop() -> None:
    """
    Background worker that refreshes every active patient's TwinState every
    300 seconds (5 minutes) from incoming telemetry perturbations.
    """
    interval = int(TWIN_SCHEMA.get("refresh_interval_seconds", 300))
    while True:
        await asyncio.sleep(interval)
        db = SessionLocal()
        try:
            patients = db.query(PatientRecord).all()
            for p in patients:
                _refresh_patient_twin(db, p.patient_id, source="scheduled_5m_refresh")
        except Exception as exc:  # pragma: no cover
            print(f"[TwinScheduler] Refresh error: {exc}")
        finally:
            db.close()


def _refresh_patient_twin(db: Session, patient_id: str, source: str = "telemetry_refresh") -> TwinStateSnapshot:
    latest = get_latest_snapshot(db, patient_id)
    if latest is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' has no twin snapshots")

    prior_state = latest.to_twin_state()
    v = prior_state.vector

    # Subtle physiological telemetry jitter simulating 5-minute wearable/monitor sync
    payload = TwinIngestPayload(
        patient_id=patient_id,
        source=source,
        timestamp=datetime.now(timezone.utc),
        heart_rate=round(v.heart_rate + random.uniform(-1.5, 1.5), 1),
        systolic_bp=round(v.systolic_bp + random.uniform(-2.0, 2.0), 1),
        diastolic_bp=round(v.diastolic_bp + random.uniform(-1.5, 1.5), 1),
        o2_saturation=round(min(100.0, max(70.0, v.o2_saturation + random.uniform(-0.3, 0.3))), 1),
    )
    new_state = TwinFusionEngine.fuse(payload, prior_state=prior_state)
    recent = get_recent_snapshots(db, patient_id, limit=11)
    history = [s.to_twin_state().vector for s in recent] + [new_state.vector]

    pred = predictor.predict(new_state.vector, history=history, include_shap=False)
    snap = TwinStateSnapshot(
        patient_id=patient_id,
        timestamp=new_state.timestamp,
        source=new_state.source,
        schema_version=new_state.schema_version,
        state_version=new_state.state_version,
        vector=new_state.vector.model_dump(),
        statuses={k: v.value for k, v in new_state.statuses.items()},
        provenance=new_state.provenance,
        risk_score_pct=pred.risk_score_pct,
        risk_level=pred.risk_level,
    )
    db.add(snap)
    db.commit()
    db.refresh(snap)
    return snap


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global _refresh_task
    init_db()
    predictor.load()
    db = SessionLocal()
    try:
        seed_demo_patients(db, predictor)
    finally:
        db.close()

    _refresh_task = asyncio.create_task(_five_minute_twin_refresh_loop())
    try:
        yield
    finally:
        if _refresh_task is not None:
            _refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _refresh_task


app = FastAPI(
    title="CardioTwin Digital Twin & MACE Prevention API",
    description="Reduced-order cardiovascular digital twin, hybrid XGBoost-LSTM 30-day MACE predictor, and pharmacodynamic what-if simulator.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class PatientSummary(BaseModel):
    patient_id: str
    full_name: str
    mrn: Optional[str] = None
    primary_diagnosis: Optional[str] = None
    active_medications: MedicationPlan
    latest_risk_pct: Optional[float] = None
    latest_risk_level: Optional[str] = None
    last_updated: Optional[datetime] = None


class RiskTrendPoint(BaseModel):
    timestamp: datetime
    state_version: int
    risk_score_pct: float
    heart_rate: float
    systolic_bp: float
    bnp: float
    ejection_fraction: float


class TwinDashboardResponse(BaseModel):
    patient_id: str
    full_name: str
    mrn: Optional[str] = None
    primary_diagnosis: Optional[str] = None
    active_medications: MedicationPlan
    twin_state: TwinState
    prediction: RiskPrediction
    trend: List[RiskTrendPoint]


class PredictRequest(BaseModel):
    vector: TwinVector
    history: Optional[List[TwinVector]] = None
    mace_risk_multiplier: float = Field(default=1.0, ge=0.5, le=1.5)


class PharmacoEvaluateRequest(BaseModel):
    vector: TwinVector
    medications: MedicationPlan


class WhatIfRequest(BaseModel):
    patient_id: str = "pt-001"
    horizon_days: int = Field(default=90, ge=14, le=365)
    baseline_vector: Optional[TwinVector] = None
    plans: List[NamedMedicationPlan] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "CardioTwin Digital Twin API",
        "schema_version": TWIN_SCHEMA["schema_version"],
        "refresh_interval_seconds": TWIN_SCHEMA["refresh_interval_seconds"],
    }


@app.get("/schema")
def get_schema():
    """Return the shared Digital Twin State Vector & Medication schema."""
    return TWIN_SCHEMA


@app.get("/models/metrics")
def get_model_metrics():
    """Return held-out evaluation metrics (AUC, Brier, false-positive reduction vs static baseline)."""
    predictor.load()
    return predictor.evaluation_metrics


@app.get("/patients", response_model=List[PatientSummary])
def list_patients(db: Session = Depends(get_db)):
    patients = db.query(PatientRecord).order_by(PatientRecord.patient_id).all()
    out: List[PatientSummary] = []
    for p in patients:
        latest = get_latest_snapshot(db, p.patient_id)
        out.append(
            PatientSummary(
                patient_id=p.patient_id,
                full_name=p.full_name,
                mrn=p.mrn,
                primary_diagnosis=p.primary_diagnosis,
                active_medications=MedicationPlan(**(p.active_medications or {})),
                latest_risk_pct=latest.risk_score_pct if latest else None,
                latest_risk_level=latest.risk_level if latest else None,
                last_updated=latest.timestamp if latest else p.updated_at,
            )
        )
    return out


@app.get("/twin/{patient_id}", response_model=TwinDashboardResponse)
def get_patient_twin(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == patient_id).first()
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")

    snapshots = get_recent_snapshots(db, patient_id, limit=24)
    if not snapshots:
        raise HTTPException(status_code=404, detail=f"No TwinState snapshots for '{patient_id}'")

    latest_snap = snapshots[-1]
    twin_state = latest_snap.to_twin_state()
    history_vecs = [s.to_twin_state().vector for s in snapshots]
    prediction = predictor.predict(twin_state.vector, history=history_vecs, include_shap=True)

    trend = [
        RiskTrendPoint(
            timestamp=s.timestamp,
            state_version=s.state_version,
            risk_score_pct=round(float(s.risk_score_pct or 0.0), 2),
            heart_rate=float(s.vector["heart_rate"]),
            systolic_bp=float(s.vector["systolic_bp"]),
            bnp=float(s.vector["bnp"]),
            ejection_fraction=float(s.vector["ejection_fraction"]),
        )
        for s in snapshots
    ]

    return TwinDashboardResponse(
        patient_id=patient.patient_id,
        full_name=patient.full_name,
        mrn=patient.mrn,
        primary_diagnosis=patient.primary_diagnosis,
        active_medications=MedicationPlan(**(patient.active_medications or {})),
        twin_state=twin_state,
        prediction=prediction,
        trend=trend,
    )


@app.post("/twin/ingest", response_model=TwinDashboardResponse)
def ingest_twin_data(payload: TwinIngestPayload, db: Session = Depends(get_db)):
    """
    Fuse incoming labs, vitals, and/or demographics into a new versioned TwinState
    snapshot, score 30-day MACE risk with SHAP explainability, and persist.
    """
    patient = db.query(PatientRecord).filter(PatientRecord.patient_id == payload.patient_id).first()
    if patient is None:
        patient = PatientRecord(
            patient_id=payload.patient_id,
            full_name=f"Patient {payload.patient_id}",
            active_medications={},
        )
        db.add(patient)
        db.flush()

    latest = get_latest_snapshot(db, payload.patient_id)
    prior_state = latest.to_twin_state() if latest else None

    new_state = TwinFusionEngine.fuse(payload, prior_state=prior_state)
    recent = get_recent_snapshots(db, payload.patient_id, limit=11)
    history_vecs = [s.to_twin_state().vector for s in recent] + [new_state.vector]

    prediction = predictor.predict(new_state.vector, history=history_vecs, include_shap=True)

    snap = TwinStateSnapshot(
        patient_id=payload.patient_id,
        timestamp=new_state.timestamp,
        source=new_state.source,
        schema_version=new_state.schema_version,
        state_version=new_state.state_version,
        vector=new_state.vector.model_dump(),
        statuses={k: v.value for k, v in new_state.statuses.items()},
        provenance=new_state.provenance,
        risk_score_pct=prediction.risk_score_pct,
        risk_level=prediction.risk_level,
    )
    db.add(snap)
    db.commit()

    return get_patient_twin(payload.patient_id, db=db)


@app.post("/twin/{patient_id}/refresh", response_model=TwinDashboardResponse)
def trigger_twin_refresh(patient_id: str, db: Session = Depends(get_db)):
    """Manually trigger a 5-minute telemetry fusion refresh for a patient."""
    _refresh_patient_twin(db, patient_id, source="manual_refresh")
    return get_patient_twin(patient_id, db=db)


@app.post("/predict", response_model=RiskPrediction)
def predict_mace(req: PredictRequest):
    """Run direct Hybrid XGBoost-LSTM inference + SHAP explainability."""
    return predictor.predict(
        current_vector=req.vector,
        history=req.history,
        mace_risk_multiplier=req.mace_risk_multiplier,
        include_shap=True,
    )


@app.post("/pharmaco/evaluate", response_model=PharmacoAssessment)
def evaluate_pharmaco(req: PharmacoEvaluateRequest):
    """Evaluate steady-state pharmacodynamic effects and safety flags for a medication plan."""
    return apply_medication_plan(req.vector, req.medications, fast_weight=1.0, slow_weight=1.0)


@app.post("/whatif/simulate", response_model=WhatIfComparisonResponse)
def run_whatif_simulation(req: WhatIfRequest, db: Session = Depends(get_db)):
    """
    Simulate competing medication plans forward in weekly steps (default 90 days)
    and return chart-ready time series + Day-90 delta table.
    """
    history_vecs: Optional[List[TwinVector]] = None
    if req.baseline_vector is not None:
        baseline_vec = req.baseline_vector
    else:
        snapshots = get_recent_snapshots(db, req.patient_id, limit=12)
        if not snapshots:
            raise HTTPException(status_code=404, detail=f"Patient '{req.patient_id}' not found")
        history_vecs = [s.to_twin_state().vector for s in snapshots]
        baseline_vec = history_vecs[-1]

    return simulator.compare_plans(
        patient_id=req.patient_id,
        baseline=baseline_vec,
        plans=req.plans,
        history=history_vecs,
        horizon_days=req.horizon_days,
    )
