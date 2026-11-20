"""
Clinical Demo Cohort Seeder.

Populates 3 realistic longitudinal cardiovascular patients (each with 12 historical
TwinState snapshots) if the database has no patients yet, so the dashboard, LSTM
temporal branch, and risk trend chart work immediately out of the box.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

from sqlalchemy.orm import Session

from backend.db.database import PatientRecord, TwinStateSnapshot
from backend.models.inference import HybridMacePredictor
from backend.twin.fusion import TwinFusionEngine
from backend.twin.schema import TwinIngestPayload


DEMO_PATIENTS = [
    {
        "patient_id": "pt-001",
        "full_name": "Arjun Mehta",
        "mrn": "MRN-40821",
        "primary_diagnosis": "HFrEF (NYHA II-III) & Coronary Artery Disease",
        "active_medications": {
            "metoprolol": 50.0,
            "lisinopril": 10.0,
            "furosemide": 20.0,
            "atorvastatin": 20.0,
            "amlodipine": 0.0,
            "digoxin": 0.0,
            "isosorbide_mononitrate": 0.0,
            "aspirin": 81.0,
        },
        "base": {
            "age": 64,
            "sex": "M",
            "bmi": 28.9,
            "smoking_status": "former",
            "diabetes_status": True,
            "family_history": True,
            "heart_rate": 84.0,
            "systolic_bp": 146.0,
            "diastolic_bp": 90.0,
            "ejection_fraction": 38.0,
            "o2_saturation": 94.0,
            "ldl": 148.0,
            "hdl": 38.0,
            "triglycerides": 210.0,
            "bnp": 620.0,
            "troponin": 0.048,
            "potassium": 4.1,
            "creatinine": 1.35,
        },
        "trend": {"hr": 0.6, "sbp": 0.5, "ef": -0.35, "bnp": 24.0, "trop": 0.002, "spo2": -0.15},
    },
    {
        "patient_id": "pt-002",
        "full_name": "Elena Rostova",
        "mrn": "MRN-51904",
        "primary_diagnosis": "Uncontrolled Hypertension & Hyperlipidemia",
        "active_medications": {
            "metoprolol": 0.0,
            "lisinopril": 10.0,
            "furosemide": 0.0,
            "atorvastatin": 10.0,
            "amlodipine": 5.0,
            "digoxin": 0.0,
            "isosorbide_mononitrate": 0.0,
            "aspirin": 0.0,
        },
        "base": {
            "age": 56,
            "sex": "F",
            "bmi": 26.4,
            "smoking_status": "never",
            "diabetes_status": False,
            "family_history": True,
            "heart_rate": 76.0,
            "systolic_bp": 152.0,
            "diastolic_bp": 94.0,
            "ejection_fraction": 58.0,
            "o2_saturation": 98.0,
            "ldl": 168.0,
            "hdl": 49.0,
            "triglycerides": 165.0,
            "bnp": 92.0,
            "troponin": 0.014,
            "potassium": 4.3,
            "creatinine": 0.88,
        },
        "trend": {"hr": 0.1, "sbp": 0.4, "ef": 0.0, "bnp": 1.5, "trop": 0.0, "spo2": 0.0},
    },
    {
        "patient_id": "pt-003",
        "full_name": "Marcus Vance",
        "mrn": "MRN-63218",
        "primary_diagnosis": "Post-NSTEMI Secondary Prevention (Stable)",
        "active_medications": {
            "metoprolol": 50.0,
            "lisinopril": 20.0,
            "furosemide": 0.0,
            "atorvastatin": 40.0,
            "amlodipine": 0.0,
            "digoxin": 0.0,
            "isosorbide_mononitrate": 0.0,
            "aspirin": 81.0,
        },
        "base": {
            "age": 49,
            "sex": "M",
            "bmi": 24.8,
            "smoking_status": "former",
            "diabetes_status": False,
            "family_history": False,
            "heart_rate": 64.0,
            "systolic_bp": 118.0,
            "diastolic_bp": 74.0,
            "ejection_fraction": 54.0,
            "o2_saturation": 98.5,
            "ldl": 84.0,
            "hdl": 52.0,
            "triglycerides": 115.0,
            "bnp": 110.0,
            "troponin": 0.012,
            "potassium": 4.4,
            "creatinine": 0.96,
        },
        "trend": {"hr": -0.2, "sbp": -0.3, "ef": 0.15, "bnp": -3.0, "trop": -0.0005, "spo2": 0.05},
    },
]


def seed_demo_patients(db: Session, predictor: HybridMacePredictor) -> None:
    """Seed demo patients if database is empty."""
    if db.query(PatientRecord).count() > 0:
        return

    now = datetime.now(timezone.utc)
    for spec in DEMO_PATIENTS:
        patient = PatientRecord(
            patient_id=spec["patient_id"],
            full_name=spec["full_name"],
            mrn=spec["mrn"],
            primary_diagnosis=spec["primary_diagnosis"],
            active_medications=spec["active_medications"],
        )
        db.add(patient)
        db.flush()

        prior_state = None
        history_vecs: List = []
        base = spec["base"]
        tr = spec["trend"]

        # Create 12 historical snapshots spaced 5 minutes apart
        for step in range(12):
            offset = step - 11  # -11 .. 0
            ts = now + timedelta(minutes=5 * offset)
            payload = TwinIngestPayload(
                patient_id=spec["patient_id"],
                source="ehr_telemetry_seed",
                timestamp=ts,
                age=base["age"],
                sex=base["sex"],
                bmi=base["bmi"],
                smoking_status=base["smoking_status"],
                diabetes_status=base["diabetes_status"],
                family_history=base["family_history"],
                heart_rate=round(base["heart_rate"] + tr["hr"] * offset, 1),
                systolic_bp=round(base["systolic_bp"] + tr["sbp"] * offset, 1),
                diastolic_bp=base["diastolic_bp"],
                ejection_fraction=round(base["ejection_fraction"] + tr["ef"] * offset, 1),
                o2_saturation=round(min(100.0, base["o2_saturation"] + tr["spo2"] * offset), 1),
                ldl=base["ldl"],
                hdl=base["hdl"],
                triglycerides=base["triglycerides"],
                bnp=round(max(10.0, base["bnp"] + tr["bnp"] * offset), 1),
                troponin=round(max(0.002, base["troponin"] + tr["trop"] * offset), 4),
                potassium=base["potassium"],
                creatinine=base["creatinine"],
            )
            state = TwinFusionEngine.fuse(payload, prior_state=prior_state)
            prior_state = state
            history_vecs.append(state.vector)

            pred = predictor.predict(state.vector, history=history_vecs, include_shap=False)
            snap = TwinStateSnapshot(
                patient_id=spec["patient_id"],
                timestamp=state.timestamp,
                source=state.source,
                schema_version=state.schema_version,
                state_version=state.state_version,
                vector=state.vector.model_dump(),
                statuses={k: v.value for k, v in state.statuses.items()},
                provenance=state.provenance,
                risk_score_pct=pred.risk_score_pct,
                risk_level=pred.risk_level,
            )
            db.add(snap)

    db.commit()
