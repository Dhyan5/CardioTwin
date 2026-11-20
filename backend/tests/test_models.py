"""Unit tests for TwinFusionEngine and HybridMacePredictor (/twin and /models)."""

from backend.models.inference import HybridMacePredictor
from backend.twin.fusion import TwinFusionEngine
from backend.twin.schema import FieldStatus, TwinIngestPayload


def test_twin_fusion_unit_normalization_imputation_and_derivation():
    payload = TwinIngestPayload(
        patient_id="pt-unit-test",
        source="lab_panel",
        age=60,
        sex="M",
        ldl=4.0,  # mmol/L -> ~154.68 mg/dL
        creatinine=132.6,  # umol/L -> ~1.5 mg/dL
        systolic_bp=18.0,  # kPa -> ~135 mmHg
        ejection_fraction=45.0,
        heart_rate=80.0,
        units={"ldl": "mmol/L", "creatinine": "umol/L", "systolic_bp": "kPa"},
    )
    state = TwinFusionEngine.fuse(payload)

    assert 150.0 < state.vector.ldl < 160.0
    assert 1.4 < state.vector.creatinine < 1.6
    assert 130.0 < state.vector.systolic_bp < 140.0
    assert state.provenance["ldl"] == "measured"
    assert state.provenance["bnp"] == "imputed"
    assert state.provenance["egfr"] == "derived"
    assert state.provenance["cardiac_output"] == "derived"
    assert state.statuses["ejection_fraction"] == FieldStatus.CAUTION
    assert state.state_version == 1

    # Incremental update increments version and preserves prior measured fields
    state2 = TwinFusionEngine.fuse(
        TwinIngestPayload(patient_id="pt-unit-test", heart_rate=135.0),
        prior_state=state,
    )
    assert state2.state_version == 2
    assert state2.vector.ldl == state.vector.ldl
    assert state2.statuses["heart_rate"] == FieldStatus.CRITICAL


def test_hybrid_model_metrics_and_latency():
    predictor = HybridMacePredictor()
    predictor.load()

    metrics = predictor.evaluation_metrics
    assert metrics["auc_hybrid"] >= 0.85
    assert metrics["false_positive_reduction_pct"] >= 30.0

    state = TwinFusionEngine.fuse(
        TwinIngestPayload(
            patient_id="pt-latency",
            age=65,
            sex="M",
            heart_rate=88.0,
            systolic_bp=148.0,
            diastolic_bp=92.0,
            ejection_fraction=36.0,
            bnp=680.0,
            troponin=0.06,
            ldl=155.0,
        )
    )
    # Warm call to measure steady-state inference latency
    _ = predictor.predict(state.vector, include_shap=True)
    pred = predictor.predict(state.vector, include_shap=True)

    assert 0.0 <= pred.risk_score_pct <= 100.0
    assert pred.latency_ms < 200.0
    assert len(pred.top_factors) == 6
    assert pred.top_factors[0].direction in ("increases_risk", "decreases_risk")
