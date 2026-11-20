"""Unit tests for the /pharmaco medication & pharmacodynamics engine."""

from backend.pharmaco.engine import MedicationPlan, apply_medication_plan
from backend.twin.fusion import TwinFusionEngine
from backend.twin.schema import TwinIngestPayload, TwinVector


def _baseline_vector(**overrides) -> TwinVector:
    base = dict(
        patient_id="test-pt",
        age=64,
        sex="M",
        bmi=28.0,
        smoking_status="former",
        diabetes_status=True,
        family_history=True,
        heart_rate=82.0,
        systolic_bp=148.0,
        diastolic_bp=92.0,
        ejection_fraction=40.0,
        o2_saturation=95.0,
        ldl=155.0,
        hdl=40.0,
        triglycerides=190.0,
        bnp=520.0,
        troponin=0.03,
        potassium=4.1,
        creatinine=1.2,
    )
    base.update(overrides)
    return TwinFusionEngine.fuse(TwinIngestPayload(**base)).vector


def test_zero_dose_produces_no_change():
    vec = _baseline_vector()
    res = apply_medication_plan(vec, MedicationPlan())
    assert res.projected_vector.heart_rate == vec.heart_rate
    assert res.projected_vector.systolic_bp == vec.systolic_bp
    assert res.projected_vector.ldl == vec.ldl
    assert res.mace_risk_multiplier == 1.0
    assert res.risk_flags == []


def test_dose_proportionality_beta_blocker_and_bradycardia_flag():
    vec = _baseline_vector(heart_rate=64.0)
    low = apply_medication_plan(vec, MedicationPlan(metoprolol=25.0))
    high = apply_medication_plan(vec, MedicationPlan(metoprolol=200.0))

    assert high.projected_vector.heart_rate < low.projected_vector.heart_rate < vec.heart_rate
    assert high.projected_vector.systolic_bp < low.projected_vector.systolic_bp < vec.systolic_bp
    assert any(f.code == "BRADYCARDIA" for f in high.risk_flags)


def test_statin_slow_onset_ramping():
    vec = _baseline_vector(ldl=170.0)
    # At t=0/immediate (slow_weight=0), statin has not lowered LDL yet
    immediate = apply_medication_plan(vec, MedicationPlan(atorvastatin=80.0), fast_weight=1.0, slow_weight=0.0)
    steady = apply_medication_plan(vec, MedicationPlan(atorvastatin=80.0), fast_weight=1.0, slow_weight=1.0)

    assert immediate.projected_vector.ldl == vec.ldl
    assert steady.projected_vector.ldl < vec.ldl - 40.0
    assert steady.projected_vector.hdl > vec.hdl
    assert any(f.code == "STATIN_HIGH_INTENSITY" for f in steady.risk_flags)


def test_ace_inhibitor_remodeling_and_potassium_renal():
    vec = _baseline_vector(ejection_fraction=38.0, potassium=4.8, creatinine=1.9)
    res = apply_medication_plan(vec, MedicationPlan(lisinopril=40.0), fast_weight=1.0, slow_weight=1.0)

    assert res.projected_vector.systolic_bp < vec.systolic_bp
    assert res.projected_vector.ejection_fraction > vec.ejection_fraction
    assert res.projected_vector.potassium > vec.potassium
    assert any(f.code == "HYPERKALEMIA" for f in res.risk_flags)
    assert any(f.code == "RENAL_IMPAIRMENT" for f in res.risk_flags)


def test_loop_diuretic_bnp_and_hypokalemia():
    vec = _baseline_vector(bnp=600.0, potassium=3.6)
    res = apply_medication_plan(vec, MedicationPlan(furosemide=80.0))

    assert res.projected_vector.bnp < vec.bnp - 100.0
    assert res.projected_vector.potassium < 3.5
    assert any(f.code == "HYPOKALEMIA" for f in res.risk_flags)


def test_digoxin_toxicity_with_hypokalemia_and_low_egfr():
    vec = _baseline_vector(potassium=3.3, creatinine=2.4, ejection_fraction=35.0)
    res = apply_medication_plan(vec, MedicationPlan(digoxin=0.25))

    assert res.projected_vector.ejection_fraction > vec.ejection_fraction
    codes = {f.code for f in res.risk_flags}
    assert "DIGOXIN_TOXICITY_HYPOKALEMIA" in codes
    assert "DIGOXIN_RENAL_ACCUMULATION" in codes


def test_vasodilator_synergy_nitrate_plus_amlodipine():
    vec = _baseline_vector(systolic_bp=105.0, diastolic_bp=68.0)
    nitrate_only = apply_medication_plan(vec, MedicationPlan(isosorbide_mononitrate=120.0))
    combo = apply_medication_plan(vec, MedicationPlan(isosorbide_mononitrate=120.0, amlodipine=10.0))

    assert combo.projected_vector.systolic_bp < nitrate_only.projected_vector.systolic_bp
    assert any(f.code == "HYPOTENSION_SYNERGY" for f in combo.risk_flags)
    # Clamped above physiological minimum
    assert combo.projected_vector.systolic_bp >= 60.0


def test_antiplatelet_mace_multiplier_and_bleeding_flag():
    vec = _baseline_vector()
    low_asa = apply_medication_plan(vec, MedicationPlan(aspirin=81.0))
    high_asa = apply_medication_plan(vec, MedicationPlan(aspirin=325.0))

    assert 0.84 <= low_asa.mace_risk_multiplier < 1.0
    assert not any(f.code == "BLEEDING_RISK" for f in low_asa.risk_flags)
    assert any(f.code == "BLEEDING_RISK" for f in high_asa.risk_flags)
