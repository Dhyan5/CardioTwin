"""
Medication & Pharmacodynamics Engine (/pharmaco).

Models 8 cardiovascular drug classes with dose-proportional pharmacodynamic effects
on the Digital Twin State Vector, explicit onset kinetics (fast vs. slow-onset),
physiologically plausible clamping, and clinically significant drug-drug and
drug-state interaction terms.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, Field

from backend.twin.fusion import (
    clamp_field,
    compute_cardiac_output,
    compute_egfr_ckd_epi_2021,
    evaluate_all_statuses,
)
from backend.twin.schema import TwinVector


class MedicationPlan(BaseModel):
    """Daily medication dosing regimen across the 8 modeled cardiovascular drug classes."""

    metoprolol: float = Field(default=0.0, ge=0.0, le=200.0, description="Beta-blocker (mg/day)")
    lisinopril: float = Field(default=0.0, ge=0.0, le=40.0, description="ACE inhibitor (mg/day)")
    furosemide: float = Field(default=0.0, ge=0.0, le=80.0, description="Loop diuretic (mg/day)")
    atorvastatin: float = Field(default=0.0, ge=0.0, le=80.0, description="Statin (mg/day)")
    amlodipine: float = Field(default=0.0, ge=0.0, le=10.0, description="Calcium channel blocker (mg/day)")
    digoxin: float = Field(default=0.0, ge=0.0, le=0.25, description="Cardiac glycoside (mg/day)")
    isosorbide_mononitrate: float = Field(default=0.0, ge=0.0, le=120.0, description="Nitrate (mg/day)")
    aspirin: float = Field(default=0.0, ge=0.0, le=325.0, description="Antiplatelet (mg/day)")


class RiskFlag(BaseModel):
    """Clinical safety or interaction warning triggered by a medication plan."""

    code: str
    severity: Literal["caution", "critical"]
    drug: str
    message: str


class PharmacoAssessment(BaseModel):
    """Result of applying a MedicationPlan to a baseline TwinVector."""

    projected_vector: TwinVector
    deltas: Dict[str, float]
    mace_risk_multiplier: float = Field(
        default=1.0,
        description="Direct antithrombotic/pleiotropic MACE risk multiplier (e.g. from Aspirin)",
    )
    risk_flags: List[RiskFlag] = Field(default_factory=list)


DRUG_CLASSES: Dict[str, Dict[str, object]] = {
    "metoprolol": {"class": "Beta-blocker", "unit": "mg/day", "max_dose": 200.0, "onset": "fast"},
    "lisinopril": {"class": "ACE inhibitor", "unit": "mg/day", "max_dose": 40.0, "onset": "slow"},
    "furosemide": {"class": "Loop diuretic", "unit": "mg/day", "max_dose": 80.0, "onset": "fast"},
    "atorvastatin": {"class": "Statin", "unit": "mg/day", "max_dose": 80.0, "onset": "slow"},
    "amlodipine": {"class": "Calcium channel blocker", "unit": "mg/day", "max_dose": 10.0, "onset": "fast"},
    "digoxin": {"class": "Cardiac glycoside", "unit": "mg/day", "max_dose": 0.25, "onset": "fast"},
    "isosorbide_mononitrate": {"class": "Nitrate", "unit": "mg/day", "max_dose": 120.0, "onset": "fast"},
    "aspirin": {"class": "Antiplatelet", "unit": "mg/day", "max_dose": 325.0, "onset": "fast"},
}


def _hill_response(dose: float, max_dose: float, hill: float = 0.85) -> float:
    """
    Saturating Emax-style normalized dose-response curve in [0, 1].
    Produces realistic diminishing marginal returns near maximum daily dose.
    """
    if dose <= 0.0 or max_dose <= 0.0:
        return 0.0
    ratio = min(1.0, dose / max_dose)
    return (ratio ** hill) / (0.35 + 0.65 * (ratio ** hill))


# ---------------------------------------------------------------------------
# Individual Drug-Class Pharmacodynamic Effect Functions
# ---------------------------------------------------------------------------


def effect_beta_blocker(dose_mg: float, fast_weight: float, slow_weight: float) -> Dict[str, float]:
    """
    Beta-blocker (Metoprolol, 0-200 mg/day) — Fast-onset sympathetic blockade.
    - Decreases heart rate (up to -18 bpm at 200 mg/day) via beta-1 SA/AV node blockade.
    - Decreases systolic/diastolic BP (up to -11 / -6 mmHg) via reduced cardiac output & renin.
    - Mild negative inotropy acutely (-1.5% EF at high dose), offset slightly by chronic remodeling.
    """
    r = _hill_response(dose_mg, 200.0)
    return {
        "heart_rate": -18.0 * r * fast_weight,
        "systolic_bp": -11.0 * r * fast_weight,
        "diastolic_bp": -6.0 * r * fast_weight,
        "ejection_fraction": (-1.5 * r * fast_weight) + (1.2 * r * slow_weight),
    }


def effect_ace_inhibitor(
    dose_mg: float,
    baseline_ef: float,
    fast_weight: float,
    slow_weight: float,
) -> Dict[str, float]:
    """
    ACE inhibitor (Lisinopril, 0-40 mg/day) — Mixed onset (BP fast, remodeling slow).
    - Lowers afterload & BP (up to -14 / -8 mmHg) via angiotensin II suppression.
    - Slow-onset reverse LV remodeling over weeks: improves EF (+4.5% when EF < 50%, +1.5% otherwise).
    - Reduces BNP via afterload/preload unloading (up to -22% of excess BNP, capped).
    - Increases serum potassium (+0.45 mEq/L) via aldosterone suppression.
    - Can mildly elevate creatinine (+0.18 mg/dL) via efferent arteriolar dilation.
    """
    r = _hill_response(dose_mg, 40.0)
    ef_gain_potential = 4.8 if baseline_ef < 50.0 else 1.5
    return {
        "systolic_bp": -14.0 * r * (0.65 * fast_weight + 0.35 * slow_weight),
        "diastolic_bp": -8.0 * r * (0.65 * fast_weight + 0.35 * slow_weight),
        "ejection_fraction": ef_gain_potential * r * slow_weight,
        "bnp": -65.0 * r * slow_weight,
        "potassium": 0.45 * r * (0.5 * fast_weight + 0.5 * slow_weight),
        "creatinine": 0.18 * r * (0.5 * fast_weight + 0.5 * slow_weight),
    }


def effect_loop_diuretic(
    dose_mg: float,
    baseline_bnp: float,
    fast_weight: float,
    slow_weight: float,
) -> Dict[str, float]:
    """
    Loop diuretic (Furosemide, 0-80 mg/day) — Fast-onset natriuresis & preload reduction.
    - Reduces intravascular volume, preload, and BP (up to -10 / -5 mmHg).
    - Markedly lowers BNP proportional to baseline volume overload (up to -38% of BNP).
    - Wastes potassium (-0.65 mEq/L at 80 mg/day) -> hypokalemia risk.
    - Mild prerenal creatinine rise (+0.16 mg/dL) from volume contraction -> dehydration risk.
    """
    r = _hill_response(dose_mg, 80.0)
    bnp_drop = -min(260.0, max(25.0, baseline_bnp * 0.38)) * r * fast_weight
    return {
        "systolic_bp": -10.0 * r * fast_weight,
        "diastolic_bp": -5.0 * r * fast_weight,
        "bnp": bnp_drop,
        "potassium": -0.65 * r * fast_weight,
        "creatinine": 0.16 * r * (0.7 * fast_weight + 0.3 * slow_weight),
    }


def effect_statin(
    dose_mg: float,
    baseline_ldl: float,
    fast_weight: float,
    slow_weight: float,
) -> Dict[str, float]:
    """
    Statin (Atorvastatin, 0-80 mg/day) — Slow-onset hepatic HMG-CoA reductase inhibition.
    - Reduces LDL cholesterol by up to ~48% of baseline over weeks (slow-onset).
    - Modestly raises HDL (+4.5 mg/dL) and reduces triglycerides (-18 mg/dL).
    """
    r = _hill_response(dose_mg, 80.0)
    ldl_reduction = -min(130.0, baseline_ldl * 0.48) * r * slow_weight
    return {
        "ldl": ldl_reduction,
        "hdl": 4.5 * r * slow_weight,
        "triglycerides": -18.0 * r * slow_weight,
    }


def effect_calcium_channel_blocker(dose_mg: float, fast_weight: float, slow_weight: float) -> Dict[str, float]:
    """
    Dihydropyridine Calcium Channel Blocker (Amlodipine, 0-10 mg/day) — Fast-onset vasodilation.
    - Decreases systemic vascular resistance and BP (up to -15 / -9 mmHg).
    - Slight reflex heart rate increase (+2.5 bpm) unless blunted by beta-blocker.
    """
    r = _hill_response(dose_mg, 10.0)
    return {
        "systolic_bp": -15.0 * r * fast_weight,
        "diastolic_bp": -9.0 * r * fast_weight,
        "heart_rate": 2.5 * r * fast_weight,
    }


def effect_cardiac_glycoside(dose_mg: float, fast_weight: float, slow_weight: float) -> Dict[str, float]:
    """
    Cardiac glycoside (Digoxin, 0-0.25 mg/day) — Fast-onset positive inotrope & vagotonic.
    - Increases myocardial contractility via Na+/K+-ATPase inhibition (+4.5% EF).
    - Slows AV nodal conduction (-6 bpm HR).
    - Narrow therapeutic index: toxicity risk rises sharply with hypokalemia or low eGFR.
    """
    r = _hill_response(dose_mg, 0.25)
    return {
        "ejection_fraction": 4.5 * r * fast_weight,
        "heart_rate": -6.0 * r * fast_weight,
    }


def effect_nitrate(dose_mg: float, fast_weight: float, slow_weight: float) -> Dict[str, float]:
    """
    Organic Nitrate (Isosorbide mononitrate, 0-120 mg/day) — Fast-onset venodilation.
    - Reduces venous return / preload and lowers BP (up to -12 / -6 mmHg).
    - Mildly reduces BNP (-30 pg/mL) via preload unloading.
    - Additive/synergistic hypotension when combined with other vasodilators.
    """
    r = _hill_response(dose_mg, 120.0)
    return {
        "systolic_bp": -12.0 * r * fast_weight,
        "diastolic_bp": -6.0 * r * fast_weight,
        "bnp": -30.0 * r * fast_weight,
    }


def effect_antiplatelet(dose_mg: float, fast_weight: float) -> float:
    """
    Antiplatelet (Aspirin, 0-325 mg/day) — Fast-onset COX-1 inhibition.
    - Reduces thrombotic 30-day MACE risk by up to ~14% (multiplier down to 0.86),
      plateauing around 81-162 mg/day; higher doses (325 mg) add bleeding risk
      without additional antithrombotic benefit.
    """
    if dose_mg <= 0.0:
        return 1.0
    # Plateaus near 100 mg/day
    effective_ratio = min(1.0, dose_mg / 100.0)
    reduction = 0.14 * math.sqrt(effective_ratio) * fast_weight
    return round(max(0.84, 1.0 - reduction), 4)


# ---------------------------------------------------------------------------
# Unified Plan Application + Drug-Drug / Drug-State Interaction Engine
# ---------------------------------------------------------------------------


def apply_medication_plan(
    baseline: TwinVector,
    plan: MedicationPlan,
    fast_weight: float = 1.0,
    slow_weight: float = 1.0,
) -> PharmacoAssessment:
    """
    Apply a MedicationPlan to a baseline TwinVector with configurable fast/slow
    onset weights (both 1.0 for steady-state evaluation; ramped in /whatif).

    Returns the projected TwinVector (clamped to physiological bounds), per-field
    deltas, the antithrombotic MACE risk multiplier, and all triggered clinical
    risk flags.
    """
    fast_weight = max(0.0, min(1.0, fast_weight))
    slow_weight = max(0.0, min(1.0, slow_weight))

    effects: List[Dict[str, float]] = [
        effect_beta_blocker(plan.metoprolol, fast_weight, slow_weight),
        effect_ace_inhibitor(plan.lisinopril, baseline.ejection_fraction, fast_weight, slow_weight),
        effect_loop_diuretic(plan.furosemide, baseline.bnp, fast_weight, slow_weight),
        effect_statin(plan.atorvastatin, baseline.ldl, fast_weight, slow_weight),
        effect_calcium_channel_blocker(plan.amlodipine, fast_weight, slow_weight),
        effect_cardiac_glycoside(plan.digoxin, fast_weight, slow_weight),
        effect_nitrate(plan.isosorbide_mononitrate, fast_weight, slow_weight),
    ]

    # Sum additive deltas across all drugs
    total_delta: Dict[str, float] = {}
    for eff in effects:
        for k, v in eff.items():
            total_delta[k] = total_delta.get(k, 0.0) + v

    # Explicit Interaction Term 1: Beta-blocker blunts CCB reflex tachycardia
    if plan.metoprolol > 0 and plan.amlodipine > 0:
        bb_ratio = _hill_response(plan.metoprolol, 200.0)
        ccb_ratio = _hill_response(plan.amlodipine, 10.0)
        total_delta["heart_rate"] = total_delta.get("heart_rate", 0.0) - (2.2 * bb_ratio * ccb_ratio * fast_weight)

    # Explicit Interaction Term 2: Vasodilator synergy (Nitrate + CCB and/or ACEi)
    if plan.isosorbide_mononitrate > 0 and (plan.amlodipine > 0 or plan.lisinopril > 0):
        nit_r = _hill_response(plan.isosorbide_mononitrate, 120.0)
        other_r = max(_hill_response(plan.amlodipine, 10.0), _hill_response(plan.lisinopril, 40.0))
        synergy_drop = 6.5 * nit_r * other_r * fast_weight
        total_delta["systolic_bp"] = total_delta.get("systolic_bp", 0.0) - synergy_drop
        total_delta["diastolic_bp"] = total_delta.get("diastolic_bp", 0.0) - (0.5 * synergy_drop)

    # Apply deltas and clamp to physiological bounds
    data = baseline.model_dump()
    mutable_fields = (
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "ejection_fraction",
        "ldl",
        "hdl",
        "triglycerides",
        "bnp",
        "potassium",
        "creatinine",
    )
    for field in mutable_fields:
        new_val = float(data[field]) + total_delta.get(field, 0.0)
        data[field] = round(clamp_field(field, new_val), 2)

    # Ensure DBP < SBP - 10
    if data["diastolic_bp"] >= data["systolic_bp"] - 10.0:
        data["diastolic_bp"] = round(max(35.0, data["systolic_bp"] - 20.0), 1)

    # Recompute derived renal & hemodynamic outputs
    data["egfr"] = compute_egfr_ckd_epi_2021(
        creatinine_mg_dl=data["creatinine"],
        age=int(data["age"]),
        sex=data["sex"],
    )
    data["cardiac_output"] = compute_cardiac_output(
        ejection_fraction=data["ejection_fraction"],
        heart_rate=data["heart_rate"],
        bmi=data["bmi"],
        sex=data["sex"],
    )

    projected = TwinVector(**data)
    base_dict = baseline.model_dump()
    proj_dict = projected.model_dump()
    deltas = {
        k: round(float(proj_dict[k]) - float(base_dict[k]), 3)
        for k in (*mutable_fields, "egfr", "cardiac_output")
    }

    mace_mult = effect_antiplatelet(plan.aspirin, fast_weight)
    flags = _evaluate_risk_flags(projected, plan)

    return PharmacoAssessment(
        projected_vector=projected,
        deltas=deltas,
        mace_risk_multiplier=mace_mult,
        risk_flags=flags,
    )


def _evaluate_risk_flags(vec: TwinVector, plan: MedicationPlan) -> List[RiskFlag]:
    """Generate clinical safety and drug-interaction alerts."""
    flags: List[RiskFlag] = []

    # Bradycardia from beta-blocker (+ digoxin synergy)
    if plan.metoprolol > 0 and vec.heart_rate < 55:
        sev = "critical" if vec.heart_rate < 48 else "caution"
        flags.append(
            RiskFlag(
                code="BRADYCARDIA",
                severity=sev,
                drug="Metoprolol" + (" + Digoxin" if plan.digoxin > 0 else ""),
                message=f"Projected HR {vec.heart_rate:.0f} bpm (<55 bpm). Monitor for symptomatic bradycardia.",
            )
        )

    # Hypotension / Vasodilator synergy
    if vec.systolic_bp < 92:
        vasodilators = [
            name
            for name, dose in [
                ("Nitrate", plan.isosorbide_mononitrate),
                ("Amlodipine", plan.amlodipine),
                ("Lisinopril", plan.lisinopril),
                ("Furosemide", plan.furosemide),
            ]
            if dose > 0
        ]
        if vasodilators:
            flags.append(
                RiskFlag(
                    code="HYPOTENSION_SYNERGY" if len(vasodilators) > 1 else "HYPOTENSION",
                    severity="critical" if vec.systolic_bp < 85 else "caution",
                    drug=" + ".join(vasodilators),
                    message=f"Projected SBP {vec.systolic_bp:.0f} mmHg (<92 mmHg) with vasodilator/diuretic regimen.",
                )
            )

    # Hyperkalemia & Renal impairment (ACEi)
    if plan.lisinopril > 0 and vec.potassium > 5.0:
        flags.append(
            RiskFlag(
                code="HYPERKALEMIA",
                severity="critical" if vec.potassium >= 5.5 else "caution",
                drug="Lisinopril",
                message=f"Projected K+ {vec.potassium:.2f} mEq/L (>5.0). Monitor serum potassium.",
            )
        )
    if plan.lisinopril > 0 and vec.egfr < 40:
        flags.append(
            RiskFlag(
                code="RENAL_IMPAIRMENT",
                severity="critical" if vec.egfr < 30 else "caution",
                drug="Lisinopril",
                message=f"Projected eGFR {vec.egfr:.1f} mL/min/1.73m2. Assess renal perfusion.",
            )
        )

    # Hypokalemia & Dehydration (Loop diuretic)
    if plan.furosemide > 0 and vec.potassium < 3.5:
        flags.append(
            RiskFlag(
                code="HYPOKALEMIA",
                severity="critical" if vec.potassium < 3.1 else "caution",
                drug="Furosemide",
                message=f"Projected K+ {vec.potassium:.2f} mEq/L (<3.5). Consider K+ supplementation.",
            )
        )

    # Digoxin narrow therapeutic index interactions
    if plan.digoxin > 0:
        if vec.potassium < 3.5:
            flags.append(
                RiskFlag(
                    code="DIGOXIN_TOXICITY_HYPOKALEMIA",
                    severity="critical",
                    drug="Digoxin",
                    message=f"High digoxin toxicity risk: hypokalemia (K+ {vec.potassium:.2f} mEq/L) sensitizes myocardium.",
                )
            )
        if vec.egfr < 45:
            flags.append(
                RiskFlag(
                    code="DIGOXIN_RENAL_ACCUMULATION",
                    severity="critical",
                    drug="Digoxin",
                    message=f"Digoxin renal clearance impaired (eGFR {vec.egfr:.1f}). Reduce dose or monitor trough levels.",
                )
            )
        if plan.digoxin >= 0.2 and vec.potassium >= 3.5 and vec.egfr >= 45:
            flags.append(
                RiskFlag(
                    code="DIGOXIN_NARROW_INDEX",
                    severity="caution",
                    drug="Digoxin",
                    message="High-range digoxin dose (>=0.2 mg/day); verify serum digoxin concentration.",
                )
            )

    # Amlodipine peripheral edema
    if plan.amlodipine >= 7.5:
        flags.append(
            RiskFlag(
                code="PERIPHERAL_EDEMA",
                severity="caution",
                drug="Amlodipine",
                message="Dose >=7.5 mg/day carries elevated risk of dose-dependent peripheral edema.",
            )
        )

    # Atorvastatin high-intensity caution
    if plan.atorvastatin >= 80.0:
        flags.append(
            RiskFlag(
                code="STATIN_HIGH_INTENSITY",
                severity="caution",
                drug="Atorvastatin",
                message="High-intensity 80 mg/day dose; monitor for myalgia/CK and LFT elevation.",
            )
        )

    # Aspirin bleeding risk
    if plan.aspirin > 162.0:
        flags.append(
            RiskFlag(
                code="BLEEDING_RISK",
                severity="caution",
                drug="Aspirin",
                message=f"Aspirin {plan.aspirin:.0f} mg/day (>162 mg) increases GI bleeding risk without added CV benefit.",
            )
        )

    return flags
