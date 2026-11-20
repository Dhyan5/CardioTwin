"""
What-If Medication & Longitudinal Outcome Simulator (/whatif).

Projects a patient's Digital Twin State Vector and 30-day MACE risk forward
in weekly steps over a configurable horizon (default 90 days):
- Fast-onset drug effects are applied immediately from Week 1 (t > 0).
- Slow-onset drug effects (e.g. statin LDL reduction, ACEi LV remodeling)
  ramp in via a saturating exponential: 1 - exp(-t / tau), with tau = 35 days (5 weeks).
- At each weekly step, the evolving trajectory window is scored by the
  Hybrid XGBoost-LSTM model to produce a projected MACE risk curve.
- Supports side-by-side comparison of multiple plans (e.g. No Treatment vs.
  Current Plan vs. Proposed Plan) with a chart-ready time series and a Day-90 delta table.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from backend.models.inference import HybridMacePredictor
from backend.pharmaco.engine import MedicationPlan, RiskFlag, apply_medication_plan
from backend.twin.schema import TwinVector

SLOW_ONSET_TAU_DAYS = 35.0  # ~5-week saturating exponential time constant


class NamedMedicationPlan(BaseModel):
    plan_id: str
    label: str
    medications: MedicationPlan


class SimulationPoint(BaseModel):
    week: int
    day: int
    risk_pct: float
    systolic_bp: float
    diastolic_bp: float
    heart_rate: float
    ejection_fraction: float
    cardiac_output: float
    ldl: float
    bnp: float
    potassium: float
    egfr: float


class PlanSimulationResult(BaseModel):
    plan_id: str
    label: str
    medications: MedicationPlan
    series: List[SimulationPoint]
    final_vector: TwinVector
    final_risk_pct: float
    risk_delta_pct: float
    risk_flags: List[RiskFlag]


class Day90DeltaRow(BaseModel):
    parameter: str
    label: str
    unit: str
    baseline: float
    plans: Dict[str, Dict[str, float]]  # plan_id -> {"value": ..., "delta": ...}


class WhatIfComparisonResponse(BaseModel):
    patient_id: str
    horizon_days: int
    baseline_risk_pct: float
    plans: List[PlanSimulationResult]
    delta_table: List[Day90DeltaRow]


class WhatIfSimulator:
    """Projects TwinVector + 30-day MACE risk across competing medication plans."""

    def __init__(self, predictor: Optional[HybridMacePredictor] = None):
        self.predictor = predictor or HybridMacePredictor()

    def simulate_plan(
        self,
        baseline: TwinVector,
        plan_spec: NamedMedicationPlan,
        history: Optional[Sequence[TwinVector]] = None,
        horizon_days: int = 90,
    ) -> PlanSimulationResult:
        self.predictor.load()
        base_history = list(history) if history else [baseline]

        # Build weekly day markers: 0, 7, 14, ..., horizon_days
        days: List[int] = list(range(0, horizon_days, 7))
        if not days or days[-1] != horizon_days:
            days.append(horizon_days)

        series: List[SimulationPoint] = []
        rolling_history = list(base_history)
        final_assessment = None

        for week_idx, day in enumerate(days):
            if day == 0:
                fast_w = 0.0
                slow_w = 0.0
            else:
                fast_w = 1.0
                slow_w = 1.0 - math.exp(-float(day) / SLOW_ONSET_TAU_DAYS)

            assessment = apply_medication_plan(
                baseline=baseline,
                plan=plan_spec.medications,
                fast_weight=fast_w,
                slow_weight=slow_w,
            )
            vec = assessment.projected_vector
            if day > 0:
                rolling_history.append(vec)

            pred = self.predictor.predict(
                current_vector=vec,
                history=rolling_history,
                mace_risk_multiplier=assessment.mace_risk_multiplier,
                include_shap=False,
            )

            series.append(
                SimulationPoint(
                    week=week_idx,
                    day=day,
                    risk_pct=pred.risk_score_pct,
                    systolic_bp=vec.systolic_bp,
                    diastolic_bp=vec.diastolic_bp,
                    heart_rate=vec.heart_rate,
                    ejection_fraction=vec.ejection_fraction,
                    cardiac_output=vec.cardiac_output,
                    ldl=vec.ldl,
                    bnp=vec.bnp,
                    potassium=vec.potassium,
                    egfr=vec.egfr,
                )
            )
            final_assessment = assessment

        baseline_risk = series[0].risk_pct
        final_risk = series[-1].risk_pct

        return PlanSimulationResult(
            plan_id=plan_spec.plan_id,
            label=plan_spec.label,
            medications=plan_spec.medications,
            series=series,
            final_vector=final_assessment.projected_vector,
            final_risk_pct=final_risk,
            risk_delta_pct=round(final_risk - baseline_risk, 2),
            risk_flags=final_assessment.risk_flags,
        )

    def compare_plans(
        self,
        patient_id: str,
        baseline: TwinVector,
        plans: Sequence[NamedMedicationPlan],
        history: Optional[Sequence[TwinVector]] = None,
        horizon_days: int = 90,
    ) -> WhatIfComparisonResponse:
        results = [
            self.simulate_plan(baseline=baseline, plan_spec=p, history=history, horizon_days=horizon_days)
            for p in plans
        ]
        baseline_risk_pct = results[0].series[0].risk_pct if results else 0.0

        delta_specs = [
            ("risk_pct", "30-Day MACE Risk", "%"),
            ("ejection_fraction", "Ejection Fraction", "%"),
            ("ldl", "LDL Cholesterol", "mg/dL"),
            ("bnp", "BNP", "pg/mL"),
            ("systolic_bp", "Systolic BP", "mmHg"),
            ("diastolic_bp", "Diastolic BP", "mmHg"),
            ("heart_rate", "Heart Rate", "bpm"),
            ("potassium", "Serum Potassium", "mEq/L"),
            ("egfr", "eGFR", "mL/min/1.73m2"),
        ]

        delta_table: List[Day90DeltaRow] = []
        for param, label, unit in delta_specs:
            base_val = getattr(results[0].series[0], param) if results else 0.0
            plan_cells: Dict[str, Dict[str, float]] = {}
            for res in results:
                end_val = getattr(res.series[-1], param)
                plan_cells[res.plan_id] = {
                    "value": round(float(end_val), 2),
                    "delta": round(float(end_val) - float(base_val), 2),
                }
            delta_table.append(
                Day90DeltaRow(
                    parameter=param,
                    label=label,
                    unit=unit,
                    baseline=round(float(base_val), 2),
                    plans=plan_cells,
                )
            )

        return WhatIfComparisonResponse(
            patient_id=patient_id,
            horizon_days=horizon_days,
            baseline_risk_pct=baseline_risk_pct,
            plans=results,
            delta_table=delta_table,
        )
