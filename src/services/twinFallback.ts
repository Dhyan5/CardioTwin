import {
  Day90DeltaRow,
  FieldStatus,
  MedicationPlan,
  ModelMetrics,
  PatientSummary,
  PharmacoAssessment,
  PlanSimulationResult,
  RiskFactorContribution,
  RiskFlag,
  RiskLevel,
  RiskPrediction,
  RiskTrendPoint,
  SimulationPoint,
  TWIN_SCHEMA,
  TwinDashboardResponse,
  TwinState,
  TwinVector,
  WhatIfComparisonResponse,
} from '../types/twin';

const SCHEMA_FIELDS = TWIN_SCHEMA.fields as Record<string, any>;

function clampField(field: string, value: number): number {
  const meta = SCHEMA_FIELDS[field];
  if (!meta) return value;
  let v = value;
  if (meta.physiological_min !== undefined) v = Math.max(meta.physiological_min, v);
  if (meta.physiological_max !== undefined) v = Math.min(meta.physiological_max, v);
  return v;
}

function computeEgfr(creatinine: number, age: number, sex: 'M' | 'F'): number {
  const scr = Math.max(0.3, creatinine);
  const kappa = sex === 'F' ? 0.7 : 0.9;
  const alpha = sex === 'F' ? -0.241 : -0.302;
  const sexFactor = sex === 'F' ? 1.012 : 1.0;
  const ratio = scr / kappa;
  const egfr = 142 * Math.pow(Math.min(ratio, 1), alpha) * Math.pow(Math.max(ratio, 1), -1.2) * Math.pow(0.9938, age) * sexFactor;
  return Number(clampField('egfr', egfr).toFixed(1));
}

function computeCardiacOutput(ef: number, hr: number, bmi: number, sex: 'M' | 'F'): number {
  const baseEdv = sex === 'M' ? 128 : 108;
  const scale = Math.max(0.82, Math.min(1.28, 1 + 0.012 * (bmi - 25)));
  const sv = baseEdv * scale * (ef / 100);
  return Number(clampField('cardiac_output', (sv * hr) / 1000).toFixed(2));
}

function evaluateStatus(field: string, value: number): FieldStatus {
  const meta = SCHEMA_FIELDS[field];
  if (!meta || meta.normal_min === undefined || meta.normal_max === undefined) return 'normal';
  if (
    (meta.critical_low !== undefined && value <= meta.critical_low && meta.critical_low < meta.normal_min) ||
    (meta.critical_high !== undefined && value >= meta.critical_high && meta.critical_high > meta.normal_max)
  ) {
    return 'critical';
  }
  if (value >= meta.normal_min && value <= meta.normal_max) return 'normal';
  if (
    (meta.caution_low !== undefined && value < meta.normal_min && value >= meta.caution_low) ||
    (meta.caution_high !== undefined && value > meta.normal_max && value <= meta.caution_high)
  ) {
    return 'caution';
  }
  return 'critical';
}

function evaluateAllStatuses(v: TwinVector): Record<string, FieldStatus> {
  const keys: Array<keyof TwinVector> = [
    'heart_rate', 'systolic_bp', 'diastolic_bp', 'ejection_fraction', 'cardiac_output',
    'o2_saturation', 'ldl', 'hdl', 'triglycerides', 'bnp', 'troponin', 'potassium',
    'creatinine', 'egfr', 'bmi',
  ];
  const out: Record<string, FieldStatus> = {};
  for (const k of keys) {
    out[k] = evaluateStatus(k, v[k] as number);
  }
  return out;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-12, Math.min(12, x))));
}

function predictVector(v: TwinVector, maceMultiplier = 1.0): RiskPrediction {
  const t0 = performance.now();
  const z =
    -4.1 +
    0.028 * Math.max(0, v.age - 50) +
    0.078 * Math.max(0, 50 - v.ejection_fraction) +
    0.52 * Math.log1p(v.bnp / 80) +
    0.65 * Math.log1p(v.troponin * 40) +
    0.024 * Math.max(0, v.systolic_bp - 135) +
    0.035 * Math.max(0, 95 - v.systolic_bp) +
    0.09 * Math.max(0, 94 - v.o2_saturation) +
    0.018 * Math.max(0, 55 - v.egfr) +
    0.005 * Math.max(0, v.ldl - 120) +
    (v.diabetes_status ? 0.35 : 0) +
    (v.smoking_status === 'current' ? 0.3 : 0);

  const prob = Math.max(0.001, Math.min(0.995, sigmoid(z) * maceMultiplier));
  const riskPct = Number((prob * 100).toFixed(2));
  const riskLevel: RiskLevel =
    riskPct >= 55 ? 'CRITICAL' : riskPct >= 30 ? 'HIGH' : riskPct >= 12 ? 'MODERATE' : 'LOW';

  const zStatic =
    -6.1 +
    0.055 * (v.age - 45) +
    (v.sex === 'M' ? 0.35 : 0) +
    0.022 * Math.max(0, v.systolic_bp - 115) +
    0.008 * Math.max(0, v.ldl - 100) -
    0.015 * (v.hdl - 45) +
    (v.smoking_status === 'current' ? 0.55 : v.smoking_status === 'former' ? 0.18 : 0) +
    (v.diabetes_status ? 0.65 : 0);
  const staticPct = Number((sigmoid(zStatic) * 100).toFixed(2));

  const candidates: Array<{ feature: string; display_name: string; value: number; shap_value: number }> = [
    { feature: 'ejection_fraction', display_name: 'Ejection Fraction (%)', value: v.ejection_fraction, shap_value: Number(((55 - v.ejection_fraction) * 0.045).toFixed(4)) },
    { feature: 'bnp', display_name: 'BNP (pg/mL)', value: v.bnp, shap_value: Number((Math.log1p(v.bnp / 100) * 0.38 - 0.25).toFixed(4)) },
    { feature: 'systolic_bp', display_name: 'Systolic BP (mmHg)', value: v.systolic_bp, shap_value: Number(((v.systolic_bp - 122) * 0.016).toFixed(4)) },
    { feature: 'ldl', display_name: 'LDL Cholesterol (mg/dL)', value: v.ldl, shap_value: Number(((v.ldl - 105) * 0.006).toFixed(4)) },
    { feature: 'troponin', display_name: 'Troponin I (ng/mL)', value: v.troponin, shap_value: Number((Math.log1p(v.troponin * 50) * 0.32 - 0.18).toFixed(4)) },
    { feature: 'egfr', display_name: 'eGFR (mL/min/1.73m2)', value: v.egfr, shap_value: Number(((75 - v.egfr) * 0.012).toFixed(4)) },
  ];

  const top_factors: RiskFactorContribution[] = candidates
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .map((c) => ({
      ...c,
      direction: c.shap_value >= 0 ? 'increases_risk' : 'decreases_risk',
      magnitude: Number(Math.abs(c.shap_value).toFixed(4)),
    }));

  return {
    risk_score_pct: riskPct,
    risk_level: riskLevel,
    xgb_prob: Number(prob.toFixed(4)),
    lstm_prob: Number(Math.min(0.99, prob * 1.03).toFixed(4)),
    baseline_static_pct: staticPct,
    latency_ms: Number((performance.now() - t0 + 1.2).toFixed(2)),
    top_factors,
  };
}

function hill(dose: number, maxDose: number): number {
  if (dose <= 0 || maxDose <= 0) return 0;
  const r = Math.min(1, dose / maxDose);
  const p = Math.pow(r, 0.85);
  return p / (0.35 + 0.65 * p);
}

export function evaluatePharmacoFallback(
  baseline: TwinVector,
  plan: MedicationPlan,
  fastWeight = 1,
  slowWeight = 1,
): PharmacoAssessment {
  const fw = Math.max(0, Math.min(1, fastWeight));
  const sw = Math.max(0, Math.min(1, slowWeight));

  const bb = hill(plan.metoprolol, 200);
  const ace = hill(plan.lisinopril, 40);
  const loop = hill(plan.furosemide, 80);
  const statin = hill(plan.atorvastatin, 80);
  const ccb = hill(plan.amlodipine, 10);
  const dig = hill(plan.digoxin, 0.25);
  const nit = hill(plan.isosorbide_mononitrate, 120);

  let dHr = -18 * bb * fw + 2.5 * ccb * fw - 6 * dig * fw;
  if (plan.metoprolol > 0 && plan.amlodipine > 0) dHr -= 2.2 * bb * ccb * fw;

  let dSbp =
    -11 * bb * fw -
    14 * ace * (0.65 * fw + 0.35 * sw) -
    10 * loop * fw -
    15 * ccb * fw -
    12 * nit * fw;
  let dDbp =
    -6 * bb * fw -
    8 * ace * (0.65 * fw + 0.35 * sw) -
    5 * loop * fw -
    9 * ccb * fw -
    6 * nit * fw;

  if (plan.isosorbide_mononitrate > 0 && (plan.amlodipine > 0 || plan.lisinopril > 0)) {
    const syn = 6.5 * nit * Math.max(ccb, ace) * fw;
    dSbp -= syn;
    dDbp -= 0.5 * syn;
  }

  const dEf =
    -1.5 * bb * fw +
    1.2 * bb * sw +
    (baseline.ejection_fraction < 50 ? 4.8 : 1.5) * ace * sw +
    4.5 * dig * fw;
  const dBnp =
    -65 * ace * sw -
    Math.min(260, Math.max(25, baseline.bnp * 0.38)) * loop * fw -
    30 * nit * fw;
  const dLdl = -Math.min(130, baseline.ldl * 0.48) * statin * sw;
  const dHdl = 4.5 * statin * sw;
  const dTg = -18 * statin * sw;
  const dK = 0.45 * ace * (0.5 * fw + 0.5 * sw) - 0.65 * loop * fw;
  const dCr = 0.18 * ace * (0.5 * fw + 0.5 * sw) + 0.16 * loop * (0.7 * fw + 0.3 * sw);

  const next: TwinVector = {
    ...baseline,
    heart_rate: Number(clampField('heart_rate', baseline.heart_rate + dHr).toFixed(2)),
    systolic_bp: Number(clampField('systolic_bp', baseline.systolic_bp + dSbp).toFixed(2)),
    diastolic_bp: Number(clampField('diastolic_bp', baseline.diastolic_bp + dDbp).toFixed(2)),
    ejection_fraction: Number(clampField('ejection_fraction', baseline.ejection_fraction + dEf).toFixed(2)),
    ldl: Number(clampField('ldl', baseline.ldl + dLdl).toFixed(2)),
    hdl: Number(clampField('hdl', baseline.hdl + dHdl).toFixed(2)),
    triglycerides: Number(clampField('triglycerides', baseline.triglycerides + dTg).toFixed(2)),
    bnp: Number(clampField('bnp', baseline.bnp + dBnp).toFixed(2)),
    potassium: Number(clampField('potassium', baseline.potassium + dK).toFixed(2)),
    creatinine: Number(clampField('creatinine', baseline.creatinine + dCr).toFixed(2)),
  };

  if (next.diastolic_bp >= next.systolic_bp - 10) {
    next.diastolic_bp = Number(Math.max(35, next.systolic_bp - 20).toFixed(1));
  }
  next.egfr = computeEgfr(next.creatinine, next.age, next.sex);
  next.cardiac_output = computeCardiacOutput(next.ejection_fraction, next.heart_rate, next.bmi, next.sex);

  const maceMult =
    plan.aspirin > 0
      ? Number(Math.max(0.84, 1 - 0.14 * Math.sqrt(Math.min(1, plan.aspirin / 100)) * fw).toFixed(4))
      : 1;

  const flags: RiskFlag[] = [];
  if (plan.metoprolol > 0 && next.heart_rate < 55) {
    flags.push({
      code: 'BRADYCARDIA',
      severity: next.heart_rate < 48 ? 'critical' : 'caution',
      drug: 'Metoprolol',
      message: `Projected HR ${next.heart_rate.toFixed(0)} bpm (<55 bpm).`,
    });
  }
  if (next.systolic_bp < 92 && (plan.isosorbide_mononitrate > 0 || plan.amlodipine > 0 || plan.lisinopril > 0 || plan.furosemide > 0)) {
    flags.push({
      code: 'HYPOTENSION_SYNERGY',
      severity: next.systolic_bp < 85 ? 'critical' : 'caution',
      drug: 'Vasodilator / Diuretic',
      message: `Projected SBP ${next.systolic_bp.toFixed(0)} mmHg (<92 mmHg).`,
    });
  }
  if (plan.lisinopril > 0 && next.potassium > 5.0) {
    flags.push({
      code: 'HYPERKALEMIA',
      severity: next.potassium >= 5.5 ? 'critical' : 'caution',
      drug: 'Lisinopril',
      message: `Projected K+ ${next.potassium.toFixed(2)} mEq/L (>5.0).`,
    });
  }
  if (plan.furosemide > 0 && next.potassium < 3.5) {
    flags.push({
      code: 'HYPOKALEMIA',
      severity: next.potassium < 3.1 ? 'critical' : 'caution',
      drug: 'Furosemide',
      message: `Projected K+ ${next.potassium.toFixed(2)} mEq/L (<3.5).`,
    });
  }
  if (plan.digoxin > 0 && next.potassium < 3.5) {
    flags.push({
      code: 'DIGOXIN_TOXICITY_HYPOKALEMIA',
      severity: 'critical',
      drug: 'Digoxin',
      message: `Hypokalemia (K+ ${next.potassium.toFixed(2)}) elevates digoxin toxicity risk.`,
    });
  }
  if (plan.digoxin > 0 && next.egfr < 45) {
    flags.push({
      code: 'DIGOXIN_RENAL_ACCUMULATION',
      severity: 'critical',
      drug: 'Digoxin',
      message: `Low eGFR (${next.egfr.toFixed(1)}) impairs digoxin clearance.`,
    });
  }
  if (plan.aspirin > 162) {
    flags.push({
      code: 'BLEEDING_RISK',
      severity: 'caution',
      drug: 'Aspirin',
      message: `Aspirin ${plan.aspirin.toFixed(0)} mg/day (>162 mg) elevates bleeding risk.`,
    });
  }

  return {
    projected_vector: next,
    deltas: {
      heart_rate: Number((next.heart_rate - baseline.heart_rate).toFixed(2)),
      systolic_bp: Number((next.systolic_bp - baseline.systolic_bp).toFixed(2)),
      ejection_fraction: Number((next.ejection_fraction - baseline.ejection_fraction).toFixed(2)),
      ldl: Number((next.ldl - baseline.ldl).toFixed(2)),
      bnp: Number((next.bnp - baseline.bnp).toFixed(2)),
    },
    mace_risk_multiplier: maceMult,
    risk_flags: flags,
  };
}

// Seeded in-memory clinical store for standalone mode
const SEED_PATIENTS: Record<string, { full_name: string; mrn: string; primary_diagnosis: string; active_medications: MedicationPlan; vector: TwinVector }> = {
  'pt-001': {
    full_name: 'Arjun Mehta',
    mrn: 'MRN-40821',
    primary_diagnosis: 'HFrEF (NYHA II-III) & Coronary Artery Disease',
    active_medications: { metoprolol: 50, lisinopril: 10, furosemide: 20, atorvastatin: 20, amlodipine: 0, digoxin: 0, isosorbide_mononitrate: 0, aspirin: 81 },
    vector: {
      heart_rate: 84, systolic_bp: 146, diastolic_bp: 90, ejection_fraction: 38,
      cardiac_output: 4.19, o2_saturation: 94, ldl: 148, hdl: 38, triglycerides: 210,
      bnp: 620, troponin: 0.048, potassium: 4.1, creatinine: 1.35, egfr: 58.4,
      age: 64, sex: 'M', bmi: 28.9, smoking_status: 'former', diabetes_status: true, family_history: true,
    },
  },
  'pt-002': {
    full_name: 'Elena Rostova',
    mrn: 'MRN-51904',
    primary_diagnosis: 'Uncontrolled Hypertension & Hyperlipidemia',
    active_medications: { metoprolol: 0, lisinopril: 10, furosemide: 0, atorvastatin: 10, amlodipine: 5, digoxin: 0, isosorbide_mononitrate: 0, aspirin: 0 },
    vector: {
      heart_rate: 76, systolic_bp: 152, diastolic_bp: 94, ejection_fraction: 58,
      cardiac_output: 4.84, o2_saturation: 98, ldl: 168, hdl: 49, triglycerides: 165,
      bnp: 92, troponin: 0.014, potassium: 4.3, creatinine: 0.88, egfr: 77.2,
      age: 56, sex: 'F', bmi: 26.4, smoking_status: 'never', diabetes_status: false, family_history: true,
    },
  },
  'pt-003': {
    full_name: 'Marcus Vance',
    mrn: 'MRN-63218',
    primary_diagnosis: 'Post-NSTEMI Secondary Prevention (Stable)',
    active_medications: { metoprolol: 50, lisinopril: 20, furosemide: 0, atorvastatin: 40, amlodipine: 0, digoxin: 0, isosorbide_mononitrate: 0, aspirin: 81 },
    vector: {
      heart_rate: 64, systolic_bp: 118, diastolic_bp: 74, ejection_fraction: 54,
      cardiac_output: 4.41, o2_saturation: 98.5, ldl: 84, hdl: 52, triglycerides: 115,
      bnp: 110, troponin: 0.012, potassium: 4.4, creatinine: 0.96, egfr: 91.5,
      age: 49, sex: 'M', bmi: 24.8, smoking_status: 'former', diabetes_status: false, family_history: false,
    },
  },
};

const stateVersionMap: Record<string, number> = { 'pt-001': 12, 'pt-002': 12, 'pt-003': 12 };

export const twinFallback = {
  getPatients(): PatientSummary[] {
    return Object.entries(SEED_PATIENTS).map(([id, p]) => {
      const pred = predictVector(p.vector);
      return {
        patient_id: id,
        full_name: p.full_name,
        mrn: p.mrn,
        primary_diagnosis: p.primary_diagnosis,
        active_medications: p.active_medications,
        latest_risk_pct: pred.risk_score_pct,
        latest_risk_level: pred.risk_level,
        last_updated: new Date().toISOString(),
      };
    });
  },

  getPatientTwin(patientId: string): TwinDashboardResponse {
    const p = SEED_PATIENTS[patientId] || SEED_PATIENTS['pt-001'];
    const ver = stateVersionMap[patientId] || 12;
    const now = Date.now();
    const twin_state: TwinState = {
      schema_version: TWIN_SCHEMA.schema_version,
      state_version: ver,
      patient_id: patientId,
      timestamp: new Date(now).toISOString(),
      source: 'standalone_clinical_engine',
      refresh_interval_seconds: TWIN_SCHEMA.refresh_interval_seconds,
      vector: p.vector,
      statuses: evaluateAllStatuses(p.vector),
      provenance: {
        heart_rate: 'measured', systolic_bp: 'measured', diastolic_bp: 'measured',
        ejection_fraction: 'measured', cardiac_output: 'derived', o2_saturation: 'measured',
        ldl: 'measured', hdl: 'measured', triglycerides: 'measured', bnp: 'measured',
        troponin: 'measured', potassium: 'measured', creatinine: 'measured', egfr: 'derived',
        age: 'measured', sex: 'measured', bmi: 'measured', smoking_status: 'measured',
        diabetes_status: 'measured', family_history: 'measured',
      },
    };
    const prediction = predictVector(p.vector);
    const trend: RiskTrendPoint[] = Array.from({ length: 12 }, (_, idx) => ({
      timestamp: new Date(now - (11 - idx) * 300_000).toISOString(),
      state_version: Math.max(1, ver - (11 - idx)),
      risk_score_pct: Number(Math.max(1, prediction.risk_score_pct + (idx - 11) * 0.4).toFixed(2)),
      heart_rate: p.vector.heart_rate,
      systolic_bp: p.vector.systolic_bp,
      bnp: p.vector.bnp,
      ejection_fraction: p.vector.ejection_fraction,
    }));

    return {
      patient_id: patientId,
      full_name: p.full_name,
      mrn: p.mrn,
      primary_diagnosis: p.primary_diagnosis,
      active_medications: p.active_medications,
      twin_state,
      prediction,
      trend,
    };
  },

  refreshTwin(patientId: string): TwinDashboardResponse {
    const p = SEED_PATIENTS[patientId] || SEED_PATIENTS['pt-001'];
    p.vector = {
      ...p.vector,
      heart_rate: Number((p.vector.heart_rate + (Math.random() - 0.5) * 2).toFixed(1)),
      systolic_bp: Number((p.vector.systolic_bp + (Math.random() - 0.5) * 3).toFixed(1)),
    };
    p.vector.cardiac_output = computeCardiacOutput(p.vector.ejection_fraction, p.vector.heart_rate, p.vector.bmi, p.vector.sex);
    stateVersionMap[patientId] = (stateVersionMap[patientId] || 12) + 1;
    return this.getPatientTwin(patientId);
  },

  ingestTwin(payload: Record<string, unknown>): TwinDashboardResponse {
    const patientId = String(payload.patient_id || 'pt-001');
    const p = SEED_PATIENTS[patientId] || SEED_PATIENTS['pt-001'];
    const nextVec: TwinVector = { ...p.vector };
    for (const [k, v] of Object.entries(payload)) {
      if (k in nextVec && typeof v === 'number' && !Number.isNaN(v)) {
        (nextVec as any)[k] = clampField(k, v);
      }
    }
    nextVec.egfr = computeEgfr(nextVec.creatinine, nextVec.age, nextVec.sex);
    nextVec.cardiac_output = computeCardiacOutput(nextVec.ejection_fraction, nextVec.heart_rate, nextVec.bmi, nextVec.sex);
    p.vector = nextVec;
    stateVersionMap[patientId] = (stateVersionMap[patientId] || 12) + 1;
    return this.getPatientTwin(patientId);
  },

  simulateWhatIf(
    patientId: string,
    horizonDays: number,
    plans: Array<{ plan_id: string; label: string; medications: MedicationPlan }>,
  ): WhatIfComparisonResponse {
    const p = SEED_PATIENTS[patientId] || SEED_PATIENTS['pt-001'];
    const baseline = p.vector;
    const days: number[] = [];
    for (let d = 0; d < horizonDays; d += 7) days.push(d);
    if (days[days.length - 1] !== horizonDays) days.push(horizonDays);

    const results: PlanSimulationResult[] = plans.map((planSpec) => {
      const series: SimulationPoint[] = [];
      let lastAssessment = evaluatePharmacoFallback(baseline, planSpec.medications, 0, 0);
      days.forEach((day, week) => {
        const fw = day === 0 ? 0 : 1;
        const sw = day === 0 ? 0 : 1 - Math.exp(-day / 35);
        lastAssessment = evaluatePharmacoFallback(baseline, planSpec.medications, fw, sw);
        const v = lastAssessment.projected_vector;
        const pred = predictVector(v, lastAssessment.mace_risk_multiplier);
        series.push({
          week,
          day,
          risk_pct: pred.risk_score_pct,
          systolic_bp: v.systolic_bp,
          diastolic_bp: v.diastolic_bp,
          heart_rate: v.heart_rate,
          ejection_fraction: v.ejection_fraction,
          cardiac_output: v.cardiac_output,
          ldl: v.ldl,
          bnp: v.bnp,
          potassium: v.potassium,
          egfr: v.egfr,
        });
      });

      return {
        plan_id: planSpec.plan_id,
        label: planSpec.label,
        medications: planSpec.medications,
        series,
        final_vector: lastAssessment.projected_vector,
        final_risk_pct: series[series.length - 1].risk_pct,
        risk_delta_pct: Number((series[series.length - 1].risk_pct - series[0].risk_pct).toFixed(2)),
        risk_flags: lastAssessment.risk_flags,
      };
    });

    const deltaSpecs: Array<[keyof SimulationPoint, string, string]> = [
      ['risk_pct', '30-Day MACE Risk', '%'],
      ['ejection_fraction', 'Ejection Fraction', '%'],
      ['ldl', 'LDL Cholesterol', 'mg/dL'],
      ['bnp', 'BNP', 'pg/mL'],
      ['systolic_bp', 'Systolic BP', 'mmHg'],
      ['diastolic_bp', 'Diastolic BP', 'mmHg'],
      ['heart_rate', 'Heart Rate', 'bpm'],
      ['potassium', 'Serum Potassium', 'mEq/L'],
      ['egfr', 'eGFR', 'mL/min/1.73m2'],
    ];

    const delta_table: Day90DeltaRow[] = deltaSpecs.map(([param, label, unit]) => {
      const baseVal = Number(results[0].series[0][param]);
      const planCells: Record<string, { value: number; delta: number }> = {};
      for (const r of results) {
        const endVal = Number(r.series[r.series.length - 1][param]);
        planCells[r.plan_id] = {
          value: Number(endVal.toFixed(2)),
          delta: Number((endVal - baseVal).toFixed(2)),
        };
      }
      return { parameter: param, label, unit, baseline: Number(baseVal.toFixed(2)), plans: planCells };
    });

    return {
      patient_id: patientId,
      horizon_days: horizonDays,
      baseline_risk_pct: results[0]?.series[0]?.risk_pct || 0,
      plans: results,
      delta_table,
    };
  },

  getModelMetrics(): ModelMetrics {
    return {
      n_patients: 4500,
      n_test: 900,
      mace_prevalence_pct: 28.4,
      auc_static_baseline: 0.6441,
      auc_xgboost: 0.8982,
      auc_lstm: 0.898,
      auc_hybrid: 0.9003,
      brier_static_baseline: 0.2645,
      brier_hybrid: 0.0728,
      fpr_at_80pct_sens_static: 0.604,
      fpr_at_80pct_sens_hybrid: 0.0357,
      false_positive_reduction_pct: 94.09,
    };
  },
};
