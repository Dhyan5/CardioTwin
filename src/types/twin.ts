import twinSchema from '../../shared/twin_schema.json';

export const TWIN_SCHEMA = twinSchema;

export type FieldStatus = 'normal' | 'caution' | 'critical';
export type ProvenanceType = 'measured' | 'imputed' | 'derived';
export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';

export interface TwinVector {
  heart_rate: number;
  systolic_bp: number;
  diastolic_bp: number;
  ejection_fraction: number;
  cardiac_output: number;
  o2_saturation: number;
  ldl: number;
  hdl: number;
  triglycerides: number;
  bnp: number;
  troponin: number;
  potassium: number;
  creatinine: number;
  egfr: number;
  age: number;
  sex: 'M' | 'F';
  bmi: number;
  smoking_status: 'never' | 'former' | 'current';
  diabetes_status: boolean;
  family_history: boolean;
}

export interface TwinState {
  schema_version: string;
  state_version: number;
  patient_id: string;
  timestamp: string;
  source: string;
  refresh_interval_seconds: number;
  vector: TwinVector;
  statuses: Record<string, FieldStatus>;
  provenance: Record<string, ProvenanceType>;
}

export interface RiskFactorContribution {
  feature: string;
  display_name: string;
  value: number;
  shap_value: number;
  direction: 'increases_risk' | 'decreases_risk';
  magnitude: number;
}

export interface RiskPrediction {
  risk_score_pct: number;
  risk_level: RiskLevel;
  xgb_prob: number;
  lstm_prob: number;
  baseline_static_pct: number;
  latency_ms: number;
  top_factors: RiskFactorContribution[];
}

export interface RiskTrendPoint {
  timestamp: string;
  state_version: number;
  risk_score_pct: number;
  heart_rate: number;
  systolic_bp: number;
  bnp: number;
  ejection_fraction: number;
}

export interface MedicationPlan {
  metoprolol: number;
  lisinopril: number;
  furosemide: number;
  atorvastatin: number;
  amlodipine: number;
  digoxin: number;
  isosorbide_mononitrate: number;
  aspirin: number;
}

export interface RiskFlag {
  code: string;
  severity: 'caution' | 'critical';
  drug: string;
  message: string;
}

export interface PharmacoAssessment {
  projected_vector: TwinVector;
  deltas: Record<string, number>;
  mace_risk_multiplier: number;
  risk_flags: RiskFlag[];
}

export interface PatientSummary {
  patient_id: string;
  full_name: string;
  mrn?: string | null;
  primary_diagnosis?: string | null;
  active_medications: MedicationPlan;
  latest_risk_pct?: number | null;
  latest_risk_level?: string | null;
  last_updated?: string | null;
}

export interface TwinDashboardResponse {
  patient_id: string;
  full_name: string;
  mrn?: string | null;
  primary_diagnosis?: string | null;
  active_medications: MedicationPlan;
  twin_state: TwinState;
  prediction: RiskPrediction;
  trend: RiskTrendPoint[];
}

export interface SimulationPoint {
  week: number;
  day: number;
  risk_pct: number;
  systolic_bp: number;
  diastolic_bp: number;
  heart_rate: number;
  ejection_fraction: number;
  cardiac_output: number;
  ldl: number;
  bnp: number;
  potassium: number;
  egfr: number;
}

export interface PlanSimulationResult {
  plan_id: string;
  label: string;
  medications: MedicationPlan;
  series: SimulationPoint[];
  final_vector: TwinVector;
  final_risk_pct: number;
  risk_delta_pct: number;
  risk_flags: RiskFlag[];
}

export interface Day90DeltaRow {
  parameter: string;
  label: string;
  unit: string;
  baseline: number;
  plans: Record<string, { value: number; delta: number }>;
}

export interface WhatIfComparisonResponse {
  patient_id: string;
  horizon_days: number;
  baseline_risk_pct: number;
  plans: PlanSimulationResult[];
  delta_table: Day90DeltaRow[];
}

export interface ModelMetrics {
  n_patients: number;
  n_test: number;
  mace_prevalence_pct: number;
  auc_static_baseline: number;
  auc_xgboost: number;
  auc_lstm: number;
  auc_hybrid: number;
  brier_static_baseline: number;
  brier_hybrid: number;
  fpr_at_80pct_sens_static: number;
  fpr_at_80pct_sens_hybrid: number;
  false_positive_reduction_pct: number;
}
