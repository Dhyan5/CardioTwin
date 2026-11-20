import {
  MedicationPlan,
  ModelMetrics,
  PatientSummary,
  PharmacoAssessment,
  TwinDashboardResponse,
  TwinVector,
  WhatIfComparisonResponse,
} from '../types/twin';

const rawBase = import.meta.env.VITE_TWIN_API_URL || '/twin-api';
const TWIN_API_BASE = rawBase.replace(/\/+$/, '');

async function twinFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${TWIN_API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers || {}),
    },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed (${res.status})`);
  }
  return res.json();
}

export const twinApi = {
  getPatients: () => twinFetch<PatientSummary[]>('/patients'),
  getPatientTwin: (patientId: string) => twinFetch<TwinDashboardResponse>(`/twin/${patientId}`),
  refreshTwin: (patientId: string) =>
    twinFetch<TwinDashboardResponse>(`/twin/${patientId}/refresh`, { method: 'POST' }),
  ingestTwin: (payload: Record<string, unknown>) =>
    twinFetch<TwinDashboardResponse>('/twin/ingest', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  evaluatePharmaco: (vector: TwinVector, medications: MedicationPlan) =>
    twinFetch<PharmacoAssessment>('/pharmaco/evaluate', {
      method: 'POST',
      body: JSON.stringify({ vector, medications }),
    }),
  simulateWhatIf: (
    patientId: string,
    horizonDays: number,
    plans: Array<{ plan_id: string; label: string; medications: MedicationPlan }>,
  ) =>
    twinFetch<WhatIfComparisonResponse>('/whatif/simulate', {
      method: 'POST',
      body: JSON.stringify({ patient_id: patientId, horizon_days: horizonDays, plans }),
    }),
  getModelMetrics: () => twinFetch<ModelMetrics>('/models/metrics'),
};
