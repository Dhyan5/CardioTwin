import {
  MedicationPlan,
  ModelMetrics,
  PatientSummary,
  PharmacoAssessment,
  TwinDashboardResponse,
  TwinVector,
  WhatIfComparisonResponse,
} from '../types/twin';
import { evaluatePharmacoFallback, twinFallback } from './twinFallback';

const rawBase = import.meta.env.VITE_TWIN_API_URL || '/twin-api';
const TWIN_API_BASE = rawBase.replace(/\/+$/, '');

let standaloneMode = false;

async function twinFetchWithFallback<T>(
  path: string,
  options: RequestInit | undefined,
  fallbackFn: () => T,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 6000);
  try {
    const res = await fetch(`${TWIN_API_BASE}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers || {}),
      },
      signal: controller.signal,
      ...options,
    });

    const contentType = res.headers.get('content-type') || '';
    if (!res.ok || !contentType.includes('application/json')) {
      standaloneMode = true;
      return fallbackFn();
    }

    standaloneMode = false;
    return (await res.json()) as T;
  } catch {
    standaloneMode = true;
    return fallbackFn();
  } finally {
    clearTimeout(timer);
  }
}

export const twinApi = {
  isStandaloneMode: () => standaloneMode,

  getPatients: () =>
    twinFetchWithFallback<PatientSummary[]>('/patients', undefined, () => twinFallback.getPatients()),

  getPatientTwin: (patientId: string) =>
    twinFetchWithFallback<TwinDashboardResponse>(`/twin/${patientId}`, undefined, () =>
      twinFallback.getPatientTwin(patientId),
    ),

  refreshTwin: (patientId: string) =>
    twinFetchWithFallback<TwinDashboardResponse>(
      `/twin/${patientId}/refresh`,
      { method: 'POST' },
      () => twinFallback.refreshTwin(patientId),
    ),

  ingestTwin: (payload: Record<string, unknown>) =>
    twinFetchWithFallback<TwinDashboardResponse>(
      '/twin/ingest',
      { method: 'POST', body: JSON.stringify(payload) },
      () => twinFallback.ingestTwin(payload),
    ),

  evaluatePharmaco: (vector: TwinVector, medications: MedicationPlan) =>
    twinFetchWithFallback<PharmacoAssessment>(
      '/pharmaco/evaluate',
      { method: 'POST', body: JSON.stringify({ vector, medications }) },
      () => evaluatePharmacoFallback(vector, medications),
    ),

  simulateWhatIf: (
    patientId: string,
    horizonDays: number,
    plans: Array<{ plan_id: string; label: string; medications: MedicationPlan }>,
  ) =>
    twinFetchWithFallback<WhatIfComparisonResponse>(
      '/whatif/simulate',
      { method: 'POST', body: JSON.stringify({ patient_id: patientId, horizon_days: horizonDays, plans }) },
      () => twinFallback.simulateWhatIf(patientId, horizonDays, plans),
    ),

  getModelMetrics: () =>
    twinFetchWithFallback<ModelMetrics>('/models/metrics', undefined, () =>
      twinFallback.getModelMetrics(),
    ),
};
