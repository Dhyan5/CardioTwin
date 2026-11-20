import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  HeartPulse,
  Moon,
  PlusCircle,
  RefreshCw,
  Sun,
  User,
  X,
} from 'lucide-react';
import { twinApi } from '../services/twinApi';
import { twinFallback } from '../services/twinFallback';
import {
  MedicationPlan,
  ModelMetrics,
  PatientSummary,
  RiskFlag,
  TwinDashboardResponse,
  TwinVector,
  WhatIfComparisonResponse,
} from '../types/twin';
import TwinStateGrid from '../components/twin/TwinStateGrid';
import RiskExplainabilityPanel from '../components/twin/RiskExplainabilityPanel';
import MedicationSimulatorPanel from '../components/twin/MedicationSimulatorPanel';

const EMPTY_PLAN: MedicationPlan = {
  metoprolol: 0,
  lisinopril: 0,
  furosemide: 0,
  atorvastatin: 0,
  amlodipine: 0,
  digoxin: 0,
  isosorbide_mononitrate: 0,
  aspirin: 0,
};

export default function ClinicalTwinDashboard() {
  const [darkMode, setDarkMode] = useState<boolean>(false);
  const [patients, setPatients] = useState<PatientSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('pt-001');
  const [dashboard, setDashboard] = useState<TwinDashboardResponse | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
  const [proposedPlan, setProposedPlan] = useState<MedicationPlan>(EMPTY_PLAN);
  const [liveFlags, setLiveFlags] = useState<RiskFlag[]>([]);
  const [simulation, setSimulation] = useState<WhatIfComparisonResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [isTwinModified, setIsTwinModified] = useState<boolean>(false);
  const [originalDashboard, setOriginalDashboard] = useState<TwinDashboardResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Vitals Ingest Modal state
  const [ingestOpen, setIngestOpen] = useState<boolean>(false);
  const [ingestForm, setIngestForm] = useState<Record<string, string>>({});

  const runSimulation = useCallback(
    async (
      patientId: string,
      current: MedicationPlan,
      proposed: MedicationPlan,
      baselineVector?: TwinVector,
    ) => {
      setIsSimulating(true);
      try {
        const res = await twinApi.simulateWhatIf(
          patientId,
          90,
          [
            { plan_id: 'no_treatment', label: 'No Treatment', medications: EMPTY_PLAN },
            { plan_id: 'current_plan', label: 'Current Plan', medications: current },
            { plan_id: 'proposed_plan', label: 'Proposed Plan', medications: proposed },
          ],
          baselineVector,
        );
        setSimulation(res);
      } catch (err: any) {
        setError(err.message || 'Failed to run 90-day projection');
      } finally {
        setIsSimulating(false);
      }
    },
    [],
  );

  const loadPatientDashboard = useCallback(
    async (patientId: string) => {
      setLoading(true);
      setError(null);
      try {
        const [data, metrics] = await Promise.all([
          twinApi.getPatientTwin(patientId),
          modelMetrics ? Promise.resolve(modelMetrics) : twinApi.getModelMetrics(),
        ]);
        setDashboard(data);
        setOriginalDashboard(data);
        setIsTwinModified(false);
        setModelMetrics(metrics);

        // Pre-populate proposed plan with an optimized GDMT suggestion above current plan
        const current = data.active_medications;
        const suggested: MedicationPlan = {
          ...current,
          metoprolol: Math.min(200, current.metoprolol + 25),
          lisinopril: Math.min(40, Math.max(current.lisinopril, 20)),
          atorvastatin: Math.min(80, Math.max(current.atorvastatin, 40)),
          aspirin: current.aspirin > 0 ? current.aspirin : 81,
        };
        setProposedPlan(suggested);
        await runSimulation(patientId, current, suggested);
      } catch (err: any) {
        setError(err.message || 'Unable to connect to CardioTwin FastAPI backend');
      } finally {
        setLoading(false);
      }
    },
    [modelMetrics, runSimulation],
  );

  useEffect(() => {
    twinApi
      .getPatients()
      .then((list) => {
        setPatients(list);
        if (list.length > 0 && !list.some((p) => p.patient_id === selectedPatientId)) {
          setSelectedPatientId(list[0].patient_id);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    loadPatientDashboard(selectedPatientId);
  }, [selectedPatientId]);

  // 5-minute (300s) automatic TwinState refresh
  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const updated = await twinApi.refreshTwin(selectedPatientId);
        setDashboard(updated);
      } catch {
        // Silent background retry on next tick
      }
    }, 300_000);
    return () => clearInterval(timer);
  }, [selectedPatientId]);

  // Live pharmacodynamic safety flag evaluation as sliders move
  useEffect(() => {
    if (!dashboard) return;
    let cancelled = false;
    twinApi
      .evaluatePharmaco(dashboard.twin_state.vector, proposedPlan)
      .then((res) => {
        if (!cancelled) setLiveFlags(res.risk_flags);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [dashboard, proposedPlan]);

  const handleUpdateTwinField = (field: keyof TwinVector, value: number | string | boolean) => {
    if (!dashboard) return;
    const rawVector: TwinVector = {
      ...dashboard.twin_state.vector,
      [field]: value,
    };
    const { vector, statuses, prediction } = twinFallback.recomputeVectorAndStatus(rawVector);

    setIsTwinModified(true);
    setDashboard({
      ...dashboard,
      twin_state: {
        ...dashboard.twin_state,
        source: 'interactive_study',
        vector,
        statuses,
      },
      prediction,
    });

    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        const livePred = await twinApi.predict(vector);
        setDashboard((prev) => (prev ? { ...prev, prediction: livePred } : prev));
        await runSimulation(selectedPatientId, dashboard.active_medications, proposedPlan, vector);
      } catch {
        // Keep immediate local recompute if backend call fails
      }
    }, 220);
  };

  const handleResetTwin = async () => {
    if (!originalDashboard) return;
    setDashboard(originalDashboard);
    setIsTwinModified(false);
    await runSimulation(
      selectedPatientId,
      originalDashboard.active_medications,
      proposedPlan,
      originalDashboard.twin_state.vector,
    );
  };

  const handleManualRefresh = async () => {
    setIsRefreshing(true);
    try {
      const updated = await twinApi.refreshTwin(selectedPatientId);
      setDashboard(updated);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleIngestSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const payload: Record<string, unknown> = {
      patient_id: selectedPatientId,
      source: 'clinician_manual_entry',
    };
    for (const [k, v] of Object.entries(ingestForm)) {
      if (v.trim() !== '') {
        payload[k] = Number(v);
      }
    }
    try {
      const updated = await twinApi.ingestTwin(payload);
      setDashboard(updated);
      setIngestOpen(false);
      setIngestForm({});
      await runSimulation(selectedPatientId, updated.active_medications, proposedPlan);
    } catch (err: any) {
      setError(err.message);
    }
  };

  return (
    <div className={darkMode ? 'dark' : ''}>
      <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
        {/* Clinical Workstation Header */}
        <header className="sticky top-0 z-30 border-b border-slate-200 dark:border-slate-800 bg-white/95 dark:bg-slate-900/95 backdrop-blur">
          <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="h-9 w-9 rounded-md bg-teal-700 text-white flex items-center justify-center">
                <HeartPulse className="h-5 w-5" aria-hidden="true" />
              </div>
              <div>
                <h1 className="text-base font-semibold tracking-tight text-slate-900 dark:text-slate-50">
                  CardioTwin Clinical Workstation
                </h1>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Reduced-Order Cardiovascular Digital Twin · 30-Day MACE Prevention
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              {/* Patient Selector */}
              <div className="flex items-center gap-1.5">
                <User className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
                <label htmlFor="patient-select" className="sr-only">
                  Select Patient
                </label>
                <select
                  id="patient-select"
                  value={selectedPatientId}
                  onChange={(e) => setSelectedPatientId(e.target.value)}
                  className="rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-800 dark:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
                >
                  {patients.map((p) => (
                    <option key={p.patient_id} value={p.patient_id}>
                      {p.full_name} ({p.mrn || p.patient_id})
                    </option>
                  ))}
                </select>
              </div>

              <button
                type="button"
                onClick={() => setIngestOpen(true)}
                className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 px-3 py-1.5 text-xs font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
              >
                <PlusCircle className="h-3.5 w-3.5 text-teal-700 dark:text-teal-400" aria-hidden="true" />
                Ingest Labs / Vitals
              </button>

              <button
                type="button"
                onClick={handleManualRefresh}
                disabled={isRefreshing}
                className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 px-3 py-1.5 text-xs font-medium disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
              >
                <RefreshCw
                  className={`h-3.5 w-3.5 ${isRefreshing ? 'animate-spin' : ''}`}
                  aria-hidden="true"
                />
                Sync (5m)
              </button>

              <button
                type="button"
                onClick={() => setDarkMode(!darkMode)}
                aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                className="rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 p-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
              >
                {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {dashboard && (
            <div className="max-w-[1440px] mx-auto px-4 sm:px-6 pb-2.5 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600 dark:text-slate-400">
              <div>
                <span className="font-semibold text-slate-900 dark:text-slate-100">
                  {dashboard.full_name}
                </span>{' '}
                · {dashboard.primary_diagnosis}
              </div>
              <div className="font-mono text-[11px] flex items-center gap-2">
                <span
                  className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded border ${
                    twinApi.isStandaloneMode()
                      ? 'bg-amber-500/10 text-amber-700 dark:text-amber-300 border-amber-500/30'
                      : 'bg-teal-500/10 text-teal-700 dark:text-teal-300 border-teal-500/30'
                  }`}
                >
                  {twinApi.isStandaloneMode() ? 'Standalone Clinical Engine' : 'Live FastAPI Backend'}
                </span>
                <span>
                  Last fused: {new Date(dashboard.twin_state.timestamp).toLocaleTimeString()} · Auto-refresh 300s
                </span>
              </div>
            </div>
          )}
        </header>

        {/* Main Content */}
        <main className="max-w-[1440px] mx-auto px-4 sm:px-6 py-5 space-y-5">
          {error && (
            <div
              role="alert"
              className="rounded-md border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-xs text-rose-800 dark:text-rose-200"
            >
              <strong>System Notice:</strong> {error}
            </div>
          )}

          {loading && !dashboard ? (
            <div className="py-20 text-center text-sm text-slate-500">
              Initializing Digital Twin State Vector & Hybrid XGBoost–LSTM Model...
            </div>
          ) : (
            dashboard && (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
                  <div className="lg:col-span-8">
                    <TwinStateGrid
                      twinState={dashboard.twin_state}
                      onUpdateField={handleUpdateTwinField}
                      onResetTwin={handleResetTwin}
                      isModified={isTwinModified}
                    />
                  </div>
                  <div className="lg:col-span-4 flex">
                    <div className="w-full">
                      <RiskExplainabilityPanel
                        prediction={dashboard.prediction}
                        trend={dashboard.trend}
                        modelMetrics={modelMetrics}
                      />
                    </div>
                  </div>
                </div>

                <MedicationSimulatorPanel
                  currentPlan={dashboard.active_medications}
                  proposedPlan={proposedPlan}
                  onChangeProposed={setProposedPlan}
                  onResetProposed={() => setProposedPlan(dashboard.active_medications)}
                  onRunProjection={() =>
                    runSimulation(
                      selectedPatientId,
                      dashboard.active_medications,
                      proposedPlan,
                      dashboard.twin_state.vector,
                    )
                  }
                  isSimulating={isSimulating}
                  simulationResult={simulation}
                  liveFlags={liveFlags}
                />
              </>
            )
          )}
        </main>

        {/* Manual Vitals / Labs Ingest Modal */}
        {ingestOpen && (
          <div
            className="fixed inset-0 z-50 bg-slate-950/60 flex items-center justify-center p-4"
            role="dialog"
            aria-modal="true"
            aria-labelledby="ingest-modal-title"
          >
            <div className="w-full max-w-lg rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-3">
                <h2 id="ingest-modal-title" className="text-sm font-semibold">
                  Ingest New Vitals / Lab Measurements
                </h2>
                <button
                  type="button"
                  onClick={() => setIngestOpen(false)}
                  aria-label="Close modal"
                  className="p-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <form onSubmit={handleIngestSubmit} className="space-y-4">
                <div className="grid grid-cols-2 gap-3 text-xs">
                  {[
                    ['heart_rate', 'Heart Rate (bpm)'],
                    ['systolic_bp', 'Systolic BP (mmHg)'],
                    ['diastolic_bp', 'Diastolic BP (mmHg)'],
                    ['ejection_fraction', 'Ejection Fraction (%)'],
                    ['bnp', 'BNP (pg/mL)'],
                    ['troponin', 'Troponin I (ng/mL)'],
                    ['ldl', 'LDL (mg/dL)'],
                    ['potassium', 'Potassium (mEq/L)'],
                    ['creatinine', 'Creatinine (mg/dL)'],
                    ['o2_saturation', 'SpO2 (%)'],
                  ].map(([key, label]) => (
                    <div key={key} className="space-y-1">
                      <label htmlFor={`ingest-${key}`} className="block font-medium text-slate-600 dark:text-slate-300">
                        {label}
                      </label>
                      <input
                        id={`ingest-${key}`}
                        type="number"
                        step="any"
                        placeholder="Leave blank to retain"
                        value={ingestForm[key] || ''}
                        onChange={(e) => setIngestForm({ ...ingestForm, [key]: e.target.value })}
                        className="w-full rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-2.5 py-1.5 font-mono text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
                      />
                    </div>
                  ))}
                </div>
                <div className="flex justify-end gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => setIngestOpen(false)}
                    className="px-3 py-1.5 rounded border border-slate-300 dark:border-slate-700 text-xs font-medium"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="px-4 py-1.5 rounded bg-teal-700 hover:bg-teal-800 text-white text-xs font-semibold"
                  >
                    Fuse & Re-Score Twin
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
