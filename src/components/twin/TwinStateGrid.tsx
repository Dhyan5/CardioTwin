import React from 'react';
import { Activity, AlertCircle, CheckCircle2, AlertTriangle, RotateCcw } from 'lucide-react';
import { FieldStatus, TWIN_SCHEMA, TwinState, TwinVector } from '../../types/twin';

interface TwinStateGridProps {
  twinState: TwinState;
  onUpdateField?: (field: keyof TwinVector, value: number | string | boolean) => void;
  onResetTwin?: () => void;
  isModified?: boolean;
}

interface FieldGroup {
  title: string;
  fields: Array<keyof TwinVector>;
}

const FIELD_GROUPS: FieldGroup[] = [
  {
    title: 'Hemodynamics & Bedside Vitals',
    fields: ['heart_rate', 'systolic_bp', 'diastolic_bp', 'ejection_fraction', 'cardiac_output', 'o2_saturation'],
  },
  {
    title: 'Cardiac Biomarkers & Renal Panel',
    fields: ['bnp', 'troponin', 'creatinine', 'egfr', 'potassium'],
  },
  {
    title: 'Lipid Profile & Anthropometrics',
    fields: ['ldl', 'hdl', 'triglycerides', 'bmi'],
  },
];

const FIELD_STEPS: Partial<Record<keyof TwinVector, number>> = {
  heart_rate: 1,
  systolic_bp: 1,
  diastolic_bp: 1,
  ejection_fraction: 1,
  o2_saturation: 0.5,
  bnp: 5,
  troponin: 0.005,
  creatinine: 0.05,
  potassium: 0.1,
  ldl: 2,
  hdl: 1,
  triglycerides: 5,
  bmi: 0.2,
};

function StatusBadge({ status }: { status: FieldStatus }) {
  if (status === 'critical') {
    return (
      <span className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-rose-500/15 text-rose-700 dark:text-rose-300 border border-rose-500/30">
        <AlertCircle className="h-3 w-3" aria-hidden="true" />
        Critical
      </span>
    );
  }
  if (status === 'caution') {
    return (
      <span className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-amber-500/15 text-amber-800 dark:text-amber-300 border border-amber-500/30">
        <AlertTriangle className="h-3 w-3" aria-hidden="true" />
        Caution
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-teal-500/10 text-teal-800 dark:text-teal-300 border border-teal-500/25">
      <CheckCircle2 className="h-3 w-3" aria-hidden="true" />
      Normal
    </span>
  );
}

function formatNormalRange(fieldKey: string): string {
  const meta = (TWIN_SCHEMA.fields as Record<string, any>)[fieldKey];
  if (!meta || meta.normal_min === undefined || meta.normal_max === undefined) return '';
  if (meta.normal_min === 0) return `Ref < ${meta.normal_max}`;
  if (meta.normal_max >= 120 && fieldKey === 'egfr') return `Ref ≥ ${meta.normal_min}`;
  return `Ref ${meta.normal_min}–${meta.normal_max}`;
}

export default function TwinStateGrid({
  twinState,
  onUpdateField,
  onResetTwin,
  isModified = false,
}: TwinStateGridProps) {
  const { vector, statuses, provenance } = twinState;
  const schemaFields = TWIN_SCHEMA.fields as Record<string, any>;

  const criticalCount = Object.values(statuses).filter((s) => s === 'critical').length;
  const cautionCount = Object.values(statuses).filter((s) => s === 'caution').length;

  return (
    <section
      aria-label="Live Digital Twin State Vector"
      className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 dark:border-slate-800 pb-3 mb-4">
        <div className="flex items-center gap-2.5">
          <Activity className="h-4 w-4 text-teal-700 dark:text-teal-400" aria-hidden="true" />
          <div>
            <h2 className="text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100">
              Unified Digital Twin State Vector
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Adjust any vital, lab, or risk factor below to study live MACE risk &amp; 90-day outcomes · Schema v{twinState.schema_version} · Snapshot v{twinState.state_version}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          {isModified && onResetTwin && (
            <button
              type="button"
              onClick={onResetTwin}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded font-medium border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              <RotateCcw className="h-3 w-3" aria-hidden="true" />
              Reset Baseline
            </button>
          )}
          {criticalCount > 0 && (
            <span className="px-2 py-0.5 rounded font-medium bg-rose-500/15 text-rose-700 dark:text-rose-300 border border-rose-500/30">
              {criticalCount} Critical
            </span>
          )}
          {cautionCount > 0 && (
            <span className="px-2 py-0.5 rounded font-medium bg-amber-500/15 text-amber-800 dark:text-amber-300 border border-amber-500/30">
              {cautionCount} Caution
            </span>
          )}
          {criticalCount === 0 && cautionCount === 0 && (
            <span className="px-2 py-0.5 rounded font-medium bg-teal-500/10 text-teal-800 dark:text-teal-300 border border-teal-500/25">
              All Parameters Within Reference
            </span>
          )}
        </div>
      </div>

      <div className="space-y-5">
        {FIELD_GROUPS.map((group) => (
          <div key={group.title}>
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-2">
              {group.title}
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-2.5">
              {group.fields.map((fieldKey) => {
                const meta = schemaFields[fieldKey];
                const rawVal = vector[fieldKey] as number;
                const status: FieldStatus = statuses[fieldKey] || 'normal';
                const prov = provenance[fieldKey] || 'measured';
                const isDerived = Boolean(meta?.derived);
                const step = FIELD_STEPS[fieldKey] ?? 1;
                const decimals =
                  fieldKey === 'troponin'
                    ? 3
                    : fieldKey === 'cardiac_output' || fieldKey === 'potassium' || fieldKey === 'creatinine'
                      ? 2
                      : 1;
                const formattedVal = rawVal.toFixed(decimals);

                const borderClass =
                  status === 'critical'
                    ? 'border-rose-500/50 bg-rose-500/[0.03]'
                    : status === 'caution'
                      ? 'border-amber-500/40 bg-amber-500/[0.02]'
                      : 'border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/40';

                return (
                  <div
                    key={fieldKey}
                    className={`rounded-md border p-3 transition-colors ${borderClass}`}
                  >
                    <div className="flex items-start justify-between gap-2 mb-1.5">
                      <label
                        htmlFor={`twin-input-${fieldKey}`}
                        className="text-xs font-medium text-slate-700 dark:text-slate-300 leading-tight"
                      >
                        {meta?.label || fieldKey}
                      </label>
                      <StatusBadge status={status} />
                    </div>

                    <div className="flex items-baseline justify-between gap-2">
                      <div className="flex items-baseline gap-1.5">
                        {onUpdateField && !isDerived ? (
                          <input
                            id={`twin-input-${fieldKey}`}
                            type="number"
                            min={meta?.physiological_min}
                            max={meta?.physiological_max}
                            step={step}
                            value={Number(formattedVal)}
                            onChange={(e) => {
                              const parsed = parseFloat(e.target.value);
                              if (!Number.isNaN(parsed)) {
                                onUpdateField(fieldKey, parsed);
                              }
                            }}
                            aria-label={`${meta?.label || fieldKey} value`}
                            className="w-20 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-1.5 py-0.5 text-lg font-semibold font-mono tabular-nums text-slate-900 dark:text-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
                          />
                        ) : (
                          <span className="text-xl font-semibold font-mono tabular-nums text-slate-900 dark:text-slate-50">
                            {formattedVal}
                          </span>
                        )}
                        <span className="text-xs text-slate-500 dark:text-slate-400">
                          {meta?.unit}
                        </span>
                      </div>
                      <span className="text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500 font-mono">
                        {isDerived ? 'derived' : prov}
                      </span>
                    </div>

                    {onUpdateField && !isDerived && meta?.physiological_min !== undefined && (
                      <input
                        type="range"
                        min={meta.physiological_min}
                        max={meta.physiological_max}
                        step={step}
                        value={rawVal}
                        onChange={(e) => onUpdateField(fieldKey, parseFloat(e.target.value))}
                        aria-label={`${meta?.label || fieldKey} slider`}
                        className="mt-2 w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none accent-teal-700 dark:accent-teal-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
                      />
                    )}

                    <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                      {formatNormalRange(fieldKey)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        {/* Interactive Demographics & Risk Profile Controls */}
        <div className="pt-3 border-t border-slate-200 dark:border-slate-800 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-slate-600 dark:text-slate-300">
          <div className="flex items-center gap-1.5">
            <label htmlFor="twin-age" className="text-slate-500 dark:text-slate-400">Age:</label>
            <input
              id="twin-age"
              type="number"
              min={18}
              max={110}
              value={vector.age}
              disabled={!onUpdateField}
              onChange={(e) => onUpdateField?.('age', Math.max(18, Math.min(110, Number(e.target.value) || 18)))}
              className="w-14 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-1.5 py-0.5 font-mono text-xs"
            />
          </div>

          <div className="flex items-center gap-1.5">
            <label htmlFor="twin-sex" className="text-slate-500 dark:text-slate-400">Sex:</label>
            <select
              id="twin-sex"
              value={vector.sex}
              disabled={!onUpdateField}
              onChange={(e) => onUpdateField?.('sex', e.target.value)}
              className="rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-1.5 py-0.5 text-xs"
            >
              <option value="M">M</option>
              <option value="F">F</option>
            </select>
          </div>

          <div className="flex items-center gap-1.5">
            <label htmlFor="twin-smoking" className="text-slate-500 dark:text-slate-400">Smoking:</label>
            <select
              id="twin-smoking"
              value={vector.smoking_status}
              disabled={!onUpdateField}
              onChange={(e) => onUpdateField?.('smoking_status', e.target.value)}
              className="rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-1.5 py-0.5 text-xs"
            >
              <option value="never">Never</option>
              <option value="former">Former</option>
              <option value="current">Current</option>
            </select>
          </div>

          <label className="inline-flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={vector.diabetes_status}
              disabled={!onUpdateField}
              onChange={(e) => onUpdateField?.('diabetes_status', e.target.checked)}
              className="rounded border-slate-300 text-teal-700 focus:ring-teal-600"
            />
            <span>Diabetes</span>
          </label>

          <label className="inline-flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={vector.family_history}
              disabled={!onUpdateField}
              onChange={(e) => onUpdateField?.('family_history', e.target.checked)}
              className="rounded border-slate-300 text-teal-700 focus:ring-teal-600"
            />
            <span>Family Hx CAD</span>
          </label>
        </div>
      </div>
    </section>
  );
}
