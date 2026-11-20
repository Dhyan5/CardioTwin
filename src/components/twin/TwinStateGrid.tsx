import React from 'react';
import { Activity, AlertCircle, CheckCircle2, AlertTriangle } from 'lucide-react';
import { FieldStatus, TWIN_SCHEMA, TwinState } from '../../types/twin';

interface TwinStateGridProps {
  twinState: TwinState;
}

interface FieldGroup {
  title: string;
  fields: Array<keyof TwinState['vector']>;
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

export default function TwinStateGrid({ twinState }: TwinStateGridProps) {
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
              Schema v{twinState.schema_version} · Snapshot v{twinState.state_version} · Source:{' '}
              <span className="font-mono">{twinState.source}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
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
                const formattedVal =
                  fieldKey === 'troponin'
                    ? rawVal.toFixed(3)
                    : fieldKey === 'cardiac_output' || fieldKey === 'potassium' || fieldKey === 'creatinine'
                      ? rawVal.toFixed(2)
                      : rawVal.toFixed(1);

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
                      <span className="text-xs font-medium text-slate-700 dark:text-slate-300 leading-tight">
                        {meta?.label || fieldKey}
                      </span>
                      <StatusBadge status={status} />
                    </div>

                    <div className="flex items-baseline justify-between gap-2">
                      <div className="flex items-baseline gap-1.5">
                        <span className="text-xl font-semibold font-mono tabular-nums text-slate-900 dark:text-slate-50">
                          {formattedVal}
                        </span>
                        <span className="text-xs text-slate-500 dark:text-slate-400">
                          {meta?.unit}
                        </span>
                      </div>
                      <span className="text-[10px] uppercase tracking-wider text-slate-400 dark:text-slate-500 font-mono">
                        {prov}
                      </span>
                    </div>

                    <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                      {formatNormalRange(fieldKey)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        {/* Demographics & Risk Profile Strip */}
        <div className="pt-2 border-t border-slate-200 dark:border-slate-800 flex flex-wrap gap-x-6 gap-y-2 text-xs text-slate-600 dark:text-slate-300">
          <div>
            <span className="text-slate-400 dark:text-slate-500">Age / Sex:</span>{' '}
            <span className="font-medium font-mono">
              {vector.age}y / {vector.sex}
            </span>
          </div>
          <div>
            <span className="text-slate-400 dark:text-slate-500">Smoking:</span>{' '}
            <span className="font-medium capitalize">{vector.smoking_status}</span>
          </div>
          <div>
            <span className="text-slate-400 dark:text-slate-500">Diabetes:</span>{' '}
            <span className="font-medium">{vector.diabetes_status ? 'Yes' : 'No'}</span>
          </div>
          <div>
            <span className="text-slate-400 dark:text-slate-500">Family Hx CAD:</span>{' '}
            <span className="font-medium">{vector.family_history ? 'Positive' : 'Negative'}</span>
          </div>
        </div>
      </div>
    </section>
  );
}
