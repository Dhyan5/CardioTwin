import React, { useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AlertTriangle, Play, RotateCcw, Pill } from 'lucide-react';
import {
  MedicationPlan,
  RiskFlag,
  TWIN_SCHEMA,
  WhatIfComparisonResponse,
} from '../../types/twin';

interface MedicationSimulatorPanelProps {
  currentPlan: MedicationPlan;
  proposedPlan: MedicationPlan;
  onChangeProposed: (plan: MedicationPlan) => void;
  onResetProposed: () => void;
  onRunProjection: () => void;
  isSimulating: boolean;
  simulationResult: WhatIfComparisonResponse | null;
  liveFlags: RiskFlag[];
}

interface DrugGroup {
  groupName: string;
  drugs: Array<keyof MedicationPlan>;
}

const DRUG_GROUPS: DrugGroup[] = [
  {
    groupName: 'Neurohormonal & Rate Control',
    drugs: ['metoprolol', 'lisinopril'],
  },
  {
    groupName: 'Volume & Preload Management',
    drugs: ['furosemide', 'isosorbide_mononitrate'],
  },
  {
    groupName: 'Vascular & Inotropic Support',
    drugs: ['amlodipine', 'digoxin'],
  },
  {
    groupName: 'Lipid & Antithrombotic Therapy',
    drugs: ['atorvastatin', 'aspirin'],
  },
];

type ChartMetricKey = 'risk_pct' | 'ejection_fraction' | 'bnp' | 'ldl' | 'systolic_bp';

const METRIC_OPTIONS: Array<{ key: ChartMetricKey; label: string; unit: string }> = [
  { key: 'risk_pct', label: '30-Day MACE Risk', unit: '%' },
  { key: 'ejection_fraction', label: 'Ejection Fraction', unit: '%' },
  { key: 'bnp', label: 'BNP', unit: 'pg/mL' },
  { key: 'ldl', label: 'LDL Cholesterol', unit: 'mg/dL' },
  { key: 'systolic_bp', label: 'Systolic BP', unit: 'mmHg' },
];

export default function MedicationSimulatorPanel({
  currentPlan,
  proposedPlan,
  onChangeProposed,
  onResetProposed,
  onRunProjection,
  isSimulating,
  simulationResult,
  liveFlags,
}: MedicationSimulatorPanelProps) {
  const [chartMetric, setChartMetric] = useState<ChartMetricKey>('risk_pct');
  const medMeta = TWIN_SCHEMA.medications as Record<string, any>;

  const handleSliderChange = (drug: keyof MedicationPlan, value: number) => {
    onChangeProposed({ ...proposedPlan, [drug]: value });
  };

  // Build chart-ready merged series across plans
  const mergedChartData =
    simulationResult && simulationResult.plans.length > 0
      ? simulationResult.plans[0].series.map((pt, idx) => {
          const row: Record<string, number | string> = {
            dayLabel: `D${pt.day}`,
            day: pt.day,
          };
          for (const planRes of simulationResult.plans) {
            const point = planRes.series[idx];
            if (point) {
              row[planRes.plan_id] = point[chartMetric];
            }
          }
          return row;
        })
      : [];

  const activeMetricMeta = METRIC_OPTIONS.find((m) => m.key === chartMetric)!;

  return (
    <section
      aria-label="Medication Pharmacodynamics and What-If Simulator"
      className="grid grid-cols-1 lg:grid-cols-12 gap-5"
    >
      {/* Left 5 Cols: Medication Sliders */}
      <div className="lg:col-span-5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 flex flex-col justify-between gap-4">
        <div>
          <div className="flex items-center justify-between gap-2 border-b border-slate-200 dark:border-slate-800 pb-3 mb-4">
            <div className="flex items-center gap-2">
              <Pill className="h-4 w-4 text-teal-700 dark:text-teal-400" aria-hidden="true" />
              <div>
                <h2 className="text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100">
                  Proposed Medication Plan
                </h2>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  8-class pharmacodynamic dosing & interaction model
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={onResetProposed}
              className="inline-flex items-center gap-1 text-xs font-medium text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white px-2 py-1 rounded border border-slate-200 dark:border-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
            >
              <RotateCcw className="h-3 w-3" aria-hidden="true" />
              Reset
            </button>
          </div>

          <div className="space-y-4">
            {DRUG_GROUPS.map((group) => (
              <div key={group.groupName} className="space-y-2.5">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  {group.groupName}
                </h3>
                <div className="space-y-3">
                  {group.drugs.map((drugKey) => {
                    const meta = medMeta[drugKey];
                    const val = proposedPlan[drugKey];
                    const curVal = currentPlan[drugKey];
                    const changed = Math.abs(val - curVal) > 1e-6;
                    const displayDose = drugKey === 'digoxin' ? val.toFixed(4).replace(/0+$/, '').replace(/\.$/, '') : val;

                    return (
                      <div key={drugKey} className="space-y-1">
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <label
                              htmlFor={`slider-${drugKey}`}
                              className="font-medium text-slate-800 dark:text-slate-200 truncate"
                            >
                              {meta.generic_name}
                            </label>
                            <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400">
                              {meta.class}
                            </span>
                            <span className="text-[10px] font-mono text-slate-400">
                              {meta.onset}
                            </span>
                          </div>
                          <span
                            className={`font-mono tabular-nums font-semibold shrink-0 ${
                              changed
                                ? 'text-teal-700 dark:text-teal-400'
                                : 'text-slate-700 dark:text-slate-300'
                            }`}
                          >
                            {displayDose} {meta.unit}
                          </span>
                        </div>

                        <input
                          id={`slider-${drugKey}`}
                          type="range"
                          min={meta.min_dose}
                          max={meta.max_dose}
                          step={meta.step}
                          value={val}
                          onChange={(e) => handleSliderChange(drugKey, parseFloat(e.target.value))}
                          aria-label={`${meta.generic_name} (${meta.class}) dose in ${meta.unit}`}
                          className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none accent-teal-700 dark:accent-teal-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600"
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Live Safety Flags */}
        {liveFlags.length > 0 && (
          <div className="rounded-md border border-rose-500/30 bg-rose-500/[0.05] p-3 space-y-1.5">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-rose-700 dark:text-rose-300">
              <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />
              Pharmacodynamic Safety Alerts ({liveFlags.length})
            </div>
            <ul className="space-y-1 text-xs text-slate-700 dark:text-slate-300">
              {liveFlags.map((flag) => (
                <li key={flag.code} className="leading-snug">
                  <span className="font-mono font-semibold text-rose-700 dark:text-rose-300">
                    [{flag.code}]
                  </span>{' '}
                  {flag.message}
                </li>
              ))}
            </ul>
          </div>
        )}

        <button
          type="button"
          onClick={onRunProjection}
          disabled={isSimulating}
          className="w-full inline-flex items-center justify-center gap-2 rounded-md bg-teal-700 hover:bg-teal-800 disabled:opacity-60 text-white font-semibold text-sm py-2.5 px-4 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600 focus-visible:ring-offset-2 dark:focus-visible:ring-offset-slate-900"
        >
          <Play className="h-4 w-4" aria-hidden="true" />
          {isSimulating ? 'Running 90-Day Projection...' : 'Run Projection (90 Days)'}
        </button>
      </div>

      {/* Right 7 Cols: What-If Projection Chart + Day-90 Delta Table */}
      <div className="lg:col-span-7 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 flex flex-col gap-5">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 dark:border-slate-800 pb-3">
          <div>
            <h2 className="text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100">
              90-Day Longitudinal What-If Projection
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Weekly steps · Fast-onset immediate, slow-onset saturating exponential (τ = 5 wks)
            </p>
          </div>

          <div className="flex flex-wrap gap-1" role="group" aria-label="Select projection metric">
            {METRIC_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                type="button"
                onClick={() => setChartMetric(opt.key)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600 ${
                  chartMetric === opt.key
                    ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Multi-plan Line Chart */}
        <div className="h-64 w-full">
          {mergedChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mergedChartData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.2} />
                <XAxis dataKey="dayLabel" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{
                    fontSize: '12px',
                    borderRadius: '6px',
                    border: '1px solid #cbd5e1',
                  }}
                  formatter={(val: number, name: string) => [
                    `${val.toFixed(1)} ${activeMetricMeta.unit}`,
                    name,
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Line
                  type="monotone"
                  dataKey="no_treatment"
                  name="No Treatment"
                  stroke="#94a3b8"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="current_plan"
                  name="Current Plan"
                  stroke="#d97706"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="proposed_plan"
                  name="Proposed Plan"
                  stroke="#0f766e"
                  strokeWidth={2.75}
                  dot={{ r: 2.5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-slate-500">
              Adjust medication sliders and click &ldquo;Run Projection&rdquo; to simulate 90-day outcomes.
            </div>
          )}
        </div>

        {/* Day-90 Delta Comparison Table */}
        {simulationResult && (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-2">
              Day-90 Projected Outcome & Delta Comparison
            </h3>
            <div className="overflow-x-auto rounded-md border border-slate-200 dark:border-slate-800">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 dark:bg-slate-950/60 border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                    <th className="py-2 px-3 font-semibold">Parameter</th>
                    <th className="py-2 px-3 font-semibold text-right">Baseline</th>
                    <th className="py-2 px-3 font-semibold text-right">No Treatment</th>
                    <th className="py-2 px-3 font-semibold text-right">Current Plan</th>
                    <th className="py-2 px-3 font-semibold text-right text-teal-700 dark:text-teal-400">
                      Proposed Plan
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono tabular-nums">
                  {simulationResult.delta_table.map((row) => {
                    const noTx = row.plans['no_treatment'];
                    const cur = row.plans['current_plan'];
                    const prop = row.plans['proposed_plan'];
                    const lowerIsBetter = ['risk_pct', 'ldl', 'bnp', 'systolic_bp', 'diastolic_bp'].includes(
                      row.parameter,
                    );
                    const propImproved = prop
                      ? lowerIsBetter
                        ? prop.delta < -0.05
                        : prop.delta > 0.05
                      : false;

                    return (
                      <tr key={row.parameter} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/30">
                        <td className="py-2 px-3 font-sans font-medium text-slate-800 dark:text-slate-200">
                          {row.label}{' '}
                          <span className="text-[10px] text-slate-400 font-normal">({row.unit})</span>
                        </td>
                        <td className="py-2 px-3 text-right text-slate-600 dark:text-slate-300">
                          {row.baseline.toFixed(1)}
                        </td>
                        <td className="py-2 px-3 text-right text-slate-500 dark:text-slate-400">
                          {noTx ? `${noTx.value.toFixed(1)}` : '—'}
                        </td>
                        <td className="py-2 px-3 text-right text-slate-700 dark:text-slate-300">
                          {cur ? (
                            <span>
                              {cur.value.toFixed(1)}{' '}
                              <span className="text-[10px] text-slate-400">
                                ({cur.delta >= 0 ? '+' : ''}
                                {cur.delta.toFixed(1)})
                              </span>
                            </span>
                          ) : (
                            '—'
                          )}
                        </td>
                        <td className="py-2 px-3 text-right font-semibold">
                          {prop ? (
                            <span
                              className={
                                propImproved
                                  ? 'text-teal-700 dark:text-teal-400'
                                  : 'text-slate-800 dark:text-slate-200'
                              }
                            >
                              {prop.value.toFixed(1)}{' '}
                              <span className="text-[10px]">
                                ({prop.delta >= 0 ? '+' : ''}
                                {prop.delta.toFixed(1)})
                              </span>
                            </span>
                          ) : (
                            '—'
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
