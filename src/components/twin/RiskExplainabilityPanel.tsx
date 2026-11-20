import React from 'react';
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { ShieldAlert, TrendingUp, Cpu } from 'lucide-react';
import { ModelMetrics, RiskPrediction, RiskTrendPoint } from '../../types/twin';

interface RiskExplainabilityPanelProps {
  prediction: RiskPrediction;
  trend: RiskTrendPoint[];
  modelMetrics?: ModelMetrics | null;
}

export default function RiskExplainabilityPanel({
  prediction,
  trend,
  modelMetrics,
}: RiskExplainabilityPanelProps) {
  const { risk_score_pct, risk_level, baseline_static_pct, xgb_prob, lstm_prob, latency_ms, top_factors } =
    prediction;

  const isElevated = risk_level === 'CRITICAL' || risk_level === 'HIGH';
  const strokeColor = isElevated ? '#be123c' : '#0f766e';
  const badgeClasses =
    risk_level === 'CRITICAL' || risk_level === 'HIGH'
      ? 'bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30'
      : risk_level === 'MODERATE'
        ? 'bg-amber-500/15 text-amber-800 dark:text-amber-300 border-amber-500/30'
        : 'bg-teal-500/10 text-teal-800 dark:text-teal-300 border-teal-500/25';

  // Semi-circular gauge math (r=64, circumference=pi*64=201.06)
  const radius = 64;
  const semiCircumference = Math.PI * radius;
  const clampedPct = Math.max(0, Math.min(100, risk_score_pct));
  const dashOffset = semiCircumference * (1 - clampedPct / 100);

  const chartData = trend.map((pt, idx) => ({
    label: `v${pt.state_version}`,
    idx,
    risk: pt.risk_score_pct,
  }));

  const maxMag = Math.max(0.05, ...top_factors.map((f) => f.magnitude));

  return (
    <aside
      aria-label="30-Day MACE Risk and Explainability"
      className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 flex flex-col justify-between gap-5"
    >
      {/* Risk Gauge Header */}
      <div>
        <div className="flex items-center justify-between gap-2 border-b border-slate-200 dark:border-slate-800 pb-3 mb-4">
          <div className="flex items-center gap-2">
            <ShieldAlert className="h-4 w-4 text-slate-700 dark:text-slate-300" aria-hidden="true" />
            <h2 className="text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100">
              30-Day MACE Risk
            </h2>
          </div>
          <span className={`text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded border ${badgeClasses}`}>
            {risk_level}
          </span>
        </div>

        {/* Semi-circular Gauge */}
        <div className="flex flex-col items-center">
          <svg viewBox="0 0 160 95" className="w-48 h-28 overflow-visible" role="img" aria-label={`30-Day MACE Risk ${risk_score_pct.toFixed(1)}%`}>
            <path
              d="M 16 82 A 64 64 0 0 1 144 82"
              fill="none"
              stroke="currentColor"
              className="text-slate-200 dark:text-slate-800"
              strokeWidth="12"
              strokeLinecap="round"
            />
            <path
              d="M 16 82 A 64 64 0 0 1 144 82"
              fill="none"
              stroke={strokeColor}
              strokeWidth="12"
              strokeLinecap="round"
              strokeDasharray={semiCircumference}
              strokeDashoffset={dashOffset}
              style={{ transition: 'stroke-dashoffset 0.5s ease' }}
            />
            <text
              x="80"
              y="68"
              textAnchor="middle"
              className="fill-slate-900 dark:fill-slate-50 font-mono font-bold text-2xl"
              style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              {risk_score_pct.toFixed(1)}%
            </text>
            <text x="80" y="86" textAnchor="middle" className="fill-slate-500 dark:fill-slate-400 text-[10px]">
              Hybrid XGBoost–LSTM
            </text>
          </svg>

          <div className="w-full grid grid-cols-3 gap-2 mt-2 pt-3 border-t border-slate-100 dark:border-slate-800 text-center">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-400">XGBoost</div>
              <div className="text-xs font-mono font-semibold text-slate-700 dark:text-slate-200">
                {(xgb_prob * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-400">LSTM Seq</div>
              <div className="text-xs font-mono font-semibold text-slate-700 dark:text-slate-200">
                {(lstm_prob * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-400">Static Calc</div>
              <div className="text-xs font-mono font-semibold text-slate-500 dark:text-slate-400">
                {baseline_static_pct.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Trend Sparkline */}
      <div>
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300">
            <TrendingUp className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
            Recent Snapshot Trend
          </span>
          <span className="text-[11px] font-mono text-slate-400">{trend.length} pts</span>
        </div>
        <div className="h-24 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
              <defs>
                <linearGradient id="riskTrendGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={strokeColor} stopOpacity={0.25} />
                  <stop offset="95%" stopColor={strokeColor} stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="label" tick={{ fontSize: 10 }} stroke="#94a3b8" tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} stroke="#94a3b8" tickLine={false} />
              <Tooltip
                contentStyle={{
                  fontSize: '12px',
                  borderRadius: '6px',
                  border: '1px solid #cbd5e1',
                }}
                formatter={(val: number) => [`${val.toFixed(1)}%`, '30-Day MACE']}
              />
              <Area
                type="monotone"
                dataKey="risk"
                stroke={strokeColor}
                strokeWidth={2}
                fill="url(#riskTrendGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* SHAP Explainability Panel */}
      <div>
        <div className="flex items-center justify-between text-xs mb-2.5">
          <span className="font-semibold text-slate-800 dark:text-slate-200">
            Top Contributing Factors (SHAP)
          </span>
          <span className="text-[10px] font-mono text-slate-400">{latency_ms.toFixed(1)} ms</span>
        </div>

        <div className="space-y-2.5">
          {top_factors.map((factor) => {
            const widthPct = Math.max(8, (factor.magnitude / maxMag) * 100);
            const increases = factor.direction === 'increases_risk';
            return (
              <div key={factor.feature} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-700 dark:text-slate-300 truncate pr-2">
                    {factor.display_name}
                  </span>
                  <span className="font-mono text-[11px] tabular-nums text-slate-500 dark:text-slate-400 shrink-0">
                    {factor.value} ({increases ? '+' : ''}
                    {factor.shap_value.toFixed(3)})
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${increases ? 'bg-rose-600 dark:bg-rose-500' : 'bg-teal-600 dark:bg-teal-500'}`}
                    style={{ width: `${widthPct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Model Validation Footer */}
      {modelMetrics && (
        <div className="pt-3 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
          <span className="inline-flex items-center gap-1">
            <Cpu className="h-3.5 w-3.5" aria-hidden="true" />
            Test AUC <strong className="font-mono text-slate-800 dark:text-slate-200">{modelMetrics.auc_hybrid.toFixed(3)}</strong>
          </span>
          <span>
            FP Reduction{' '}
            <strong className="font-mono text-teal-700 dark:text-teal-400">
              -{modelMetrics.false_positive_reduction_pct.toFixed(0)}%
            </strong>{' '}
            vs static
          </span>
        </div>
      )}
    </aside>
  );
}
