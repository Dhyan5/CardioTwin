# CardioTwin — Reduced-Order Cardiovascular Digital Twin Backend

## Architecture

```
shared/twin_schema.json   <── Single source of truth for units, bounds, & clinical thresholds
backend/
├── twin/                 # State-vector fusion engine (unit normalization, imputation, eGFR/CO derivation)
├── models/               # Hybrid XGBoost + PyTorch VitalsLSTM + LogisticRegression late-fusion stack
│   ├── artifacts/        # Trained xgb_model.json, lstm_model.pt, seq_scaler.joblib, meta_model.joblib
│   ├── features.py       # 26 tabular/derived features + 8-channel 12-step time-series window
│   ├── lstm_net.py       # 2-layer PyTorch LSTM encoder
│   ├── train.py          # Synthetic longitudinal cohort generator, trainer, & evaluator
│   └── inference.py      # <200ms inference + SHAP TreeExplainer attribution
├── pharmaco/             # 8-class cardiovascular pharmacodynamics + interaction & safety flag engine
├── whatif/               # Weekly longitudinal simulator (tau = 35d slow-onset) + Day-90 delta table
├── db/                   # PostgreSQL DDL (schema.sql) + SQLAlchemy ORM (PostgreSQL or SQLite fallback)
├── api/                  # FastAPI REST service + 5-minute background TwinState refresh loop
├── notebooks/            # model_evaluation.ipynb (ROC-AUC, calibration, FPR vs static baseline)
└── tests/                # Unit & integration test suite (pytest)
```

## Quickstart (Local Development)

### 1. Start the FastAPI Backend (Port 8000)
```bash
# Optional: point to PostgreSQL (defaults to local SQLite backend/cardiotwin.db if unset)
# set DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/cardiotwin

python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the React + TypeScript Clinical Workstation (Port 5173)
```bash
npm install --legacy-peer-deps
npm run dev
```
Open `http://localhost:5173` (or `http://localhost:5173/twin`).

### 3. Re-train Hybrid XGBoost–LSTM & Run Tests
```bash
python -m backend.models.train
python -m pytest backend/tests -v
```

## Model Evaluation Summary (Held-Out Test Set, n = 900)
| Model | ROC-AUC | Brier Score | FPR @ 80% Sensitivity |
|-------|---------|-------------|-----------------------|
| Static Baseline Calculator | 0.6441 | 0.2645 | 0.6040 |
| XGBoost Only | 0.8982 | — | — |
| PyTorch LSTM Only | 0.8980 | — | — |
| **Hybrid XGBoost–LSTM** | **0.9003** | **0.0728** | **0.0357 (-94.09% FP)** |

## Engineering Assumptions & Design Decisions
1. **Shared Schema Contract (`shared/twin_schema.json`)**: Both Python (`backend/twin/schema.py`) and TypeScript (`src/types/twin.ts`) import the same JSON schema so units, physiological clamp bounds, and `normal`/`caution`/`critical` thresholds never drift.
2. **Derived Hemodynamics & Renal Function**: `egfr` is computed via the race-free 2021 CKD-EPI creatinine equation when not directly supplied; `cardiac_output` is estimated from `ejection_fraction`, `heart_rate`, `bmi`, and `sex`.
3. **5-Minute Twin Refresh**: `backend/api/main.py` runs an `asyncio` background loop every 300 seconds (`refresh_interval_seconds`) and exposes `POST /twin/{patient_id}/refresh` and `POST /twin/ingest` for immediate updates.
4. **Pharmacodynamic Kinetics**: Dose-response uses a saturating Hill/Emax curve (`_hill_response`). Fast-onset drugs (beta-blocker, loop diuretic, CCB, digoxin, nitrate, antiplatelet) take effect immediately from Week 1; slow-onset effects (statin LDL lowering, ACEi LV remodeling) ramp via `1 - exp(-t / 35 days)`.
5. **Storage**: PostgreSQL DDL is provided in `backend/db/schema.sql`, and `backend/db/database.py` uses SQLAlchemy with `DATABASE_URL` (falling back to SQLite when PostgreSQL is unavailable locally).
