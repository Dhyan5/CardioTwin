-- PostgreSQL Schema for CardioTwin Digital Twin State & Time-Series Storage
-- Matches SQLAlchemy models in backend/db/database.py

CREATE TABLE IF NOT EXISTS patients (
    patient_id VARCHAR(128) PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    mrn VARCHAR(64) UNIQUE,
    primary_diagnosis VARCHAR(255),
    active_medications JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS twin_state_snapshots (
    id BIGSERIAL PRIMARY KEY,
    patient_id VARCHAR(128) NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source VARCHAR(64) NOT NULL DEFAULT 'fusion_engine',
    schema_version VARCHAR(32) NOT NULL DEFAULT '1.0.0',
    state_version INTEGER NOT NULL DEFAULT 1,
    vector JSONB NOT NULL,
    statuses JSONB NOT NULL,
    provenance JSONB NOT NULL,
    risk_score_pct DOUBLE PRECISION,
    risk_level VARCHAR(16),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_twin_snapshots_patient_ts
    ON twin_state_snapshots (patient_id, timestamp DESC);
