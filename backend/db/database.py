"""
Database Layer (PostgreSQL primary, SQLite fallback for zero-config local dev/tests).

Stores patient records, active medication regimens, and versioned TwinState
snapshots (timestamp, patient_id, vector, source, statuses, risk_score_pct).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Generator, List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, create_engine, desc
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from backend.twin.schema import TwinState

DEFAULT_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./backend/cardiotwin.db")


def _build_engine(db_url: str = DEFAULT_DB_URL):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args, pool_pre_ping=True)


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class PatientRecord(Base):
    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mrn: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True)
    primary_diagnosis: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    active_medications: Mapped[Dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    snapshots: Mapped[List["TwinStateSnapshot"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        order_by="desc(TwinStateSnapshot.timestamp)",
    )


class TwinStateSnapshot(Base):
    __tablename__ = "twin_state_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(128), ForeignKey("patients.patient_id", ondelete="CASCADE"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    source: Mapped[str] = mapped_column(String(64), default="fusion_engine", nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0.0", nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    vector: Mapped[Dict] = mapped_column(JSON, nullable=False)
    statuses: Mapped[Dict] = mapped_column(JSON, nullable=False)
    provenance: Mapped[Dict] = mapped_column(JSON, nullable=False)
    risk_score_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_level: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    patient: Mapped["PatientRecord"] = relationship(back_populates="snapshots")

    def to_twin_state(self) -> TwinState:
        return TwinState(
            schema_version=self.schema_version,
            state_version=self.state_version,
            patient_id=self.patient_id,
            timestamp=self.timestamp,
            source=self.source,
            vector=self.vector,
            statuses=self.statuses,
            provenance=self.provenance,
        )


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_latest_snapshot(db: Session, patient_id: str) -> Optional[TwinStateSnapshot]:
    return (
        db.query(TwinStateSnapshot)
        .filter(TwinStateSnapshot.patient_id == patient_id)
        .order_by(desc(TwinStateSnapshot.timestamp), desc(TwinStateSnapshot.id))
        .first()
    )


def get_recent_snapshots(db: Session, patient_id: str, limit: int = 12) -> List[TwinStateSnapshot]:
    rows = (
        db.query(TwinStateSnapshot)
        .filter(TwinStateSnapshot.patient_id == patient_id)
        .order_by(desc(TwinStateSnapshot.timestamp), desc(TwinStateSnapshot.id))
        .limit(limit)
        .all()
    )
    return list(reversed(rows))
