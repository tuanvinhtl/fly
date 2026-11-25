"""Experiment persistence (PostgreSQL in Docker, SQLite fallback for local dev/tests)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, LargeBinary, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    seed: Mapped[int] = mapped_column(Integer)
    preset: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.UTC))
    initial_world: Mapped[dict] = mapped_column(JSON)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="running")
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)


class Sample(Base):
    """Organism state at 10 Hz."""

    __tablename__ = "samples"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    t: Mapped[float] = mapped_column(Float)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    heading: Mapped[float] = mapped_column(Float)
    speed: Mapped[float] = mapped_column(Float)
    energy: Mapped[float] = mapped_column(Float)
    health: Mapped[float] = mapped_column(Float)
    forward: Mapped[float] = mapped_column(Float)
    turn: Mapped[float] = mapped_column(Float)
    proboscis: Mapped[float] = mapped_column(Float)
    total_spikes: Mapped[int] = mapped_column(Integer)
    motor: Mapped[dict] = mapped_column(JSON)
    sensory: Mapped[dict] = mapped_column(JSON)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    t: Mapped[float] = mapped_column(Float)
    tick: Mapped[int] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String(20), index=True)
    name: Mapped[str] = mapped_column(String(80))
    data: Mapped[dict] = mapped_column(JSON)


class Command(Base):
    """Every command applied, with its tick, for deterministic replay."""

    __tablename__ = "commands"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    tick: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON)


class Memory(Base):
    """A saved mushroom-body memory: the KC->MBON weight scales of an organism."""

    __tablename__ = "memories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=lambda: dt.datetime.now(dt.UTC))
    preset: Mapped[str] = mapped_column(String(50))
    source_experiment_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_t: Mapped[float] = mapped_column(Float, default=0.0)
    n_connections: Mapped[int] = mapped_column(Integer)
    conn_hash: Mapped[str] = mapped_column(String(64))
    depressed: Mapped[int] = mapped_column(Integer)
    mean_weight: Mapped[float] = mapped_column(Float)
    weights: Mapped[bytes] = mapped_column(LargeBinary)  # float32 scale per plastic connection


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine


def make_sessionmaker(engine: Engine) -> sessionmaker:
    return sessionmaker(engine, expire_on_commit=False)
