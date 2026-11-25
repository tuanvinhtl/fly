"""Runtime configuration from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str
    seed: int
    brain_preset: str
    brain_engine: str
    log_level: str
    cors_origins: tuple[str, ...]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"


def load_settings() -> Settings:
    data_dir = Path(_env("DLL_DATA_DIR", str(_REPO / "data")))
    return Settings(
        data_dir=data_dir,
        database_url=_env("DLL_DATABASE_URL", f"sqlite:///{data_dir / 'runs' / 'dll.sqlite3'}"),
        seed=int(_env("DLL_SEED", "42")),
        brain_preset=_env("DLL_BRAIN_PRESET", "al_std"),
        brain_engine=_env("DLL_BRAIN_ENGINE", "numba"),
        log_level=_env("DLL_LOG_LEVEL", "INFO"),
        cors_origins=tuple(o for o in _env("DLL_CORS_ORIGINS", "http://localhost:5173").split(",") if o),
    )
