"""Interface every brain backend must implement so the organism can swap models."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class BrainModel(Protocol):
    n: int
    spike_count: np.ndarray  # cumulative spikes per neuron since reset

    @property
    def time(self) -> float: ...

    @property
    def dt(self) -> float: ...

    def reset(self, seed: int | None = None) -> None: ...

    def set_input_rates(self, idx: np.ndarray, rates_hz: np.ndarray) -> None: ...

    def step(self, n_steps: int = 1, record: bool = False) -> Any: ...

    def snapshot(self) -> Any: ...

    def restore(self, state: Any) -> None: ...
