"""EXPERIMENTAL: sensory habituation applied at the sensory encoder.

    dh/dt = (1 - h) / tau_recover - (rate / rate_ref) * h / tau_habituate
    effective_rate = h * rate

This is a phenomenological rule (cf. Das et al. 2011 for olfactory habituation in
Drosophila), NOT a mechanism present in the connectome model. Off by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Habituation:
    tau_habituate: float = 20.0  # s of sustained full-rate input to habituate by 1/e
    tau_recover: float = 60.0  # s
    rate_ref: float = 100.0
    min_gain: float = 0.2
    gains: dict[str, float] = field(default_factory=dict)

    def apply(self, rates: dict[str, float], dt: float) -> dict[str, float]:
        out = {}
        for key, rate in rates.items():
            h = self.gains.get(key, 1.0)
            h += dt * ((1 - h) / self.tau_recover - (rate / self.rate_ref) * h / self.tau_habituate)
            h = min(1.0, max(self.min_gain, h))
            self.gains[key] = h
            out[key] = rate * h
        return out

    def reset(self) -> None:
        self.gains.clear()
