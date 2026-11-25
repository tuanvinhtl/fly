"""Ventral nerve cord (VNC) adapter: descending-neuron rates -> leg joint targets.

The FlyWire v783 brain connectome does not include the VNC, so descending neuron (DN)
activity cannot drive motor neurons through real wiring. This module is an explicit
APPROXIMATION of the VNC: a tripod-gait central pattern generator (CPG) whose
frequency, direction and left/right amplitude are set from DN firing rates.

Mapping (see brain/circuits.py for references):
  forward   = DNp09 (both sides)            -> stepping frequency
  backward  = MDN   (both sides)            -> reverses stance direction
  turning   = DNa02 (gain 1) + DNa01 (0.3)  -> ipsilateral turn (left DN > right = turn left)
  escape    = DNp01                         -> wing raise + jump impulse
  proboscis = MN9                           -> proboscis extension (motor neuron in brain)

Nothing here knows about food, light or obstacles: it only converts DN rates into
rhythmic joint targets. All goal-directedness must come from the brain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fly_model import LEGS

# tripod groups: (L1, R2, L3) in phase, (R1, L2, R3) in anti-phase
_PHASE_OFFSET = {"L1": 0.0, "R2": 0.0, "L3": 0.0, "R1": math.pi, "L2": math.pi, "R3": math.pi}


@dataclass(frozen=True)
class VNCParams:
    rate_ref: float = 60.0  # Hz; DN rate giving full command (DNp09 under drive fires 60-80 Hz)
    turn_ref: float = 30.0  # Hz of left-right DNa difference for full turn
    f_max: float = 8.0  # stepping frequency at full drive (Hz); real flies 5-15 Hz
    swing_amp: float = 0.45  # rad coxa yaw amplitude
    lift_amp: float = 0.5  # rad coxa pitch lift during swing
    stance_press: float = 0.15  # rad extra downward pitch during stance
    min_drive: float = 0.08  # below this |drive| the legs hold still
    escape_threshold: float = 20.0  # DNp01 Hz
    proboscis_threshold: float = 5.0  # MN9 Hz for full extension at 4x threshold


@dataclass
class MotorCommand:
    forward: float  # -1..1 (negative = backward)
    turn: float  # -1..1 (positive = left)
    escape: bool
    proboscis: float  # 0..1 extension


class VNC:
    def __init__(self, params: VNCParams | None = None):
        self.p = params or VNCParams()
        self.phase = 0.0

    def reset(self) -> None:
        self.phase = 0.0

    def command(self, rates: dict[str, float]) -> MotorCommand:
        p = self.p
        fwd = (rates["dn_forward_left"] + rates["dn_forward_right"]) / 2.0
        bwd = (rates["dn_backward_left"] + rates["dn_backward_right"]) / 2.0
        steer_l = rates["dn_steer_high_left"] + 0.3 * rates["dn_steer_low_left"]
        steer_r = rates["dn_steer_high_right"] + 0.3 * rates["dn_steer_low_right"]
        forward = float(np.clip((fwd - bwd) / p.rate_ref, -1.0, 1.0))
        turn = float(np.clip((steer_l - steer_r) / p.turn_ref, -1.0, 1.0))
        escape = max(rates["dn_escape_left"], rates["dn_escape_right"]) > p.escape_threshold
        mn9 = max(rates["mn_proboscis_left"], rates["mn_proboscis_right"])
        proboscis = float(np.clip(mn9 / (4 * p.proboscis_threshold), 0.0, 1.0))
        return MotorCommand(forward, turn, escape, proboscis)

    def joint_targets(self, cmd: MotorCommand, dt: float) -> dict[str, float]:
        """Advance the CPG by dt and return position targets per actuator (radians)."""
        p = self.p
        drive = max(abs(cmd.forward), abs(cmd.turn))
        active = drive > p.min_drive
        if active:
            direction = 1.0 if cmd.forward >= -0.05 else -1.0
            self.phase = (self.phase + direction * 2 * math.pi * p.f_max * drive * dt) % (2 * math.pi)
        # per-side stride amplitude: turning shortens (or reverses) the inner side's stride
        fwd_part = cmd.forward if abs(cmd.forward) > 0.05 else 0.0
        amp_l = np.clip(abs(fwd_part) - cmd.turn, -1.0, 1.0) if active else 0.0
        amp_r = np.clip(abs(fwd_part) + cmd.turn, -1.0, 1.0) if active else 0.0
        out: dict[str, float] = {}
        for leg in LEGS:
            s = 1.0 if leg[0] == "L" else -1.0
            amp = amp_l if leg[0] == "L" else amp_r
            ph = self.phase + _PHASE_OFFSET[leg]
            swing = math.sin(ph) > 0  # swing phase: leg lifted, moving forward
            # swing (sin>0): back->front; stance: front->back, pushing the body forward
            protraction = -p.swing_amp * amp * math.cos(ph)
            lift = -p.lift_amp * math.sin(ph) if (swing and active) else p.stance_press * (1 if active else 0.5)
            out[f"{leg}_yaw"] = -s * protraction
            out[f"{leg}_pitch"] = lift
            out[f"{leg}_knee"] = 0.0
        out["proboscis"] = 0.35 * cmd.proboscis
        wing = 1.3 if cmd.escape else 0.0
        out["wing_L"] = wing
        out["wing_R"] = wing
        return out
