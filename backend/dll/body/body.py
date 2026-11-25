"""MuJoCo body wrapper: applies actuator targets, steps physics, reports body state."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from .fly_model import LEGS, build_mjcf

log = logging.getLogger(__name__)

# full integration state plus solver warm-start, so snapshot/restore is bit-exact
_STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION | mujoco.mjtState.mjSTATE_WARMSTART
_ATT_KP = np.array([120.0, 480.0, 0.0])
_ATT_KD = np.array([6.0, 24.0, 10.0])
_YAW_GAIN = 10.0  # -> ~1 rad/s yaw rate at full turn command
_CRUISE_MM_S = 30.0


@dataclass
class BodyState:
    position: np.ndarray  # thorax xyz (mm)
    heading: float  # yaw (rad), 0 = +x
    velocity: np.ndarray  # thorax linear velocity (mm/s)
    upright: float  # z-component of body up vector (1 = upright)


class FlyBody:
    def __init__(self, boxes: list | None = None) -> None:
        self._build(boxes or [])
        self.reset()
        log.info("FlyBody: nq=%d nu=%d dt=%.4fs mass=%.3f mg", self.model.nq, self.model.nu, self.model.opt.timestep, self.total_mass)

    def _build(self, boxes: list) -> None:
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(boxes=boxes))
        self.data = mujoco.MjData(self.model)
        self._act = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self._geom_body = {self.model.geom(i).name: i for i in range(self.model.ngeom)}
        self.thorax_id = self.model.body("thorax").id
        self.head_geom = self.model.geom("head").id
        self.sites = {n: self.model.site(n).id for n in ("antenna_L", "antenna_R", "ocellus_L", "ocellus_R", "labellum")}
        self.foot_geoms = {leg: self.model.geom(f"{leg}_tarsus").id for leg in LEGS}
        self.obstacle_geoms = {i for i in range(self.model.ngeom) if self.model.geom(i).name.startswith("obs_")}

    def rebuild(self, boxes: list) -> None:
        """Recompile with new static obstacles, preserving the fly's full physical state."""
        state, ctrl = self.get_state_vector(), self.data.ctrl.copy()
        self._build(boxes)
        self.set_state_vector(state)
        self.data.ctrl[:] = ctrl

    @property
    def total_mass(self) -> float:
        return float(self.model.body_subtreemass[self.thorax_id])

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:2] = (x, y)
        self.data.qpos[3:7] = (np.cos(heading / 2), 0, 0, np.sin(heading / 2))
        mujoco.mj_forward(self.model, self.data)

    def set_targets(self, targets: dict[str, float]) -> None:
        for name, value in targets.items():
            self.data.ctrl[self._act[name]] = value

    def step(self, n: int) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    def apply_impulse(self, force_xyz: np.ndarray, duration_steps: int) -> None:
        self.data.xfrc_applied[self.thorax_id, :3] = force_xyz
        mujoco.mj_step(self.model, self.data, nstep=duration_steps)
        self.data.xfrc_applied[self.thorax_id, :3] = 0

    def feet_down(self, floor_clearance: float = 0.12) -> bool:
        """True when at least two tarsi are within `floor_clearance` mm of the floor."""
        z = [self.data.geom_xpos[g][2] for g in self.foot_geoms.values()]
        return sum(zz < floor_clearance for zz in z) >= 2

    def step_flight(self, n: int, power: float, yaw_cmd: float, chunk: int = 5) -> None:
        """Advance n physics steps with quasi-steady flight forces on the thorax.

        APPROXIMATION (not a wing aerodynamics model): lift and thrust scale with flight
        power, linear drag sets cruise speed scaled down to the arena (30 mm/s instead of
        ~0.5 m/s), and a PD torque keeps the body level, standing in for haltere-mediated
        stabilisation. Forces are refreshed every `chunk` steps (1 ms).
        """
        weight = self.total_mass * 9810.0
        tid = self.thorax_id
        z_up = np.array([0.0, 0.0, 1.0])
        done = 0
        while done < n:
            k = min(chunk, n - done)
            R = self.data.xmat[tid].reshape(3, 3)
            up, fwd = R[:, 2], R[:, 0]
            v = self.data.qvel[0:3]
            f_h = np.array([fwd[0], fwd[1], 0.0])
            nh = np.linalg.norm(f_h)
            f_h = f_h / nh if nh > 1e-9 else f_h
            lift = weight * (0.55 + 0.7 * power)
            thrust = weight * 0.45 * power
            force = lift * z_up + thrust * f_h
            force[:2] -= (weight * 0.45 / _CRUISE_MM_S) * v[:2]
            force[2] -= (weight / 25.0) * v[2]
            # PD attitude control in the body frame. Gains from the whole-body rotational
            # inertia (~0.07 / 0.29 / 0.32 mg mm^2 roll/pitch/yaw): omega = 40 rad/s, zeta = 1.
            err = R.T @ np.cross(up, z_up)
            w_loc = self.data.qvel[3:6]
            tq_loc = _ATT_KP * err - _ATT_KD * w_loc
            tq_loc[2] += _YAW_GAIN * yaw_cmd
            # xfrc acts at the thorax's own COM; shift it to the whole-fly COM so lift and
            # thrust create no spurious pitch moment (the abdomen sits behind the thorax).
            arm = self.data.subtree_com[tid] - self.data.xipos[tid]
            self.data.xfrc_applied[tid, :3] = force
            self.data.xfrc_applied[tid, 3:] = R @ tq_loc + np.cross(arm, force)
            mujoco.mj_step(self.model, self.data, nstep=k)
            done += k
        self.data.xfrc_applied[tid, :] = 0

    def righting_torque(self, n: int) -> None:
        """Rotate an upside-down fly back onto its legs (APPROXIMATION of the leg-driven
        righting reflex; applied only while the body is inverted on the ground)."""
        tid = self.thorax_id
        R = self.data.xmat[tid].reshape(3, 3)
        w = self.data.qvel[3:6]
        # roll about the body's long axis; strong enough to lift the COM over the edge
        tq = np.array([1500.0 - 40.0 * w[0], -40.0 * w[1], -10.0 * w[2]])
        self.data.xfrc_applied[tid, 3:] = R @ tq
        self.data.xfrc_applied[tid, 2] = 0.5 * self.total_mass * 9810.0
        mujoco.mj_step(self.model, self.data, nstep=n)
        self.data.xfrc_applied[tid, :] = 0

    def state(self) -> BodyState:
        q = self.data.qpos
        w, x, y, z = q[3:7]
        heading = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        up = float(1 - 2 * (x * x + y * y))
        return BodyState(q[0:3].copy(), heading, self.data.qvel[0:3].copy(), up)

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.sites[name]].copy()

    def head_contacts(self) -> list[tuple[int, np.ndarray]]:
        """Contacts between the head and obstacle/wall geoms: (other geom id, contact pos)."""
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.head_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.head_geom else c.geom1
                if other in self.obstacle_geoms:
                    out.append((int(other), c.pos.copy()))
        return out

    def geom_poses(self) -> dict[str, tuple[list[float], list[float]]]:
        """World pose (pos, quat wxyz) of every named body geom, for rendering."""
        out = {}
        quat = np.zeros(4)
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if not name or name == "floor" or name.startswith("obs_"):
                continue
            mujoco.mju_mat2Quat(quat, self.data.geom_xmat[i])
            out[name] = (self.data.geom_xpos[i].round(4).tolist(), quat.round(4).tolist())
        return out

    def geom_specs(self) -> list[dict]:
        """Static render description of body geoms (MuJoCo conventions: capsule axis = local z)."""
        names = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"}
        out = []
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            if not name or name == "floor" or name.startswith("obs_"):
                continue
            out.append({
                "name": name,
                "type": names.get(int(self.model.geom_type[i]), "sphere"),
                "size": self.model.geom_size[i].round(5).tolist(),
                "rgba": self.model.geom_rgba[i].round(3).tolist(),
            })
        return out

    def get_state_vector(self) -> np.ndarray:
        size = mujoco.mj_stateSize(self.model, _STATE_SPEC)
        s = np.empty(size)
        mujoco.mj_getState(self.model, self.data, s, _STATE_SPEC)
        return s

    def set_state_vector(self, s: np.ndarray) -> None:
        mujoco.mj_setState(self.model, self.data, s, _STATE_SPEC)
        mujoco.mj_forward(self.model, self.data)
