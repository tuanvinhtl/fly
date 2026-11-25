"""Simplified Drosophila body as MuJoCo MJCF.

Units: millimetre, milligram, second (so force unit = 1e-9 N, gravity = 9810 mm/s^2).
Body proportions follow an adult female (~2.5 mm long, ~1 mg). This is NOT an
anatomically detailed model (see NeuroMechFly for that): legs have 3 actuated DoF
(coxa yaw, coxa pitch, femur-tibia), wings and proboscis have 1 DoF each.

Joint armature is increased well above the true segment inertia so that the model is
stable at a 0.2 ms physics step; this slows leg dynamics and is documented as an
approximation in ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

LEGS = ("L1", "L2", "L3", "R1", "R2", "R3")  # left/right, front/middle/hind

# leg attachment points on thorax (x forward, y left, z up), mm
_LEG_ROOT = {
    "1": (0.30, 0.18),
    "2": (0.05, 0.22),
    "3": (-0.20, 0.20),
}
# resting coxa yaw (deg): front legs point forward, hind legs backward
_LEG_YAW0 = {"1": 50.0, "2": 0.0, "3": -45.0}


@dataclass(frozen=True)
class BodyGeometry:
    stand_height: float = 0.75  # thorax centre height (mm)
    femur_len: float = 0.75
    tibia_len: float = 0.95
    thorax_radius: float = 0.38


def leg_xml(leg: str, g: BodyGeometry) -> str:
    side, pos = leg[0], leg[1]
    s = 1.0 if side == "L" else -1.0
    x, y = _LEG_ROOT[pos]
    yaw0 = _LEG_YAW0[pos] * s
    return f"""
      <body name="{leg}_coxa" pos="{x} {s * y} -0.12" euler="0 0 {yaw0}">
        <joint name="{leg}_yaw" type="hinge" axis="0 0 1" range="-60 60" class="leg"/>
        <joint name="{leg}_pitch" type="hinge" axis="{-s} 0 0" range="-30 70" class="leg"/>
        <geom name="{leg}_femur" type="capsule" fromto="0 0 0 0 {s * g.femur_len * 0.8} {g.femur_len * 0.6}" size="0.045" class="leg"/>
        <body name="{leg}_tibia" pos="0 {s * g.femur_len * 0.8} {g.femur_len * 0.6}">
          <joint name="{leg}_knee" type="hinge" axis="{-s} 0 0" range="-70 70" class="leg"/>
          <geom name="{leg}_tibia" type="capsule" fromto="0 0 0 0 {s * g.tibia_len * 0.35} {-g.tibia_len * 0.94}" size="0.035" class="leg"/>
          <geom name="{leg}_tarsus" type="sphere" pos="0 {s * g.tibia_len * 0.35} {-g.tibia_len * 0.94}" size="0.05" class="foot"/>
        </body>
      </body>"""


def obstacles_xml(boxes: list[tuple[str, float, float, float, float, float]]) -> str:
    return "".join(
        f'\n    <geom name="obs_{b[0]}" type="box" pos="{b[1]} {b[2]} {(b[6] if len(b) > 6 else 0.0) + b[5] / 2}" '
        f'size="{b[3]} {b[4]} {b[5] / 2}" contype="1" conaffinity="1" rgba="0.35 0.38 0.45 1"/>'
        for b in boxes
    )


def build_mjcf(g: BodyGeometry | None = None, boxes: list | None = None) -> str:
    g = g or BodyGeometry()
    legs = "".join(leg_xml(leg, g) for leg in LEGS)
    actuators = "\n".join(
        f'    <position name="{leg}_{j}" joint="{leg}_{j}" class="{"knee" if j == "knee" else "leg"}"/>'
        for leg in LEGS
        for j in ("yaw", "pitch", "knee")
    )
    return f"""<mujoco model="dll_fly">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.0002" gravity="0 0 -9810" integrator="implicitfast" cone="elliptic" impratio="5"/>
  <size memory="20M"/>
  <default>
    <geom density="1.0" friction="1.2 0.02 0.002" solref="0.004 1" solimp="0.9 0.95 0.001" condim="3"/>
    <default class="leg">
      <joint armature="0.02" damping="25" limited="true"/>
      <geom contype="0" conaffinity="0" rgba="0.45 0.30 0.20 1"/>
      <position kp="8000" ctrlrange="-3.0 3.0" forcerange="-20000 20000"/>
    </default>
    <default class="knee">
      <position kp="8000" ctrlrange="-3.0 3.0" forcerange="-20000 20000"/>
    </default>
    <default class="foot">
      <geom contype="1" conaffinity="1" friction="2.0 0.02 0.002" rgba="0.2 0.15 0.1 1"/>
    </default>
    <default class="body">
      <geom contype="1" conaffinity="1"/>
    </default>
  </default>
  <worldbody>
    <light name="sun" pos="0 0 200" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="500 500 0.1" rgba="0.15 0.17 0.2 1" contype="1" conaffinity="1"/>{obstacles_xml(boxes or [])}
    <body name="thorax" pos="0 0 {g.stand_height}">
      <freejoint name="root"/>
      <geom name="thorax" type="ellipsoid" size="0.45 {g.thorax_radius} {g.thorax_radius * 0.9}" class="body" rgba="0.55 0.38 0.22 1"/>
      <body name="head" pos="0.62 0 0.08">
        <geom name="head" type="sphere" size="0.30" class="body" rgba="0.6 0.35 0.2 1"/>
        <geom name="eye_L" type="ellipsoid" pos="0.05 0.2 0.04" size="0.14 0.1 0.18" contype="0" conaffinity="0" rgba="0.75 0.1 0.08 1"/>
        <geom name="eye_R" type="ellipsoid" pos="0.05 -0.2 0.04" size="0.14 0.1 0.18" contype="0" conaffinity="0" rgba="0.75 0.1 0.08 1"/>
        <site name="antenna_L" pos="0.3 0.08 0.08" size="0.03"/>
        <site name="antenna_R" pos="0.3 -0.08 0.08" size="0.03"/>
        <site name="ocellus_L" pos="0.0 0.06 0.29" size="0.02"/>
        <site name="ocellus_R" pos="0.0 -0.06 0.29" size="0.02"/>
        <body name="proboscis" pos="0.12 0 -0.22">
          <joint name="proboscis" type="slide" axis="0.3 0 -1" range="0 0.35" armature="0.01" damping="5"/>
          <geom name="proboscis" type="capsule" fromto="0 0 0 0.05 0 -0.15" size="0.04" contype="0" conaffinity="0" rgba="0.8 0.6 0.4 1"/>
          <site name="labellum" pos="0.05 0 -0.18" size="0.03"/>
        </body>
      </body>
      <body name="abdomen" pos="-0.75 0 -0.02">
        <geom name="abdomen" type="ellipsoid" size="0.62 0.36 0.34" class="body" rgba="0.5 0.36 0.24 1"/>
      </body>
      <body name="wing_L" pos="0.0 0.2 0.3">
        <joint name="wing_L" type="hinge" axis="0 0 1" range="0 80" armature="0.01" damping="5"/>
        <geom name="wing_L" type="ellipsoid" pos="-0.9 0.15 0" size="0.95 0.32 0.01" density="0.05" contype="0" conaffinity="0" rgba="0.8 0.85 0.95 0.35"/>
      </body>
      <body name="wing_R" pos="0.0 -0.2 0.3">
        <joint name="wing_R" type="hinge" axis="0 0 -1" range="0 80" armature="0.01" damping="5"/>
        <geom name="wing_R" type="ellipsoid" pos="-0.9 -0.15 0" size="0.95 0.32 0.01" density="0.05" contype="0" conaffinity="0" rgba="0.8 0.85 0.95 0.35"/>
      </body>{legs}
    </body>
  </worldbody>
  <actuator>
{actuators}
    <position name="proboscis" joint="proboscis" kp="2000" ctrlrange="0 0.35"/>
    <position name="wing_L" joint="wing_L" kp="300" ctrlrange="0 1.4"/>
    <position name="wing_R" joint="wing_R" kp="300" ctrlrange="0 1.4"/>
  </actuator>
</mujoco>
"""
