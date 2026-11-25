import math

import numpy as np

from dll.body.body import FlyBody
from dll.body.vnc import VNC, MotorCommand


def _drive(cmd: MotorCommand, seconds: float = 2.0, boxes=None):
    body, vnc = FlyBody(boxes), VNC()
    n = int(round(0.01 / body.dt))
    for _ in range(int(seconds / 0.01)):
        body.set_targets(vnc.joint_targets(cmd, 0.01))
        body.step(n)
    return body.state()


def test_stands_upright_without_drive():
    s = _drive(MotorCommand(0, 0, False, 0))
    assert s.upright > 0.95
    assert s.position[2] > 0.3  # thorax supported by legs, not lying on the floor
    assert np.linalg.norm(s.position[:2]) < 0.5


def test_forward_command_walks_forward():
    s = _drive(MotorCommand(1.0, 0, False, 0))
    assert s.position[0] > 8.0  # mm in 2 s along initial heading (+x)
    assert s.upright > 0.9


def test_backward_command_walks_backward():
    s = _drive(MotorCommand(-1.0, 0, False, 0))
    assert s.position[0] < -5.0


def test_turn_sign_convention():
    left = _drive(MotorCommand(0.0, 0.8, False, 0))
    right = _drive(MotorCommand(0.0, -0.8, False, 0))
    assert left.heading > math.radians(30)
    assert right.heading < -math.radians(30)


def test_proboscis_follows_command():
    body, vnc = FlyBody(), VNC()
    body.set_targets(vnc.joint_targets(MotorCommand(0, 0, False, 1.0), 0.01))
    body.step(500)
    assert body.data.joint("proboscis").qpos[0] > 0.25


def test_obstacle_blocks_and_rebuild_preserves_state():
    body = FlyBody()
    body.step(200)
    before = body.get_state_vector()
    body.rebuild([("box", 3.0, 0.0, 0.5, 3.0, 2.0)])
    np.testing.assert_allclose(body.get_state_vector(), before)
    vnc = VNC()
    for _ in range(300):
        body.set_targets(vnc.joint_targets(MotorCommand(1.0, 0, False, 0), 0.01))
        body.step(50)
    assert body.state().position[0] < 3.0  # wall at x = 2.5 stops it
    assert body.head_contacts() or body.state().position[0] > 1.5


def test_vnc_maps_dn_rates():
    vnc = VNC()
    base = {k: 0.0 for k in (
        "dn_forward_left", "dn_forward_right", "dn_backward_left", "dn_backward_right",
        "dn_steer_high_left", "dn_steer_high_right", "dn_steer_low_left", "dn_steer_low_right",
        "dn_escape_left", "dn_escape_right", "mn_proboscis_left", "mn_proboscis_right")}
    assert vnc.command(base).forward == 0
    assert vnc.command({**base, "dn_forward_left": 60, "dn_forward_right": 60}).forward == 1.0
    assert vnc.command({**base, "dn_backward_left": 60, "dn_backward_right": 60}).forward == -1.0
    assert vnc.command({**base, "dn_steer_high_left": 30}).turn == 1.0
    assert vnc.command({**base, "dn_steer_high_right": 15}).turn == -0.5
    assert vnc.command({**base, "dn_escape_left": 30}).escape
    assert vnc.command({**base, "mn_proboscis_right": 20}).proboscis == 1.0
