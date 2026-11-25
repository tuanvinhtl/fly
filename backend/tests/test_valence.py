import numpy as np

from dll.learning.valence import ValenceReadout


def test_groups_follow_transmitter(connectome):
    ro = ValenceReadout(connectome)
    nt = np.array(connectome.annotations["top_nt"])
    assert set(nt[ro.avoid]) == {"glutamate"}
    assert set(nt[ro.approach]) <= {"acetylcholine", "gaba"}
    assert ro.approach.size + ro.avoid.size == int((connectome.annotations["cell_class"] == "MBON").sum())


def test_valence_sign_and_steering(connectome):
    ro = ValenceReadout(connectome, k_app=1.0, k_av=1.0)
    for _ in range(100):  # approach MBONs active -> positive value
        ro.observe(ro.approach[:10], np.full(10, 1), 0.01)
    assert ro.value > 0
    assert ro.steering(odor_left=1.0, odor_right=0.9) > 0  # steer toward the left (stronger) side
    assert ro.steering(odor_left=0.9, odor_right=1.0) < 0
    assert ro.steering(0.0, 0.0) == 0.0  # no odour, no steering
    ro.reset()
    for _ in range(100):
        ro.observe(ro.avoid[:10], np.full(10, 1), 0.01)
    assert ro.value < 0
    assert ro.steering(odor_left=1.0, odor_right=0.9) < 0  # negative valence steers away
