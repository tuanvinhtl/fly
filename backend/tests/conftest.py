import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dll.brain.connectome import RAW_CONNECTIVITY, Connectome, load_connectome  # noqa: E402
from dll.config import load_settings  # noqa: E402

logging.basicConfig(level=logging.WARNING)


@pytest.fixture(scope="session")
def connectome() -> Connectome:
    s = load_settings()
    if not (s.raw_dir / RAW_CONNECTIVITY).is_file() and not s.cache_dir.is_dir():
        pytest.skip("connectome data not downloaded — run `python -m scripts.fetch_data`")
    return load_connectome(s.raw_dir, s.cache_dir)


@pytest.fixture(scope="session")
def sugar_subnet(connectome: Connectome) -> tuple[Connectome, np.ndarray, np.ndarray]:
    """Left sugar GRNs, everything 2 synaptic hops downstream of them, and MN9.

    Returns (subnetwork, input indices, MN9 indices) in subnetwork coordinates.
    """
    ann = connectome.annotations
    sugar = np.flatnonzero((ann.cell_sub_class == "sugar").to_numpy() & (ann.side == "left").to_numpy())
    keep = set(sugar.tolist())
    frontier = sugar
    for _ in range(2):
        nxt = np.unique(np.concatenate([connectome.outgoing(int(i))[0] for i in frontier]))
        keep |= set(nxt.tolist())
        frontier = nxt
    mn9 = np.flatnonzero((ann.cell_type == "CB0701").to_numpy())
    keep |= set(mn9.tolist())
    keep_arr = np.array(sorted(keep))
    sub = connectome.subset(keep_arr, "sugar 2-hop subnetwork")
    pos = {int(g): i for i, g in enumerate(keep_arr)}
    return sub, np.array([pos[int(i)] for i in sugar]), np.array([pos[int(i)] for i in mn9])
