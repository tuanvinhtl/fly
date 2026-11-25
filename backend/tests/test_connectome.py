import numpy as np


def test_dataset_size_matches_publication(connectome):
    # Shiu et al. 2024 Completeness_783 / Connectivity_783
    assert connectome.n_neurons == 138_639
    assert connectome.n_connections == 15_091_983


def test_signs_are_per_neuron_and_mostly_match_annotation_transmitter(connectome):
    """Shiu et al. sign each neuron's outputs by its predicted transmitter (GABA/Glu negative).

    The current annotation release's `top_nt` differs for a minority of neurons (it is a
    later aggregate), so we require >85% agreement rather than identity. See DATA_SOURCES.md.
    """
    ann = connectome.annotations
    deg = np.diff(connectome.indptr)
    pre = np.repeat(np.arange(connectome.n_neurons), deg)
    neg = np.bincount(pre, weights=(connectome.counts < 0).astype(float), minlength=connectome.n_neurons)
    has = deg > 0
    frac = np.zeros(connectome.n_neurons)
    frac[has] = neg[has] / deg[has]
    assert np.all((frac[has] == 0) | (frac[has] == 1)), "mixed-sign presynaptic neuron"
    inhib = frac == 1
    for nt, expect in (("acetylcholine", False), ("gaba", True), ("glutamate", True), ("dopamine", False)):
        m = (ann.top_nt == nt).to_numpy() & has
        assert np.mean(inhib[m] == expect) > 0.85, nt


def test_root_id_lookup_round_trip(connectome):
    for i in (0, 1234, connectome.n_neurons - 1):
        assert connectome.index_of(int(connectome.root_ids[i])) == i


def test_incoming_is_transpose_of_outgoing(connectome):
    i = int(np.flatnonzero((connectome.annotations.cell_type == "CB0701").to_numpy())[0])
    pre, w = connectome.incoming(i)
    for p, wi in zip(pre[:20], w[:20]):
        post, wp = connectome.outgoing(int(p))
        assert wp[post == i][0] == wi


def test_positions_present_for_most_neurons(connectome):
    assert np.isfinite(connectome.positions_um).all(axis=1).mean() > 0.99


def test_subset_preserves_internal_edges(sugar_subnet, connectome):
    sub, _, _ = sugar_subnet
    assert 0 < sub.n_neurons < connectome.n_neurons
    assert sub.indices.max() < sub.n_neurons
