"""Connectome loading.

Source data (see DATA_SOURCES.md):
  * Completeness_783.csv / Connectivity_783.parquet  — Shiu et al. 2024 (Nature),
    derived from the FlyWire v783 whole-brain connectome (Dorkenwald et al. 2024).
    Signed synapse counts: "Excitatory x Connectivity" = synapse count, negative
    when the presynaptic neuron's predicted transmitter is GABA or glutamate.
  * flywire_annotations_783.tsv — Schlegel et al. 2024 cell-type annotations.

The raw files are converted once into a compact cache (CSR, pre-synaptic rows) so
that start-up does not re-parse 15 M rows of parquet.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RAW_COMPLETENESS = "Completeness_783.csv"
RAW_CONNECTIVITY = "Connectivity_783.parquet"
RAW_ANNOTATIONS = "flywire_annotations_783.tsv"
CACHE_VERSION = 2

ANNOTATION_COLUMNS = [
    "super_class",
    "cell_class",
    "cell_sub_class",
    "cell_type",
    "top_nt",
    "side",
    "flow",
]


@dataclass(frozen=True)
class Connectome:
    """Directed, signed synaptic graph stored as CSR with presynaptic rows.

    indptr/indices/counts follow scipy CSR conventions: the post-synaptic targets of
    neuron i are indices[indptr[i]:indptr[i+1]] with signed synapse counts counts[...].
    """

    name: str
    root_ids: np.ndarray  # int64 FlyWire root id per model index
    indptr: np.ndarray  # int64, len N+1
    indices: np.ndarray  # int32 post-synaptic index
    counts: np.ndarray  # int32 signed synapse count (negative = inhibitory)
    annotations: pd.DataFrame  # indexed by model index, columns ANNOTATION_COLUMNS
    positions_um: np.ndarray  # (N, 3) float32 representative point in FAFB space, micrometres (NaN if unknown)

    @property
    def n_neurons(self) -> int:
        return int(self.root_ids.shape[0])

    @property
    def n_connections(self) -> int:
        return int(self.indices.shape[0])

    @property
    def n_synapses(self) -> int:
        return int(np.abs(self.counts).sum())

    def index_of(self, root_id: int) -> int:
        idx = np.searchsorted(self._sorted_ids, root_id)
        if idx >= self.n_neurons or self._sorted_ids[idx] != root_id:
            raise KeyError(f"root id {root_id} not in connectome {self.name}")
        return int(self._sorted_order[idx])

    @property
    def _sorted_order(self) -> np.ndarray:
        cached = self.__dict__.get("_order")
        if cached is None:
            cached = np.argsort(self.root_ids)
            object.__setattr__(self, "_order", cached)
        return cached

    @property
    def _sorted_ids(self) -> np.ndarray:
        cached = self.__dict__.get("_sids")
        if cached is None:
            cached = self.root_ids[self._sorted_order]
            object.__setattr__(self, "_sids", cached)
        return cached

    def outgoing(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.indptr[i], self.indptr[i + 1]
        return self.indices[s:e], self.counts[s:e]

    def incoming(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Presynaptic partners of i (O(E) scan on first call, then cached CSC)."""
        csc = self.__dict__.get("_csc")
        if csc is None:
            pre = np.repeat(np.arange(self.n_neurons, dtype=np.int32), np.diff(self.indptr))
            order = np.argsort(self.indices, kind="stable")
            c_indptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
            np.cumsum(np.bincount(self.indices, minlength=self.n_neurons), out=c_indptr[1:])
            csc = (c_indptr, pre[order], self.counts[order])
            object.__setattr__(self, "_csc", csc)
        c_indptr, c_pre, c_counts = csc
        s, e = c_indptr[i], c_indptr[i + 1]
        return c_pre[s:e], c_counts[s:e]

    def subset(self, keep: np.ndarray, name: str) -> "Connectome":
        """Induced sub-graph on the (sorted, unique) model indices in ``keep``."""
        keep = np.unique(keep)
        remap = np.full(self.n_neurons, -1, dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        pre = np.repeat(np.arange(self.n_neurons), np.diff(self.indptr))
        mask = (remap[pre] >= 0) & (remap[self.indices] >= 0)
        new_pre = remap[pre[mask]]
        new_post = remap[self.indices[mask]].astype(np.int32)
        new_counts = self.counts[mask]
        indptr = np.zeros(keep.size + 1, dtype=np.int64)
        np.cumsum(np.bincount(new_pre, minlength=keep.size), out=indptr[1:])
        order = np.argsort(new_pre, kind="stable")
        ann = self.annotations.iloc[keep].reset_index(drop=True)
        return Connectome(name, self.root_ids[keep], indptr, new_post[order], new_counts[order], ann, self.positions_um[keep])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cache(raw_dir: Path, cache_dir: Path) -> Path:
    """Convert raw Shiu/FlyWire files into the compact cache. Returns cache path."""
    comp_path = raw_dir / RAW_COMPLETENESS
    con_path = raw_dir / RAW_CONNECTIVITY
    ann_path = raw_dir / RAW_ANNOTATIONS
    for p in (comp_path, con_path, ann_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing — run `python -m scripts.fetch_data` first")

    t0 = time.perf_counter()
    comp = pd.read_csv(comp_path, index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    n = root_ids.size
    con = pd.read_parquet(
        con_path, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
    )
    pre = con["Presynaptic_Index"].to_numpy(np.int64)
    post = con["Postsynaptic_Index"].to_numpy(np.int32)
    w = con["Excitatory x Connectivity"].to_numpy(np.int32)
    if pre.max() >= n or post.max() >= n:
        raise ValueError("connectivity indices exceed neuron table size")
    order = np.argsort(pre, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])

    ann = pd.read_csv(
        ann_path, sep="\t", low_memory=False, usecols=["root_id", "pos_x", "pos_y", "pos_z", *ANNOTATION_COLUMNS]
    )
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(root_ids)
    # FAFB14 voxel = 4 x 4 x 40 nm
    pos = (ann[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float64) * np.array([4.0, 4.0, 40.0]) / 1000.0).astype(np.float32)
    ann = ann[ANNOTATION_COLUMNS]
    ann = ann.astype("string").fillna("").reset_index(drop=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"flywire783_v{CACHE_VERSION}.npz"
    np.savez(
        out,
        root_ids=root_ids,
        indptr=indptr,
        indices=post[order],
        counts=w[order],
        positions_um=pos,
        sha_connectivity=np.array(_sha256(con_path)),
    )
    ann.to_parquet(cache_dir / f"flywire783_v{CACHE_VERSION}_annotations.parquet")
    log.info(
        "built connectome cache: %d neurons, %d connections in %.1fs -> %s",
        n,
        post.size,
        time.perf_counter() - t0,
        out,
    )
    return out


def load_connectome(raw_dir: Path, cache_dir: Path) -> Connectome:
    cache = cache_dir / f"flywire783_v{CACHE_VERSION}.npz"
    ann_cache = cache_dir / f"flywire783_v{CACHE_VERSION}_annotations.parquet"
    if not cache.is_file() or not ann_cache.is_file():
        build_cache(raw_dir, cache_dir)
    t0 = time.perf_counter()
    z = np.load(cache)
    ann = pd.read_parquet(ann_cache)
    c = Connectome(
        name="FlyWire v783 (Shiu et al. 2024 signed weights)",
        root_ids=z["root_ids"],
        indptr=z["indptr"],
        indices=z["indices"],
        counts=z["counts"],
        annotations=ann,
        positions_um=z["positions_um"],
    )
    log.info("loaded connectome: %d neurons, %d connections (%.2fs)", c.n_neurons, c.n_connections, time.perf_counter() - t0)
    return c
