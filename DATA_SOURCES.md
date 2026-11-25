# Data sources

Everything the simulator knows about the fly's nervous system comes from the files
below. Nothing is synthesised. `backend/scripts/fetch_data.py` downloads them and checks
each one against a pinned SHA-256, so an upstream change is detected instead of being
silently used.

| File (in `data/raw/`) | Size | Origin | Licence / terms | SHA-256 |
|---|---|---|---|---|
| `Completeness_783.csv` | 3.3 MB | Shiu et al. 2024 model repository, [`philshiu/Drosophila_brain_model`](https://github.com/philshiu/Drosophila_brain_model) | MIT (repository licence, Philip Shiu & Nico Spiller) | `bbb847a4…a311` |
| `Connectivity_783.parquet` | 100.8 MB | same | MIT (repository); derived from FlyWire | `efeb23fb…0347` |
| `flywire_annotations_783.tsv` | 31.7 MB | [`flyconnectome/flywire_annotations`](https://github.com/flyconnectome/flywire_annotations), `supplemental_files/Supplemental_file1_neuron_annotations.tsv` | FlyWire public-release terms; cite as below | `b214970b…683cd` |

The full hashes are in `fetch_data.py`.

## What each file contains

**Completeness_783.csv**: 138,639 FlyWire root IDs (materialization 783). The row order
defines the model's neuron index.

**Connectivity_783.parquet**: 15,091,983 directed neuron pairs with
`Presynaptic_Index`, `Postsynaptic_Index`, `Connectivity` (synapse count) and
`Excitatory x Connectivity` (signed count). Summed |signed count| = 54,492,922
synapses. Shiu et al. assign signs per *presynaptic neuron* from its predicted
neurotransmitter (Eckstein et al. 2024): GABA and glutamate are negative; acetylcholine,
dopamine, serotonin and octopamine are positive.

**flywire_annotations_783.tsv**: 139,248 rows of hierarchical annotations (Schlegel et
al. 2024): `super_class`, `cell_class`, `cell_sub_class`, `cell_type`, `side`, `top_nt`,
`flow`, plus a representative position (`pos_x/y/z`, FAFB14 voxels of 4×4×40 nm).
138,625 of the 138,639 model neurons have an annotation row.

## Verified properties (`backend/tests/test_connectome.py`)

- 138,639 neurons and 15,091,983 connections, matching the published model.
- Every presynaptic neuron's outgoing signs are uniform; no neuron has mixed signs.
- Agreement between Shiu's signs and the annotation release's `top_nt`:
  acetylcholine 98.0% excitatory, GABA 92.9% inhibitory, glutamate 88.1% inhibitory,
  dopamine 99.9% excitatory. The remainder reflect later revisions of the transmitter
  predictions in the annotation release. **The simulator uses Shiu's signs unchanged**,
  so that it stays comparable with the published model.
- 99%+ of neurons have a finite position (used only for visualisation).

## Derived cache (`data/cache/`)

`connectome.build_cache` converts the raw files once into
`flywire783_v2.npz` (CSR arrays with presynaptic rows, int32 signed counts, float32
positions in µm) plus `flywire783_v2_annotations.parquet`. It takes about 1 s to build and
about 0.1 s to load.

## Sources evaluated but not downloaded

| Source | Status in this environment | Why not used |
|---|---|---|
| FlyWire Codex (codex.flywire.ai) | host not reachable through this environment's network policy | The same v783 data is available through the two GitHub repositories above. |
| Zenodo (Dorkenwald et al. data release) | host not reachable | Same reason; the synapse-level tables (~GBs) are not needed for a neuron-level LIF model. |
| NeuroMechFly / FlyGym (`flygym` 2.1.1 on PyPI) | package index reachable | See RESEARCH.md §3: a simplified body was built instead for the MVP. |
| Virtual Fly Brain | not queried | The annotation table already carries `vfb_id` / `fbbt_id` cross-references. |
| MANC (male adult nerve cord connectome) | not downloaded | Would replace the VNC approximation; it needs neuron matching between brain and VNC datasets (future work). |

## Required citations

- Dorkenwald, S. et al. *Neuronal wiring diagram of an adult brain.* Nature 634, 124–138 (2024).
- Schlegel, P. et al. *Whole-brain annotation and multi-connectome cell typing of Drosophila.* Nature 634, 139–152 (2024).
- Shiu, P. K. et al. *A Drosophila computational brain model reveals sensorimotor processing.* Nature 634, 210–219 (2024).
- Eckstein, N. et al. *Neurotransmitter classification from electron microscopy images at synaptic sites in Drosophila melanogaster.* Cell 187 (2024).
- Matsliah, A. et al. *Neuronal parts list and wiring diagram for a visual system.* Nature 634 (2024). This is requested by the annotation repository for its newer versions.

Please also follow the citation guidance in the FlyWire annotation repository README,
which changes between annotation versions.
