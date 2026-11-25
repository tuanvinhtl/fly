"""Download the connectome and annotation files (≈135 MB) and verify checksums.

    python -m scripts.fetch_data            # into $DLL_DATA_DIR/raw (default ../data/raw)

Files are pinned by SHA-256 so a silently changed upstream file is detected.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dll.brain.connectome import build_cache  # noqa: E402
from dll.config import load_settings  # noqa: E402

log = logging.getLogger("fetch_data")

SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"
ANN = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files"

FILES = {
    "Completeness_783.csv": (f"{SHIU}/Completeness_783.csv", "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311"),
    "Connectivity_783.parquet": (f"{SHIU}/Connectivity_783.parquet", "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"),
    "flywire_annotations_783.tsv": (f"{ANN}/Supplemental_file1_neuron_annotations.tsv", "b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd"),
    "SHIU_LICENSE": (f"{SHIU}/LICENSE", None),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = load_settings()
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in FILES.items():
        dest = s.raw_dir / name
        if dest.is_file() and (digest is None or sha256(dest) == digest):
            log.info("ok (cached) %s", name)
            continue
        log.info("downloading %s", url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        if digest is not None and sha256(tmp) != digest:
            tmp.unlink()
            log.error("checksum mismatch for %s — upstream file changed; refusing to use it", name)
            return 1
        tmp.rename(dest)
        log.info("ok %s (%.1f MB)", name, dest.stat().st_size / 1e6)
    build_cache(s.raw_dir, s.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
