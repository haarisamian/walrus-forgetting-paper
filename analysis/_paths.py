"""Single place for the filesystem root. The paper was produced with
ROOT=/cephfs/store/gr-mc2473/ham51 (Cambridge Swirles). Set WALRUS_PAPER_ROOT to relocate.
Sub-projects expected under ROOT: walrus/ (this repo's experiments/), realpdebench/ (PIV + sim
data, logs), walrus_model/ (checkpoints/walrus.pt), the_well/ (datasets)."""
import os
ROOT = os.environ.get("WALRUS_PAPER_ROOT", "/cephfs/store/gr-mc2473/ham51")
