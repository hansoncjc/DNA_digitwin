"""Write the R3.1 ground-truth coefficients (gt_config.GROUND_TRUTH) as a
mapping file, mapping/ground_truth.json, for checks through the
ground-truth mapping (R6.10, R7.2).

    python campaigns/r3.1-gt/write_mapping.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gt_config import GROUND_TRUTH  # noqa: E402
from mapping import file_sha256, save_mapping  # noqa: E402

OUT = HERE / "mapping" / "ground_truth.json"


def main():
    src = HERE / "gt_config.py"
    m = save_mapping(OUT, GROUND_TRUTH, source={
        "run": "R3.1", "name": "GROUND_TRUTH",
        "file": src.relative_to(REPO).as_posix(), "sha256": file_sha256(src),
        "note": "coefficients that generated the R3.1 virtual ground truth",
    })
    print(f"[mapping] wrote {OUT} (sha256 {m.sha256}): {m.coefficients}")


if __name__ == "__main__":
    main()
