"""Explicit manual-evidence ablation; does not manufacture full-foot contact GT."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hmr4d.backends.portable import _write_new_artifact

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
arrays = dict(np.load(args.input, allow_pickle=False))
meta = json.loads(arrays["motiforge_video_json"].item())
assert (
    meta["source_sha256"]
    == "80fa6378001d92df61459105ef59cb2043d5c4c141ce71a383d7bd0cf0827f1f"
)
assert meta["fps"] == 30.0 and meta["static_camera"] and not meta["mirror"]
annotations = []
for first, end, side in ((561, 579, "right"), (594, 601, "left")):
    key = f"{side}_contact_confidence"
    annotations.append(
        {
            "frames_half_open": [first, end],
            "side": side,
            "original_confidence": arrays[key][first:end].tolist(),
            "assigned_evidence_strength_not_calibrated_probability": 0.99,
        }
    )
    arrays[key][first:end] = 0.99
meta["experimental_manual_contact_evidence"] = {
    "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    "annotations": annotations,
    "limitation": "Continuous original video supports stationary forefoot candidates, not heel-flat / entire sole rigidity or exact 3D contact ground truth; kinematic guards remain active in the unchanged engine.",
}
arrays["motiforge_video_json"] = np.asarray(
    json.dumps(meta, ensure_ascii=False, sort_keys=True)
)
_write_new_artifact(args.output, arrays)
print(args.output)
