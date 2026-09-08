"""Prepare the fixed first-frame camera diagnostic from real cached predictions.

This is a case-study input builder, not physical camera calibration. World and
incam predictions can disagree even when the video used a fixed physical camera.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hmr4d.backends.portable import _load_body_model, _write_new_artifact

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--keypoints", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
p = dict(np.load(args.input, allow_pickle=False))
meta = json.loads(p["motiforge_video_json"].item())
assert (
    meta["source_sha256"]
    == "80fa6378001d92df61459105ef59cb2043d5c4c141ce71a383d7bd0cf0827f1f"
)
assert meta["fps"] == 30.0 and meta["static_camera"] and not meta["mirror"]
for name in ("body_pose", "betas"):
    np.testing.assert_array_equal(
        p[f"smpl_params_global.{name}"], p[f"smpl_params_incam.{name}"]
    )
world = p["pred_w_j3d"].astype(np.float64)
K = p["K_fullimg"]
np.testing.assert_array_equal(K, np.broadcast_to(K[:1], K.shape))
kp = torch.load(args.keypoints, map_location="cpu", weights_only=True).numpy()
assert kp.shape == (len(world), 17, 3) and np.isfinite(kp).all()
rw = Rotation.from_rotvec(p["smpl_params_global.global_orient"]).as_matrix()
rc = Rotation.from_rotvec(p["smpl_params_incam.global_orient"]).as_matrix()
rot = rc @ rw.transpose(0, 2, 1)
camera_root = (
    world[:, 0] - p["smpl_params_global.transl"] + p["smpl_params_incam.transl"]
)
camera = np.einsum("tij,tkj->tki", rot, world - world[:, :1]) + camera_root[:, None]
trans = camera_root - np.einsum("tij,tj->ti", rot, world[:, 0])
model = _load_body_model(args.model).eval().cpu()
frames = np.asarray([0, 498, 499, 553, 554, 561, 578, 594, 600, 659])
with torch.no_grad():
    params = {
        name: torch.tensor(p[f"smpl_params_incam.{name}"][frames])
        for name in ("body_pose", "betas", "global_orient", "transl")
    }
    error = float(np.abs(model(**params).joints[:, :22].numpy() - camera[frames]).max())
assert error < 1e-4, error
buffer = np.arange(360, 750)
_write_new_artifact(
    args.output,
    {
        "frames": buffer,
        "world_body22_yup": world[buffer],
        "incam_body22": camera[buffer],
        "K": K[0],
        "keypoints_coco17": kp[buffer],
        "body22_projection_indices": np.array(
            [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
        ),
        "coco17_projection_indices": np.arange(5, 17),
        "frame0_root_extrinsic.R": rot[0],
        "frame0_root_extrinsic.t": trans[0],
        "source_video_sha256": np.asarray(meta["source_sha256"]),
    },
)
print(json.dumps({"output": str(args.output), "incam_fk_max_error_m": error}))
