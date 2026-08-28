from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hmr4d.backends.motiforge import (
    COMMON_ASSETS,
    _ffmpeg_command,
    _portable_prediction,
    _write_portable_npz,
    diagnose,
)


class MotiForgeBackendTests(unittest.TestCase):
    def test_ffmpeg_normalization_contract(self) -> None:
        command = _ffmpeg_command(
            Path("input video.mov"),
            Path("output.mp4"),
            mirror=True,
            max_frames=90,
        )
        self.assertEqual(command[0], "ffmpeg")
        self.assertEqual(command[command.index("-vf") + 1], "fps=30,hflip")
        self.assertEqual(command[command.index("-frames:v") + 1], "90")
        self.assertIn("-an", command)

    def test_doctor_reports_every_missing_required_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = diagnose(Path(temporary), require_cuda=False, check_runtime=False)
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["errors"]), len(COMMON_ASSETS))

    def test_asset_only_doctor_accepts_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in COMMON_ASSETS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            report = diagnose(root, require_cuda=False, check_runtime=False)
        self.assertTrue(report["ok"])

    def test_portable_artifact_contract_contains_world_joints_and_revisions(self) -> None:
        import torch

        class Endecoder:
            @staticmethod
            def fk_v2(**kwargs):
                return torch.zeros((1, 3, 22, 3))

        class Pipeline:
            endecoder = Endecoder()

        class Model:
            pipeline = Pipeline()

        params = {
            "body_pose": torch.zeros((3, 63)),
            "global_orient": torch.zeros((3, 3)),
            "transl": torch.zeros((3, 3)),
            "betas": torch.zeros((3, 10)),
        }
        portable = _portable_prediction(
            pred={
                "smpl_params_global": params,
                "smpl_params_incam": params,
                "K_fullimg": torch.eye(3).repeat(3, 1, 1),
            },
            model=Model(),
            detach_to_cpu=lambda value: value,
            item={"source": "/videos/clip.mp4", "source_sha256": "source-sha"},
            options={"simple_vo_workers": 4},
            revision="gvhmr-revision",
            backend_id="backend-revision",
            normalized_frames=3,
        )

        self.assertEqual(tuple(portable["pred_w_j3d"].shape), (3, 22, 3))
        metadata = portable["motiforge_video"]
        self.assertEqual(metadata["gvhmr_revision"], "gvhmr-revision")
        self.assertEqual(metadata["gvhmr_backend_revision"], "backend-revision")
        self.assertEqual(metadata["simple_vo_workers"], 4)

    def test_portable_npz_is_pickle_free_and_preserves_nested_parameters(self) -> None:
        import json

        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.npz"
            _write_portable_npz(
                path,
                {
                    "pred_w_j3d": np.zeros((2, 22, 3), dtype=np.float32),
                    "smpl_params_global": {
                        "body_pose": np.zeros((2, 63), dtype=np.float32),
                        "transl": np.zeros((2, 3), dtype=np.float32),
                    },
                    "motiforge_video": {"protocol": 3, "fps": 30.0},
                },
            )

            with np.load(path, allow_pickle=False) as archive:
                self.assertIn("pred_w_j3d", archive.files)
                self.assertIn("smpl_params_global.body_pose", archive.files)
                metadata = json.loads(str(archive["motiforge_video_json"].item()))
            self.assertEqual(metadata["protocol"], 3)


if __name__ == "__main__":
    unittest.main()
