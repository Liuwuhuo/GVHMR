from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hmr4d.backends.motiforge import (
    COMMON_ASSETS,
    _ffmpeg_command,
    _portable_prediction,
    _stabilize_world_ground,
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
                "net_outputs": {
                    "static_conf_logits": torch.full((1, 3, 6), 10.0),
                },
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
        self.assertEqual(tuple(portable["left_contact_confidence"].shape), (3,))
        self.assertEqual(tuple(portable["floor_correction_y"].shape), (3,))
        metadata = portable["motiforge_video"]
        self.assertEqual(metadata["gvhmr_revision"], "gvhmr-revision")
        self.assertEqual(metadata["gvhmr_backend_revision"], "backend-revision")
        self.assertEqual(metadata["simple_vo_workers"], 4)
        self.assertEqual(metadata["ground_stabilization"]["version"], "contact-floor-v1")

    def test_ground_stabilization_removes_floor_drift_but_preserves_flight(self) -> None:
        import numpy as np

        frames = 90
        joints = np.zeros((frames, 22, 3), dtype=np.float32)
        joints[:, :, 1] = 1.0
        drift = np.linspace(0.0, 0.12, frames, dtype=np.float32)
        for index in (7, 10, 8, 11):
            joints[:, index, 1] = drift
        flight = slice(36, 51)
        joints[flight, (7, 10, 8, 11), 1] += 0.16
        transl = np.zeros((frames, 3), dtype=np.float32)
        transl[:, 1] = 1.0 + drift
        left = np.ones(frames, dtype=np.float32)
        right = np.ones(frames, dtype=np.float32)
        left[flight] = 0.01
        right[flight] = 0.01

        world, root, correction, diagnostics = _stabilize_world_ground(
            joints,
            transl,
            (left, right),
            fps=30.0,
            enabled=True,
        )

        support_y = np.minimum(world[:, 10, 1], world[:, 11, 1])
        contacts = np.ones(frames, dtype=bool)
        contacts[flight] = False
        self.assertTrue(diagnostics["applied"])
        self.assertLess(float(np.percentile(np.abs(support_y[contacts]), 95)), 0.01)
        self.assertGreater(float(np.min(support_y[flight])), 0.12)
        self.assertGreaterEqual(float(support_y.min()), -1e-6)
        self.assertLessEqual(float(np.max(np.abs(np.diff(correction))) * 30.0), 0.200001)
        np.testing.assert_allclose(root[:, 1], 1.0 + drift - correction, atol=1e-6)

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
