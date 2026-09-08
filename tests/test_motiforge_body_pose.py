"""CPU body22 pose evidence tests without licensed model assets or video inference."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix

import hmr4d.backends.body_pose as pose_module
import hmr4d.backends.portable as portable_module
from hmr4d.backends.body_pose import BODY22_NAMES, BODY22_PARENTS, export_body_pose
from hmr4d.backends.motiforge import main


class FakeBodyModel:
    def __init__(self, invalid=None):
        self.bm = SimpleNamespace(parents=torch.tensor(BODY22_PARENTS))
        self.invalid = invalid
        self.calls = []
        self.neutral = torch.zeros((22, 3))
        self.neutral[0] = torch.tensor([0.02, 0.9, -0.01])
        for joint, parent in enumerate(BODY22_PARENTS[1:], start=1):
            self.neutral[joint] = self.neutral[parent] + torch.tensor(
                [0.05 * (-1) ** joint, 0.035 + (joint % 3) * 0.008, 0.02 * (joint % 4)]
            )

    def eval(self):
        return self

    def cpu(self):
        return self

    def get_skeleton(self, betas):
        skeleton = self.neutral[None] * (1 + 0.1 * betas[:, :1, None])
        if self.invalid == "nonfinite_bind":
            skeleton[:, 1, 0] = float("nan")
        return skeleton

    def __call__(self, **params):
        assert not torch.is_grad_enabled()
        assert all(value.device.type == "cpu" for value in params.values())
        self.calls.append({key: value.clone() for key, value in params.items()})
        neutral = self.neutral[None] * (1 + 0.1 * params["betas"][:, :1, None])
        pose = torch.cat((params["global_orient"], params["body_pose"]), dim=-1).reshape(-1, 22, 3)
        local = axis_angle_to_matrix(pose)
        rotations = [local[:, 0]]
        joints = [neutral[:, 0] + params["transl"]]
        for joint, parent in enumerate(BODY22_PARENTS[1:], start=1):
            rotations.append(rotations[parent] @ local[:, joint])
            offset = neutral[:, joint] - neutral[:, parent]
            joints.append(joints[parent] + (rotations[parent] @ offset[:, :, None])[:, :, 0])
        joints = torch.stack(joints, dim=1)
        if self.invalid == "fk_mismatch":
            joints[:, 7, 1] += 0.002
        elif self.invalid == "nonfinite_joints":
            joints[:, 1, 2] = float("inf")
        elif self.invalid == "bad_joint_shape":
            joints = joints[:, :21]
        return SimpleNamespace(joints=joints)


class BodyPoseExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "input.npz"
        self.output = self.root / "body-pose.npz"
        self.model_path = self.root / portable_module.MODEL_RELATIVE_PATH
        self.model_path.parent.mkdir(parents=True)
        self.model_path.write_bytes(b"fake-body-pose-model")
        self.model = FakeBodyModel()

    def arrays(self, frames=5):
        time = np.arange(frames, dtype=np.float32) / 30
        params = {
            "global_orient": np.stack((time * 0.1, 0.4 + time * 0.2, time * 0), axis=-1),
            "transl": np.stack((time * 0.2, time * 0.1 + 0.1, time * -0.1), axis=-1),
            "body_pose": np.broadcast_to(np.linspace(-0.4, 0.4, 63, dtype=np.float32), (frames, 63)).copy(),
            "betas": np.full((frames, 10), 0.3, dtype=np.float32),
        }
        with torch.no_grad():
            joints = FakeBodyModel()(**{key: torch.from_numpy(value) for key, value in params.items()}).joints.numpy()
        metadata = {
            "protocol": 3, "fps": 30.0, "normalized_num_frames": frames,
            "source_path": "/original.mp4", "source_sha256": "a" * 64,
            "gvhmr_backend_revision": "original-backend", "gvhmr_revision": "upstream-revision",
            "ground_stabilization": {"version": "static-camera-contact-floor-v2", "applied": True},
            "foot_surface": {"schema": "smplx-foot-surface-v1", "custom": "preserve"},
        }
        return {
            **{f"smpl_params_global.{key}": value for key, value in params.items()},
            "pred_w_j3d": joints,
            "smpl_params_incam.transl": params["transl"] + 0.4,
            "floor_correction_y": np.linspace(0.01, 0.04, frames, dtype=np.float32),
            "left_contact_confidence": np.linspace(0, 1, frames, dtype=np.float32),
            "right_contact_confidence": np.linspace(1, 0, frames, dtype=np.float32),
            "foot_surface_y": np.full((frames, 2), 0.02, dtype=np.float32),
            "custom_ints": np.arange(frames, dtype=np.int16),
            "motiforge_video_json": np.asarray(json.dumps(metadata)),
        }

    def write(self, arrays):
        np.savez_compressed(self.input, **arrays)

    def export(self):
        return export_body_pose(
            self.input, self.output, self.root, backend_id="pose-backend",
            model_factory=lambda path: self.model,
        )

    def load(self):
        with np.load(self.output, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def test_export_preserves_all_input_arrays_and_original_source_metadata(self):
        arrays = self.arrays()
        self.write(arrays)
        original_bytes = self.input.read_bytes()
        report = self.export()
        output = self.load()
        self.assertEqual(self.input.read_bytes(), original_bytes)
        self.assertEqual(set(output) - set(arrays), set(pose_module.POSE_ARRAYS))
        for key, value in arrays.items():
            if key != "motiforge_video_json":
                np.testing.assert_array_equal(output[key], value, err_msg=key)
                self.assertEqual(output[key].dtype, value.dtype)
        before = json.loads(arrays["motiforge_video_json"].item())
        metadata = json.loads(output["motiforge_video_json"].item())
        for key, value in before.items():
            if key != "gvhmr_backend_revision":
                self.assertEqual(metadata[key], value)
        self.assertEqual(report["frames"], len(arrays["pred_w_j3d"]))
        for array in output.values():
            self.assertFalse(array.dtype.hasobject)

    def test_exact_pose_schema_wxyz_shaped_bind_and_world_bone_reconstruction(self):
        self.write(self.arrays())
        self.export()
        output = self.load()
        pose = json.loads(output["motiforge_video_json"].item())["body_pose"]
        self.assertEqual(pose["schema"], "smplx-body22-pose-v1")
        self.assertEqual((pose["up_axis"], pose["units"], pose["quaternion_order"]), ("y", "m", "wxyz"))
        self.assertEqual(pose["joint_names"], list(BODY22_NAMES))
        self.assertEqual(pose["parents"], list(BODY22_PARENTS))
        q = output["body22_world_rotations"]
        self.assertEqual(q.shape, (5, 22, 4))
        self.assertEqual(q.dtype, np.dtype("float32"))
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-7)
        expected_bind = self.model.neutral.numpy() * np.float32(1.03)
        np.testing.assert_array_equal(output["body22_bind_positions"], expected_bind)
        np.testing.assert_array_equal(output["body22_bind_rotations"], np.tile([1, 0, 0, 0], (22, 1)))
        matrices = quaternion_to_matrix(torch.from_numpy(q).double()).numpy()
        world = output["pred_w_j3d"]
        bind = output["body22_bind_positions"]
        for joint, parent in enumerate(BODY22_PARENTS[1:], start=1):
            np.testing.assert_allclose(
                matrices[:, parent] @ (bind[joint] - bind[parent]),
                world[:, joint] - world[:, parent], atol=3e-7, rtol=0,
            )
        root = axis_angle_to_matrix(torch.from_numpy(output["smpl_params_global.global_orient"])).numpy()
        np.testing.assert_allclose(matrices[:, 0], root, atol=1e-7)
        self.assertLessEqual(pose["body22_max_error_m"], 1e-6)
        self.assertLessEqual(pose["bind_reconstruction_max_error_m"], 1e-6)

    def test_records_actual_exporter_shared_helper_model_and_parent_artifact_identity(self):
        self.write(self.arrays())
        digest = hashlib.sha256(self.input.read_bytes()).hexdigest()
        self.export()
        metadata = json.loads(self.load()["motiforge_video_json"].item())
        pose = metadata["body_pose"]
        self.assertEqual(pose["exporter_sha256"], hashlib.sha256(Path(pose_module.__file__).read_bytes()).hexdigest())
        self.assertEqual(pose["body_model_sha256"], hashlib.sha256(self.model_path.read_bytes()).hexdigest())
        self.assertEqual(pose["helper_sha256"], portable_module._helper_sha256())
        identity = metadata["body_pose_export"]
        self.assertEqual(identity["source_artifact_sha256"], digest)
        self.assertEqual(identity["source_artifact"], str(self.input.resolve()))
        self.assertEqual(identity["source_backend_revision"], "original-backend")
        self.assertEqual(identity["backend_revision"], "pose-backend")
        self.assertEqual(metadata["gvhmr_backend_revision"], "pose-backend")

    def test_cpu_no_grad_chunk_bounds_include_neutral_model_verification(self):
        arrays = self.arrays(frames=130)
        self.write(arrays)
        self.export()
        self.assertEqual([len(call["transl"]) for call in self.model.calls], [1, 64, 64, 2])
        for field in ("body_pose", "betas", "global_orient", "transl"):
            actual = torch.cat([call[field] for call in self.model.calls[1:]]).numpy()
            np.testing.assert_array_equal(actual, arrays[f"smpl_params_global.{field}"])

    def test_dynamic_shape_even_tiny_variation_is_explicitly_rejected(self):
        arrays = self.arrays()
        arrays["smpl_params_global.betas"][1, 0] = np.nextafter(np.float32(0.3), np.float32(1.0))
        self.write(arrays)
        with self.assertRaisesRegex(ValueError, "constant betas.*dynamic shape"):
            self.export()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.model.calls, [])

    def test_invalid_parent_model_fk_bind_or_output_is_rejected(self):
        self.write(self.arrays())
        for invalid in ("fk_mismatch", "nonfinite_joints", "bad_joint_shape", "nonfinite_bind", "parent"):
            with self.subTest(invalid=invalid):
                self.model = FakeBodyModel(invalid=invalid)
                if invalid == "parent":
                    self.model.bm.parents[4] = 0
                with self.assertRaisesRegex(ValueError, "body22|parent"):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_incompatible_serialized_rotation_is_rejected(self):
        self.write(self.arrays())
        def wrong_quaternions(matrices):
            identity = torch.zeros((*matrices.shape[:-2], 4), dtype=matrices.dtype)
            identity[..., 0] = 1
            return identity
        with (
            patch("pytorch3d.transforms.matrix_to_quaternion", side_effect=wrong_quaternions),
            self.assertRaisesRegex(ValueError, "Serialized rotation/bind"),
        ):
            self.export()
        self.assertFalse(self.output.exists())

    def test_bad_input_shape_nonfinite_metadata_and_pickle_are_rejected(self):
        for case in ("shape", "nonfinite", "metadata", "pickle"):
            with self.subTest(case=case):
                arrays = self.arrays()
                if case == "shape":
                    arrays["smpl_params_global.body_pose"] = np.zeros((5, 62))
                elif case == "nonfinite":
                    arrays["smpl_params_global.global_orient"][0, 0] = np.nan
                elif case == "metadata":
                    arrays["motiforge_video_json"] = np.asarray("[]")
                else:
                    arrays["unsafe_extra"] = np.asarray([{}], dtype=object)
                self.write(arrays)
                with self.assertRaises(ValueError):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_refuses_existing_evidence_missing_asset_same_path_and_existing_output(self):
        arrays = self.arrays()
        arrays["body22_bind_positions"] = np.zeros((22, 3))
        self.write(arrays)
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.export()
        self.write(self.arrays())
        original = self.input.read_bytes()
        with self.assertRaises(ValueError):
            export_body_pose(self.input, self.input, self.root, backend_id="id")
        self.assertEqual(original, self.input.read_bytes())
        with self.assertRaisesRegex(FileNotFoundError, "SMPL-X body model"):
            export_body_pose(self.input, self.output, self.root / "absent", backend_id="id")
        self.output.write_bytes(b"sentinel")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(self.output.read_bytes(), b"sentinel")

    def test_atomic_publish_race_never_replaces_winner_and_cleans_temporary(self):
        self.write(self.arrays())
        def race(source, output):
            output.write_bytes(b"race-winner")
            raise FileExistsError(output)
        with (
            patch.object(portable_module.os, "link", side_effect=race),
            self.assertRaises(FileExistsError),
        ):
            self.export()
        self.assertEqual(self.output.read_bytes(), b"race-winner")
        self.assertEqual(list(self.root.glob(".body-pose.npz.*.tmp")), [])

    def test_explicit_cli_dispatches_only_requested_export_and_reports_failure(self):
        argv = ["export-body-pose", str(self.input), "--output", str(self.output), "--asset-root", str(self.root)]
        with patch.object(pose_module, "export_body_pose", return_value={"ok": True}) as export:
            self.assertEqual(main(argv), 0)
            self.assertEqual(export.call_args.args, (self.input, self.output, self.root))
        with patch.object(pose_module, "export_body_pose", side_effect=ValueError("fixture-error")):
            self.assertEqual(main(argv), 1)


if __name__ == "__main__":
    unittest.main()
