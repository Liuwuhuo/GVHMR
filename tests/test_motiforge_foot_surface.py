"""CPU-only foot-surface export tests; no licensed body model is required."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import hmr4d.backends.foot_surface as foot_surface_module
from hmr4d.backends.foot_surface import export_foot_surface

_MODEL_RELATIVE_PATH = "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
_BACKEND_ID = "surface-aware-backend"
_BASE_VERTEX_Y = np.asarray([-0.05, -0.20, -0.10, -0.15, -0.30, -0.20, -9.0, -8.0])


def _joint_offsets() -> np.ndarray:
    # Extra output joints must not participate in the body22 consistency check.
    offsets = np.zeros((55, 3), dtype=np.float32)
    offsets[:, 0] = np.arange(55, dtype=np.float32) * 0.01
    offsets[:, 1] = np.arange(55, dtype=np.float32) * 0.005
    return offsets


class _FakeBodyModel:
    def __init__(self, *, invalid_output: str | None = None) -> None:
        weights = torch.zeros((8, 55), dtype=torch.float32)
        weights[0, 7] = 1.0
        weights[1, 7], weights[1, 10] = 0.25, 0.25  # Included at exactly 0.5.
        weights[2, 10] = 0.75
        weights[3, 8] = 1.0
        weights[4, 8], weights[4, 11] = 0.25, 0.25  # Included at exactly 0.5.
        weights[5, 11] = 0.75
        weights[6, 7], weights[6, 10] = 0.25, 0.24  # Lower but excluded.
        weights[7, 8], weights[7, 11] = 0.25, 0.24  # Lower but excluded.
        weights[:, 0] = 1.0 - weights.sum(dim=1)
        self.bm = SimpleNamespace(lbs_weights=weights)
        self.calls: list[dict[str, torch.Tensor]] = []
        self.invalid_output = invalid_output

    def eval(self):
        return self

    def cpu(self):
        return self

    def __call__(self, **params):
        assert not torch.is_grad_enabled(), "Surface FK must run under torch.no_grad()."
        for value in params.values():
            assert isinstance(value, torch.Tensor)
            assert value.device.type == "cpu", "Surface FK must not require CUDA."
        self.calls.append({key: value.detach().clone() for key, value in params.items()})
        transl = params["transl"]
        vertices = transl[:, None, :].repeat(1, 8, 1)
        vertices[:, :, 1] += torch.as_tensor(_BASE_VERTEX_Y, dtype=transl.dtype)
        joints = transl[:, None, :] + torch.as_tensor(_joint_offsets(), dtype=transl.dtype)
        if self.invalid_output == "fk_mismatch":
            joints[:, 7, 1] += 0.02
        elif self.invalid_output == "nonfinite_vertices":
            vertices[:, 1, 1] = float("nan")
        elif self.invalid_output == "nonfinite_joints":
            joints[:, 1, 1] = float("inf")
        elif self.invalid_output == "bad_vertices_shape":
            vertices = vertices[:, :, :2]
        elif self.invalid_output == "bad_joints_shape":
            joints = joints[:, :21, :]
        return SimpleNamespace(vertices=vertices, joints=joints)


class FootSurfaceExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input_path = self.root / "prediction.npz"
        self.output_path = self.root / "prediction-with-surface.npz"
        self.asset_root = self.root / "assets"
        self.model_path = self.asset_root / _MODEL_RELATIVE_PATH
        self.model_path.parent.mkdir(parents=True)
        # The injected factory makes a real SMPL-X model unnecessary.
        self.model_path.write_bytes(b"fake-neutral-smplx-model")
        self.factory_paths: list[Path] = []
        self.model = _FakeBodyModel()

    def _arrays(self, frames: int = 5) -> dict[str, np.ndarray]:
        correction = np.linspace(0.04, 0.12, frames, dtype=np.float32)
        transl = np.zeros((frames, 3), dtype=np.float32)
        transl[:, 0] = np.arange(frames, dtype=np.float32) * 0.02
        # These portable parameters already contain the existing floor correction.
        transl[:, 1] = 0.8 + np.arange(frames, dtype=np.float32) * 0.0005 - correction
        metadata = {
            "protocol": 3,
            "fps": 30.0,
            "normalized_num_frames": frames,
            "source_path": "/source/fixture.mp4",
            "source_sha256": "1" * 64,
            "gvhmr_revision": "existing-gvhmr-revision",
            "gvhmr_backend_revision": "existing-backend-revision",
            "ground_stabilization": {"version": "contact-floor-v1", "applied": True},
            "custom_metadata": {"preserve": [1, 2, 3]},
        }
        return {
            "pred_w_j3d": transl[:, None, :] + _joint_offsets()[None, :22, :],
            "smpl_params_global.body_pose": np.zeros((frames, 63), dtype=np.float32),
            "smpl_params_global.betas": np.zeros((frames, 10), dtype=np.float32),
            "smpl_params_global.global_orient": np.zeros((frames, 3), dtype=np.float32),
            "smpl_params_global.transl": transl,
            "smpl_params_incam.transl": transl + np.float32(0.3),
            "floor_correction_y": correction,
            "left_contact_confidence": np.linspace(0.1, 0.9, frames, dtype=np.float32),
            "right_contact_confidence": np.linspace(0.9, 0.1, frames, dtype=np.float32),
            "custom_integer_array": np.arange(frames, dtype=np.int16),
            "motiforge_video_json": np.asarray(json.dumps(metadata)),
        }

    def _write(self, arrays: dict[str, np.ndarray]) -> None:
        np.savez_compressed(self.input_path, **arrays)

    def _factory(self, model_path):
        self.factory_paths.append(Path(model_path))
        return self.model

    def _export(self):
        return export_foot_surface(
            self.input_path,
            self.output_path,
            self.asset_root,
            backend_id=_BACKEND_ID,
            model_factory=self._factory,
        )

    def _load_output(self) -> dict[str, np.ndarray]:
        with np.load(self.output_path, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    def test_export_selects_threshold_vertices_and_per_side_min_y(self) -> None:
        arrays = self._arrays()
        self._write(arrays)
        report = self._export()
        output = self._load_output()
        surface = output["foot_surface_y"]
        self.assertIsInstance(report, dict)
        self.assertEqual(surface.shape, (5, 2))
        self.assertEqual(surface.dtype, np.dtype("float32"))
        expected = arrays["smpl_params_global.transl"][:, 1, None] + np.asarray([-0.20, -0.30], dtype=np.float32)
        np.testing.assert_allclose(surface, expected, atol=1e-7, rtol=0.0)
        self.assertEqual(self.factory_paths, [self.model_path])

    def test_metadata_records_geometry_backend_and_source_artifact_identity(self) -> None:
        arrays = self._arrays()
        self._write(arrays)
        input_sha = hashlib.sha256(self.input_path.read_bytes()).hexdigest()
        self._export()
        metadata = json.loads(str(self._load_output()["motiforge_video_json"].item()))
        surface = metadata["foot_surface"]
        self.assertEqual(surface["schema"], "smplx-foot-surface-v1")
        self.assertEqual(surface["foot_order"], ["left", "right"])
        self.assertEqual(surface["up_axis"], "y")
        self.assertEqual(surface["units"], "m")
        self.assertEqual(surface["vertex_counts"], [3, 3])
        self.assertEqual(surface["body_model_sha256"], hashlib.sha256(self.model_path.read_bytes()).hexdigest())
        self.assertEqual(
            surface["exporter_sha256"],
            hashlib.sha256(Path(foot_surface_module.__file__).read_bytes()).hexdigest(),
        )
        self.assertEqual(metadata["gvhmr_backend_revision"], _BACKEND_ID)
        identity = metadata["foot_surface_export"]
        self.assertEqual(identity["source_artifact"], str(self.input_path.resolve()))
        self.assertEqual(identity["source_artifact_sha256"], input_sha)
        self.assertEqual(identity["source_backend_revision"], "existing-backend-revision")
        self.assertEqual(identity["backend_revision"], _BACKEND_ID)
        self.assertLessEqual(identity["body22_max_error_m"], 1e-6)
        original_metadata = json.loads(str(arrays["motiforge_video_json"].item()))
        for key, value in original_metadata.items():
            if key != "gvhmr_backend_revision":
                self.assertEqual(metadata[key], value, key)

    def test_export_preserves_input_bytes_arrays_and_existing_floor_correction(self) -> None:
        arrays = self._arrays()
        self._write(arrays)
        original_bytes = self.input_path.read_bytes()
        self._export()
        self.assertEqual(self.input_path.read_bytes(), original_bytes)
        output = self._load_output()
        self.assertEqual(set(output), set(arrays) | {"foot_surface_y"})
        for key, expected in arrays.items():
            if key != "motiforge_video_json":
                np.testing.assert_array_equal(output[key], expected, err_msg=key)
                self.assertEqual(output[key].dtype, expected.dtype, key)
        np.testing.assert_array_equal(self.model.calls[0]["transl"].numpy(), arrays["smpl_params_global.transl"])
        # In particular, the already-applied correction is not subtracted again.
        np.testing.assert_allclose(
            output["foot_surface_y"][:, 0],
            arrays["smpl_params_global.transl"][:, 1] - np.float32(0.20),
            atol=1e-7,
            rtol=0.0,
        )
        for value in output.values():
            self.assertFalse(value.dtype.hasobject)

    def test_fk_runs_on_cpu_without_gradients_in_bounded_chunks(self) -> None:
        arrays = self._arrays(frames=130)
        self._write(arrays)
        self._export()
        sizes = [len(call["transl"]) for call in self.model.calls]
        self.assertEqual(sum(sizes), 130)
        self.assertGreater(len(sizes), 1)
        self.assertTrue(all(0 < size <= 64 for size in sizes), sizes)
        for field in ("transl", "body_pose", "global_orient", "betas"):
            actual = torch.cat([call[field] for call in self.model.calls]).numpy()
            np.testing.assert_array_equal(actual, arrays[f"smpl_params_global.{field}"])
        self.assertEqual(self._load_output()["foot_surface_y"].shape, (130, 2))

    def test_body22_fk_mismatch_is_an_explicit_error(self) -> None:
        self._write(self._arrays())
        self.model = _FakeBodyModel(invalid_output="fk_mismatch")
        with self.assertRaisesRegex(ValueError, r"(?i)(body.?22|FK|joint|关节)"):
            self._export()
        self.assertFalse(self.output_path.exists())

    def test_invalid_model_output_is_rejected_without_publishing_an_artifact(self) -> None:
        self._write(self._arrays())
        for invalid in (
            "nonfinite_vertices",
            "nonfinite_joints",
            "bad_vertices_shape",
            "bad_joints_shape",
        ):
            with self.subTest(invalid=invalid):
                self.model = _FakeBodyModel(invalid_output=invalid)
                with self.assertRaises(ValueError):
                    self._export()
                self.assertFalse(self.output_path.exists())

    def test_missing_required_array_is_rejected(self) -> None:
        for key in (
            "pred_w_j3d",
            "smpl_params_global.body_pose",
            "smpl_params_global.betas",
            "smpl_params_global.global_orient",
            "smpl_params_global.transl",
            "motiforge_video_json",
        ):
            with self.subTest(key=key):
                arrays = self._arrays()
                del arrays[key]
                self._write(arrays)
                with self.assertRaises(ValueError):
                    self._export()
                self.assertFalse(self.output_path.exists())

    def test_bad_parameter_and_joint_shapes_are_rejected(self) -> None:
        invalid_shapes = {
            "pred_w_j3d": (5, 21, 3),
            "smpl_params_global.body_pose": (5, 62),
            "smpl_params_global.betas": (5, 9),
            "smpl_params_global.global_orient": (4, 3),
            "smpl_params_global.transl": (5, 1, 3),
        }
        for key, shape in invalid_shapes.items():
            with self.subTest(key=key):
                arrays = self._arrays()
                arrays[key] = np.zeros(shape, dtype=np.float32)
                self._write(arrays)
                with self.assertRaises(ValueError):
                    self._export()
                self.assertFalse(self.output_path.exists())

    def test_nonfinite_required_geometry_is_rejected(self) -> None:
        for key in (
            "pred_w_j3d",
            "smpl_params_global.body_pose",
            "smpl_params_global.betas",
            "smpl_params_global.global_orient",
            "smpl_params_global.transl",
        ):
            for value in (np.nan, np.inf):
                with self.subTest(key=key, value=value):
                    arrays = self._arrays()
                    arrays[key].flat[0] = value
                    self._write(arrays)
                    with self.assertRaises(ValueError):
                        self._export()
                    self.assertFalse(self.output_path.exists())

    def test_invalid_portable_metadata_is_rejected(self) -> None:
        for key, value in (
            ("protocol", 2),
            ("normalized_num_frames", 4),
            ("fps", 0.0),
            ("source_sha256", ""),
        ):
            with self.subTest(key=key):
                arrays = self._arrays()
                metadata = json.loads(str(arrays["motiforge_video_json"].item()))
                metadata[key] = value
                arrays["motiforge_video_json"] = np.asarray(json.dumps(metadata))
                self._write(arrays)
                with self.assertRaises(ValueError):
                    self._export()
                self.assertFalse(self.output_path.exists())

    def test_pickle_requiring_array_is_rejected_even_when_unrelated_to_geometry(self) -> None:
        arrays = self._arrays()
        arrays["unsafe_extra"] = np.asarray([{"not": "portable"}], dtype=object)
        self._write(arrays)
        with self.assertRaises(ValueError):
            self._export()
        self.assertFalse(self.output_path.exists())

    def test_missing_body_model_asset_is_an_explicit_error(self) -> None:
        self._write(self._arrays())
        with self.assertRaisesRegex(FileNotFoundError, r"SMPLX_NEUTRAL|SMPL-X|SMPLX|body.model"):
            export_foot_surface(
                self.input_path,
                self.output_path,
                self.root / "missing-assets",
                backend_id=_BACKEND_ID,
                model_factory=self._factory,
            )
        self.assertEqual(self.factory_paths, [])
        self.assertFalse(self.output_path.exists())

    def test_input_equal_to_output_is_refused_without_mutation(self) -> None:
        self._write(self._arrays())
        original_bytes = self.input_path.read_bytes()
        with self.assertRaises(ValueError):
            export_foot_surface(
                self.input_path,
                self.input_path,
                self.asset_root,
                backend_id=_BACKEND_ID,
                model_factory=self._factory,
            )
        self.assertEqual(self.input_path.read_bytes(), original_bytes)

    def test_existing_output_is_refused_without_mutation(self) -> None:
        self._write(self._arrays())
        sentinel = b"existing-output-must-survive"
        self.output_path.write_bytes(sentinel)
        with self.assertRaises(FileExistsError):
            self._export()
        self.assertEqual(self.output_path.read_bytes(), sentinel)


if __name__ == "__main__":
    unittest.main()
