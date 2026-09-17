from __future__ import annotations

import copy
import unittest

import numpy as np

from hmr4d.backends.motiforge import (
    _portable_prediction,
    _stabilize_prediction_ground,
    _stabilize_world_ground,
    _static_camera_height_correction,
)


class StaticCameraHeightTests(unittest.TestCase):
    fps = 30.0

    def setUp(self) -> None:
        self.time = np.arange(300) / self.fps
        self.world = np.zeros((len(self.time), 22, 3), dtype=np.float64)
        self.world[:, :, 0] = np.arange(22)[None, :] * 0.03
        self.world[:, :, 1] = 1.0
        self.world[:, [7, 8, 10, 11], 1] = 0.1
        self.world[:, :, 2] = self.time[:, None] * 0.02
        self.transl = self.world[:, 0].copy()
        angle = 0.35
        self.rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(angle), np.sin(angle)],
             [0.0, -np.sin(angle), np.cos(angle)]]
        )
        self.orient = np.zeros_like(self.transl)
        self.incam = {
            "transl": self.transl @ self.rotation,
            "global_orient": np.tile([angle, 0.0, 0.0], (len(self.time), 1)),
        }
        drift = 0.08 * self.time + 0.07 * np.sin(1.2 * self.time)
        self.world[:, :, 1] += drift[:, None]
        self.transl[:, 1] += drift
        self.contacts = (np.ones(len(self.time)), np.ones(len(self.time)))

    def stabilize(self, world=None, transl=None, incam=None, contacts=None, **kwargs):
        return _stabilize_prediction_ground(
            self.world if world is None else world,
            self.transl if transl is None else transl,
            self.contacts if contacts is None else contacts,
            global_orient=self.orient,
            incam=self.incam if incam is None else incam,
            fps=self.fps,
            enabled=kwargs.get("enabled", True),
            static_camera=kwargs.get("static_camera", True),
        )

    def inject_common_motion(self, signal, *, fixed_feet=False):
        world = self.world.copy()
        root = self.transl.copy()
        incam = copy.deepcopy(self.incam)
        world[:, :, 1] += signal[:, None]
        if fixed_feet:
            world[:, [7, 8, 10, 11], 1] -= signal[:, None]
        root[:, 1] += signal
        # Inject into the raw camera input, not a prefiltered reference.
        incam["transl"] += signal[:, None] * self.rotation[1]
        return world, root, incam

    def test_raw_input_common_jump_and_squat_cancel_before_camera_filter(self):
        baseline, diagnostics = _static_camera_height_correction(
            self.world, self.transl, self.orient, self.incam, fps=self.fps
        )
        self.assertEqual(diagnostics["incam_prefilter"], "none")
        self.assertEqual(diagnostics["discrepancy_gaussian_sigma_seconds"], 0.5)
        for duration in (0.4, 0.6, 0.8):
            jump = np.maximum(0.0, 4.905 * (self.time - 4.0) * (4.0 + duration - self.time))
            points, root, incam = self.inject_common_motion(jump)
            changed, _ = _static_camera_height_correction(
                points, root, self.orient, incam, fps=self.fps
            )
            np.testing.assert_allclose(changed, baseline, atol=1e-14, rtol=0)
        squat = -0.4 * np.exp(-((self.time - 5.0) / 1.5) ** 2)
        points, root, incam = self.inject_common_motion(squat, fixed_feet=True)
        changed, _ = _static_camera_height_correction(
            points, root, self.orient, incam, fps=self.fps
        )
        np.testing.assert_allclose(changed, baseline, atol=1e-14, rtol=0)

    def test_complete_floor_preserves_ballistic_height_and_squat_scale(self):
        for duration in (0.4, 0.6, 0.8):
            with self.subTest(duration=duration):
                active = (self.time >= 4.0) & (self.time <= 4.0 + duration)
                jump = np.maximum(
                    0.0, 4.905 * (self.time - 4.0) * (4.0 + duration - self.time)
                )
                contacts = tuple(np.where(active, 0.0, value) for value in self.contacts)
                baseline = self.stabilize(contacts=contacts)[0]
                points, root, incam = self.inject_common_motion(jump)
                changed = self.stabilize(points, root, incam, contacts)[0]
                recovered = changed[:, 0, 1] - baseline[:, 0, 1]
                # The legacy floor ceiling can respond at takeoff/landing;
                # require preservation within 1 cm, not an exact-floor claim.
                np.testing.assert_allclose(recovered, jump, atol=0.01, rtol=0)
                self.assertGreater(recovered[np.argmax(jump)], 0.95 * jump.max())
        squat = -0.4 * np.exp(-((self.time - 5.0) / 1.5) ** 2)
        baseline = self.stabilize()[0]
        changed = self.stabilize(*self.inject_common_motion(squat, fixed_feet=True))[0]
        np.testing.assert_allclose(changed[:, 0, 1] - baseline[:, 0, 1], squat, atol=5e-7)

    def test_total_rate_pose_xy_nonmutation_and_metadata(self):
        self.world[100:, :, 1] += 0.6
        self.transl[100:, 1] += 0.6
        before = copy.deepcopy((self.world, self.transl, self.orient, self.incam, self.contacts))
        world, root, correction, diagnostics = self.stabilize()
        np.testing.assert_array_equal(world[:, :, [0, 2]], self.world[:, :, [0, 2]].astype("f4"))
        np.testing.assert_array_equal(root[:, [0, 2]], self.transl[:, [0, 2]].astype("f4"))
        np.testing.assert_allclose(
            world - world[:, :1], self.world - self.world[:, :1], atol=3e-7, rtol=0
        )
        expected_world = self.world.astype("f4")
        expected_world[:, :, 1] -= correction[:, None]
        expected_root = self.transl.astype("f4")
        expected_root[:, 1] -= correction
        np.testing.assert_array_equal(world, expected_world)
        np.testing.assert_array_equal(root, expected_root)
        for current, original in zip((self.world, self.transl, self.orient), before[:3]):
            np.testing.assert_array_equal(current, original)
        for key, value in self.incam.items():
            np.testing.assert_array_equal(value, before[3][key])
        for current, original in zip(self.contacts, before[4]):
            np.testing.assert_array_equal(current, original)
        speed = float(np.max(np.abs(np.diff(correction))) * self.fps)
        rounding = np.finfo(np.float32).eps * float(np.abs(correction).max()) * self.fps
        self.assertLessEqual(speed, 0.2 + rounding)  # Serialized float32 rounding.
        self.assertGreater(diagnostics["total_projection_max_change_m"], 0.01)
        self.assertEqual(diagnostics["max_correction_speed_mps"], speed)
        self.assertEqual(diagnostics["total_speed_limit_mps"], 0.2)
        self.assertEqual(diagnostics["version"], "camera-height-no-support-v3")
        self.assertEqual(diagnostics["max_abs_correction_m"], float(np.abs(correction).max()))
        self.assertFalse(diagnostics["support_anchor_enabled"])
        self.assertNotIn("target_foot_height_m", diagnostics)

    def test_below_threshold_floor_still_exports_camera_only_total_and_final_metrics(self):
        old_drift = 0.08 * self.time + 0.07 * np.sin(1.2 * self.time)
        new_drift = 0.01 * self.time
        self.world[:, :, 1] += (new_drift - old_drift)[:, None]
        self.transl[:, 1] += new_drift - old_drift
        camera, _ = _static_camera_height_correction(
            self.world, self.transl, self.orient, self.incam, fps=self.fps
        )
        world, _, correction, diagnostics = self.stabilize()
        self.assertNotIn("floor_stage", diagnostics)
        self.assertTrue(diagnostics["applied"])
        self.assertTrue(diagnostics["camera_stage"]["applied"])
        self.assertNotIn("reason", diagnostics)
        np.testing.assert_array_equal(correction, camera.astype(np.float32))
        self.assertEqual(diagnostics["max_abs_correction_m"], float(np.abs(correction).max()))
        self.assertFalse(diagnostics["support_anchor_enabled"])

    def test_dynamic_and_disabled_do_not_apply_custom_support_even_with_invalid_incam(self):
        for enabled, static in ((True, False), (False, True), (False, False)):
            actual = self.stabilize(incam={}, enabled=enabled, static_camera=static)
            np.testing.assert_array_equal(actual[0], self.world.astype('f4'))
            np.testing.assert_array_equal(actual[1], self.transl.astype('f4'))
            np.testing.assert_array_equal(actual[2], 0)
            self.assertFalse(actual[3]['support_anchor_enabled'])

    def test_camera_path_is_independent_of_static_probabilities(self):
        brief = np.zeros(len(self.time))
        brief[:2] = 1
        expected = self.stabilize()
        for contacts in (None, (brief, brief), (brief * 0, brief * 0)):
            actual = _stabilize_prediction_ground(
                self.world, self.transl, contacts, global_orient=self.orient,
                incam=self.incam, fps=self.fps, enabled=True, static_camera=True,
            )
            for left, right in zip(actual[:3], expected[:3]):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(actual[3], expected[3])

    def test_missing_invalid_camera_does_not_fall_back_to_support(self):
        for incam in (None, {}, {**self.incam, "transl": np.zeros((2, 3))},
                      {**self.incam, "global_orient": self.orient + np.nan}):
            actual = _stabilize_prediction_ground(
                self.world, self.transl, self.contacts, global_orient=self.orient,
                incam=incam, fps=self.fps, enabled=True, static_camera=True,
            )
            np.testing.assert_array_equal(actual[0], self.world.astype('f4'))
            np.testing.assert_array_equal(actual[1], self.transl.astype('f4'))
            np.testing.assert_array_equal(actual[2], 0)
            camera = actual[3]["camera_stage"]
            self.assertFalse(camera["applied"])
            self.assertEqual(camera["reason"], "invalid_or_missing_incam_parameters")
            self.assertTrue(camera["detail"])

    def test_portable_static_branch_exports_total_without_mutating_prediction(self):
        import torch

        world = torch.tensor(self.world, dtype=torch.float32)

        class Endecoder:
            @staticmethod
            def fk_v2(**kwargs):
                return world[None]

        class Model:
            pipeline = type("Pipeline", (), {"endecoder": Endecoder()})()

        params = {
            "body_pose": torch.zeros((len(world), 63)),
            "global_orient": torch.tensor(self.orient, dtype=torch.float32),
            "transl": torch.tensor(self.transl, dtype=torch.float32),
            "betas": torch.zeros((len(world), 10)),
        }
        incam = {key: torch.tensor(value, dtype=torch.float32) for key, value in self.incam.items()}
        original = params["transl"].clone()
        portable = _portable_prediction(
            pred={"smpl_params_global": params, "smpl_params_incam": incam,
                  "K_fullimg": torch.eye(3),
                  "net_outputs": {"static_conf_logits": torch.full((1, len(world), 6), 10.0)}},
            model=Model(), detach_to_cpu=lambda value: value,
            item={"source": "fixture.mp4", "source_sha256": "fixture"},
            options={"static_camera": True, "ground_stabilization": True},
            revision="revision", backend_id="backend", normalized_frames=len(world),
        )
        diagnostics = portable["motiforge_video"]["ground_stabilization"]
        self.assertEqual(diagnostics["version"], "camera-height-no-support-v3")
        expected = original.numpy().copy()
        expected[:, 1] -= portable["floor_correction_y"]
        np.testing.assert_array_equal(portable["smpl_params_global"]["transl"], expected)
        np.testing.assert_array_equal(params["transl"], original)
        for key in ("body_pose", "global_orient", "betas"):
            self.assertIs(portable["smpl_params_global"][key], params[key])
        self.assertIs(portable["smpl_params_incam"], incam)


if __name__ == "__main__":
    unittest.main()
