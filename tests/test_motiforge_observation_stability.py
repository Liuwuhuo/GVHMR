"""CPU-only regression evidence for raw observation audits and bounded repair."""

from __future__ import annotations

import json
import unittest

import numpy as np

from hmr4d.backends.observation_stability import (
    POLICY,
    audit_observations,
    audit_world_motion,
    detect_and_repair,
)


def observations(frames=20, dtype=np.float32):
    keypoints = np.zeros((frames, 17, 3), dtype=dtype)
    keypoints[..., 0] = 50
    keypoints[..., 1] = 80
    keypoints[..., 2] = 0.95
    keypoints[:, 7:11:2, 0] = 20
    keypoints[:, 8:11:2, 0] = 80
    return keypoints, np.full(frames, 100, dtype=dtype)


class ConservativeRepairTests(unittest.TestCase):
    def test_all_four_joint_rules_preserve_confidence_dtype_and_other_coordinates(self):
        for joint, opposite in ((7, 8), (8, 7), (9, 10), (10, 9)):
            for dtype in (np.float32, np.float64):
                with self.subTest(joint=joint, dtype=dtype):
                    keypoints, scale = observations(dtype=dtype)
                    keypoints[5:8, joint, :2] = keypoints[5:8, opposite, :2]
                    keypoints[5:8, joint, 2] = 0.3
                    original = keypoints.copy()
                    repaired, accepted, rejected = detect_and_repair(keypoints, scale, 30)
                    self.assertEqual(len(accepted), 1)
                    self.assertEqual(rejected, [])
                    self.assertEqual(accepted[0]["start_frame"], 5)
                    self.assertEqual(accepted[0]["stop_frame_exclusive"], 8)
                    self.assertEqual(accepted[0]["duration_seconds"], 0.1)
                    expected = original.copy()
                    expected[5:8, joint, :2] = original[4, joint, :2]
                    np.testing.assert_array_equal(repaired, expected)
                    np.testing.assert_array_equal(keypoints, original)
                    self.assertEqual(repaired[..., 2].tobytes(), keypoints[..., 2].tobytes())
                    self.assertEqual(repaired.dtype, dtype)
                    self.assertFalse(np.shares_memory(repaired, keypoints))
                    json.dumps([accepted, rejected], allow_nan=False)

    def test_exact_linear_bridge_uses_immediate_endpoints(self):
        keypoints, scale = observations()
        keypoints[4, 9, :2] = [20, 40]
        keypoints[8, 9, :2] = [40, 60]
        keypoints[5:8, 9, :2] = keypoints[5:8, 10, :2]
        keypoints[5:8, 9, 2] = 0.3
        repaired, accepted, _ = detect_and_repair(keypoints, scale, 30)
        np.testing.assert_array_equal(repaired[5:8, 9, :2], [[25, 45], [30, 50], [35, 55]])
        self.assertEqual(accepted[0]["candidate_xy"], [[25, 45], [30, 50], [35, 55]])

    def test_short_bound_is_seconds_not_a_fixed_frame_count(self):
        for fps, gap, accepted_count in ((30, 5, 1), (30, 6, 0), (60, 10, 1), (60, 11, 0), (12, 2, 1)):
            with self.subTest(fps=fps, gap=gap):
                keypoints, scale = observations(30)
                keypoints[5:5 + gap, 9, :2] = keypoints[5:5 + gap, 10, :2]
                keypoints[5:5 + gap, 9, 2] = 0.3
                repaired, accepted, rejected = detect_and_repair(keypoints, scale, fps)
                self.assertEqual(len(accepted), accepted_count)
                if not accepted_count:
                    self.assertIn("gap_too_long", rejected[0]["rejection_reasons"])
                    np.testing.assert_array_equal(repaired, keypoints)

    def test_endpoint_occlusion_is_never_repaired(self):
        for interval in (slice(0, 2), slice(18, 20), slice(None)):
            keypoints, scale = observations()
            keypoints[interval, 9, :2] = keypoints[interval, 10, :2]
            keypoints[interval, 9, 2] = 0.3
            repaired, accepted, rejected = detect_and_repair(keypoints, scale, 30)
            self.assertEqual(accepted, [])
            self.assertIn("missing_two_sided_anchors", rejected[0]["rejection_reasons"])
            np.testing.assert_array_equal(repaired, keypoints)

    def test_unreliable_symmetric_not_collapsed_or_continuous_observations_are_not_repaired(self):
        for reason in (
            "opposite_not_reliable", "no_confidence_asymmetry", "not_a_cross_side_collapse",
            "insufficient_temporal_discontinuity",
        ):
            with self.subTest(reason=reason):
                keypoints, scale = observations()
                keypoints[5:7, 9, :2] = keypoints[5:7, 10, :2]
                keypoints[5:7, 9, 2] = 0.3
                if reason == "opposite_not_reliable":
                    keypoints[5:7, 10, 2] = 0.7
                elif reason == "no_confidence_asymmetry":
                    keypoints[5:7, 9, 2] = 0.69
                    keypoints[5:7, 10, 2] = 0.8
                elif reason == "not_a_cross_side_collapse":
                    keypoints[5:7, 9, 0] = 100
                else:
                    keypoints[:, 9, 0] = 80
                repaired, accepted, rejected = detect_and_repair(keypoints, scale, 30)
                self.assertEqual(accepted, [])
                self.assertIn(reason, rejected[0]["rejection_reasons"])
                np.testing.assert_array_equal(repaired, keypoints)

    def test_native_scores_above_one_are_not_clipped_or_rejected(self):
        keypoints, scale = observations()
        keypoints[..., 2] = 1.02
        repaired, accepted, rejected = detect_and_repair(keypoints, scale, 30)
        np.testing.assert_array_equal(repaired, keypoints)
        self.assertEqual((accepted, rejected), ([], []))
        self.assertEqual(POLICY["maximum_gap_seconds"], 1 / 6)


class ObservationAuditTests(unittest.TestCase):
    def test_healthy_observations_produce_no_warning_and_json_report(self):
        keypoints, scale = observations()
        keypoints[..., 2] = 1.02
        boxes = np.tile([10, 10, 110, 150], (len(keypoints), 1))
        original = keypoints.copy()
        report = audit_observations(keypoints, scale, np.float32(30), image_size=(120, 160), boxes=boxes)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["scope"], "single_selected_track")
        self.assertEqual(report["frame_count"], 20)
        self.assertEqual(report["fps"], 30.0)
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["events"], [])
        self.assertGreater(report["summary"]["native_score_max"], 1)
        np.testing.assert_array_equal(keypoints, original)
        json.dumps(report, allow_nan=False)

    def test_missing_start_and_low_visibility_tail_are_compact_inclusive_intervals(self):
        keypoints, scale = observations(1000)
        keypoints[:50, 5:, 2] = 0
        keypoints[900:, 5:10, 2] = 0.5
        report = audit_observations(keypoints, scale, 25)
        boundary = [event for event in report["events"] if event["code"] == "low_body_visibility_at_clip_boundary"]
        self.assertEqual([(event["start_frame"], event["end_frame"]) for event in boundary], [(0, 49), (900, 999)])
        self.assertEqual(boundary[0]["start_seconds"], 0)
        self.assertEqual(boundary[0]["end_seconds"], 49 / 25)
        self.assertEqual(boundary[0]["duration_seconds"], 50 / 25)
        self.assertTrue(all(event["reasons"] for event in report["events"]))
        self.assertEqual(report["summary"]["missing_body_frame_count"], 50)
        self.assertEqual(report["summary"]["low_body_visibility_frame_count"], 150)
        self.assertLess(len(report["events"]), 20)
        self.assertLess(len(json.dumps(report, allow_nan=False)), 9000)

    def test_interior_visibility_loss_is_not_a_clip_boundary_warning(self):
        keypoints, scale = observations()
        keypoints[5:8, 5:, 2] = 0
        report = audit_observations(keypoints, scale, 30)
        self.assertIn("missing_body_observations", report["warnings"])
        self.assertNotIn("low_body_visibility_at_clip_boundary", report["warnings"])

    def test_image_edge_evidence_is_optional_and_not_proof_of_missing_track(self):
        keypoints, scale = observations()
        boxes = np.tile([5, 5, 115, 155], (len(keypoints), 1)).astype(float)
        boxes[4:6, 0] = -3
        keypoints[8, 15, 0] = 130
        report = audit_observations(keypoints, scale, 30, image_size=(120, 160), boxes=boxes)
        self.assertIn("selected_box_at_image_boundary", report["warnings"])
        self.assertIn("visible_body_keypoint_outside_image", report["warnings"])
        self.assertNotIn("missing_body_observations", report["warnings"])
        self.assertEqual(audit_observations(keypoints, scale, 30)["warnings"], [])

    def test_single_frame_missing_track_has_valid_zero_time_interval(self):
        keypoints, scale = observations(1)
        keypoints[..., 2] = 0
        report = audit_observations(keypoints, scale, 30)
        self.assertTrue(report["warnings"])
        self.assertTrue(all(event["start_frame"] == event["end_frame"] == 0 for event in report["events"]))
        self.assertTrue(all(event["start_seconds"] == event["end_seconds"] == 0 for event in report["events"]))

    def test_observation_contract_errors_raise(self):
        keypoints, scale = observations()
        invalid = (
            (keypoints[:0], scale[:0], 30), (keypoints[:, :16], scale, 30),
            (keypoints.astype(int), scale, 30), (keypoints * np.nan, scale, 30),
            (keypoints, scale[:-1], 30), (keypoints, scale * 0, 30),
            (keypoints, scale * np.inf, 30), (keypoints, scale, 0),
            (keypoints, scale, np.nan), (keypoints, scale, [30]), (keypoints, scale, "30"),
        )
        for arguments in invalid:
            for function in (audit_observations, detect_and_repair):
                with (
                    self.subTest(shape=np.shape(arguments[0]), fps=arguments[2], function=function.__name__),
                    self.assertRaises(ValueError),
                ):
                    function(*arguments)
        for extras in (
            {"image_size": (0, 100)}, {"image_size": (100, np.inf)}, {"image_size": [100]},
            {"boxes": np.zeros((20, 3))}, {"boxes": np.zeros((20, 4))},
            {"boxes": np.full((20, 4), np.nan)},
        ):
            with self.subTest(extras=extras), self.assertRaises(ValueError):
                audit_observations(keypoints, scale, 30, **extras)


class WorldMotionAuditTests(unittest.TestCase):
    def test_high_speed_translation_is_not_a_failure_or_warning(self):
        joints = np.zeros((50, 22, 3), dtype=np.float32)
        joints[..., 0] = np.arange(50)[:, None] * 2
        original = joints.copy()
        report = audit_world_motion(joints, 30)
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["summary"]["maximum_root_speed_mps"], 60)
        self.assertFalse(report["summary"]["motion_modified"])
        np.testing.assert_array_equal(joints, original)
        json.dumps(report, allow_nan=False)

    def test_local_one_frame_reversal_spike_is_evidence_with_frame_time(self):
        joints = np.zeros((40, 22, 3), dtype=np.float32)
        joints[12, 20, 0] = 0.5
        original = joints.copy()
        report = audit_world_motion(joints, 30)
        self.assertEqual(report["warnings"], ["temporal_joint_reversal_spike"])
        self.assertEqual(len(report["events"]), 1)
        event = report["events"][0]
        self.assertEqual((event["start_frame"], event["end_frame"], event["joint_index"]), (12, 12, 20))
        self.assertEqual(event["start_seconds"], 12 / 30)
        self.assertTrue(event["reasons"])
        np.testing.assert_array_equal(joints, original)

    def test_root_spike_is_labelled_and_never_corrected(self):
        joints = np.zeros((40, 22, 3))
        joints[12, 0, 1] = 0.5
        report = audit_world_motion(joints, 30)
        self.assertEqual(report["warnings"], ["temporal_root_reversal_spike"])
        self.assertEqual(joints[12, 0, 1], 0.5)

    def test_fast_smooth_and_repeated_fast_motion_are_not_raw_velocity_rejections(self):
        for frequency in (2, 5):
            joints = np.zeros((300, 22, 3))
            time = np.arange(300) / 30
            joints[:, 20, 0] = np.cos(2 * np.pi * frequency * time)
            original = joints.copy()
            report = audit_world_motion(joints, 30)
            self.assertGreater(report["summary"]["maximum_joint_speed_mps"], 3)
            self.assertEqual(report["warnings"], [])
            np.testing.assert_array_equal(joints, original)

    def test_tiny_high_frequency_jitter_and_slow_reversal_are_not_reported(self):
        for amplitude, fps in ((0.01, 1000), (0.5, 2)):
            joints = np.zeros((40, 22, 3))
            joints[12, 20, 0] = amplitude
            self.assertEqual(audit_world_motion(joints, fps)["warnings"], [])

    def test_one_and_two_frame_clips_have_no_invented_temporal_evidence(self):
        for frames in (1, 2):
            report = audit_world_motion(np.zeros((frames, 22, 3)), 30)
            self.assertEqual(report["events"], [])
            self.assertFalse(report["summary"]["temporal_window_available"])
            json.dumps(report, allow_nan=False)

    def test_world_contract_errors_raise(self):
        for joints, fps in (
            (np.zeros((0, 22, 3)), 30), (np.zeros((5, 21, 3)), 30),
            (np.full((5, 22, 3), np.nan), 30), (np.full((5, 22, 3), np.inf), 30),
            (np.zeros((5, 22, 3)), -1), (np.zeros((5, 22, 3)), True),
        ):
            with self.subTest(shape=joints.shape, fps=fps), self.assertRaises(ValueError):
                audit_world_motion(joints, fps)


if __name__ == "__main__":
    unittest.main()
