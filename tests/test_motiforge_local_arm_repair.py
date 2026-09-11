"""CPU-only contract tests for bounded, source-side SO(3) arm repair."""

import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from hmr4d.backends.local_arm_repair import POLICY, localize_arm_pose

ARM_JOINTS = {"left": (16, 18, 20), "right": (17, 19, 21)}


def columns(side):
    return np.array([3 * (joint - 1) + axis for joint in ARM_JOINTS[side] for axis in range(3)])


def repair(start=40, stop=43, joint="right_wrist"):
    return {"joint": joint, "start_frame": start, "stop_frame_exclusive": stop}


def poses(frames=100, dtype=np.float64):
    rng = np.random.default_rng(71)
    original = rng.uniform(-0.3, 0.3, (frames, 63)).astype(dtype)
    candidate = (original + rng.uniform(-0.2, 0.2, original.shape)).astype(dtype)
    return original, candidate


class LocalArmRepairTests(unittest.TestCase):
    def test_each_coco_arm_hit_changes_only_its_three_local_joints(self):
        for joint in ("left_elbow", "left_wrist", "right_elbow", "right_wrist"):
            with self.subTest(joint=joint):
                original, candidate = poses()
                result, metadata = localize_arm_pose(original, candidate, [repair(joint=joint)], 40)
                selected = columns(joint.split("_")[0])
                other = np.setdiff1d(np.arange(63), selected)
                np.testing.assert_array_equal(result[:, other], original[:, other])
                # Ten-frame support at 40 Hz; more distant times are bit-identical.
                np.testing.assert_array_equal(result[:29], original[:29])
                np.testing.assert_array_equal(result[54:], original[54:])
                self.assertGreater(np.max(np.abs(result[40:43, selected] - original[40:43, selected])), 0)
                json.dumps(metadata, allow_nan=False)

    def test_no_inputs_are_mutated_and_output_owns_its_storage(self):
        original, candidate = poses()
        before_original, before_candidate = original.copy(), candidate.copy()
        runs = [repair()]
        before_runs = json.loads(json.dumps(runs))
        result, _ = localize_arm_pose(original, candidate, runs, 40)
        np.testing.assert_array_equal(original, before_original)
        np.testing.assert_array_equal(candidate, before_candidate)
        self.assertEqual(runs, before_runs)
        self.assertFalse(np.shares_memory(result, original))
        self.assertFalse(np.shares_memory(result, candidate))

    def test_empty_repairs_return_exact_independent_copy(self):
        original, candidate = poses()
        result, metadata = localize_arm_pose(original, candidate, [], 40)
        np.testing.assert_array_equal(result, original)
        self.assertFalse(np.shares_memory(result, original))
        json.dumps(metadata, allow_nan=False)

    def test_dtype_is_preserved(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                original, candidate = poses(dtype=dtype)
                result, _ = localize_arm_pose(original, candidate, [repair()], 40)
                self.assertEqual(result.dtype, original.dtype)
                self.assertEqual(result.shape, original.shape)

    def test_identical_pose_is_not_reencoded_or_perturbed(self):
        original, _ = poses()
        result, _ = localize_arm_pose(original, original.copy(), [repair()], 40)
        np.testing.assert_array_equal(result, original)

    def test_zero_original_and_constant_candidate_use_per_joint_gain_on_hit(self):
        original = np.zeros((100, 63))
        candidate = original.copy()
        candidate[:, columns("right")] = np.tile([0.0, 0.0, 0.4], 3)
        result, _ = localize_arm_pose(original, candidate, [repair()], 40)
        gains = [POLICY["correction_gains"][joint] for joint in ("shoulder", "elbow", "wrist")]
        expected = np.array([[0.0, 0.0, 0.4 * gain] for gain in gains]).reshape(9)
        np.testing.assert_allclose(result[40:43, columns("right")],
                                   np.broadcast_to(expected, (3, 9)), atol=1e-14)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_shortest_path_crosses_pi_without_collapsing_to_zero(self):
        original = np.zeros((100, 63))
        candidate = original.copy()
        z = 3 * (19 - 1) + 2
        original[:, z] = np.deg2rad(179)
        candidate[:, z] = np.deg2rad(-179)
        result, _ = localize_arm_pose(original, candidate, [repair()], 40)
        before = Rotation.from_rotvec(original[41, 54:57])
        after = Rotation.from_rotvec(result[41, 54:57])
        self.assertAlmostEqual((before.inv() * after).magnitude(), np.deg2rad(1), places=12)
        self.assertAlmostEqual(after.magnitude(), np.pi, places=12)

    def test_noncommuting_rotations_use_so3_geodesic_not_rotvec_average(self):
        original = np.zeros((100, 63))
        candidate = original.copy()
        original[:, 54:57] = [1.0, 0.0, 0.0]
        candidate[:, 54:57] = [0.0, 1.1, 0.3]
        result, _ = localize_arm_pose(original, candidate, [repair()], 40)
        before = Rotation.from_rotvec(original[41, 54:57])
        end = Rotation.from_rotvec(candidate[41, 54:57])
        expected = before * Rotation.from_rotvec(0.5 * (before.inv() * end).as_rotvec())
        actual = Rotation.from_rotvec(result[41, 54:57])
        self.assertLess((expected.inv() * actual).magnitude(), 1e-12)
        self.assertGreater(np.linalg.norm(result[41, 54:57] -
                                          0.5 * (original[41, 54:57] + candidate[41, 54:57])), 1e-3)

    def test_bilateral_hits_equal_independently_computed_sides(self):
        original, candidate = poses()
        left = repair(40, 43, "left_wrist")
        right = repair(45, 47, "right_elbow")
        together, _ = localize_arm_pose(original, candidate, [left, right], 40)
        left_only, _ = localize_arm_pose(original, candidate, [left], 40)
        right_only, _ = localize_arm_pose(original, candidate, [right], 40)
        np.testing.assert_array_equal(together[:, columns("left")], left_only[:, columns("left")])
        np.testing.assert_array_equal(together[:, columns("right")], right_only[:, columns("right")])

    def test_duplicate_and_reversed_hits_are_deterministic(self):
        original, candidate = poses()
        runs = [repair(40, 43), repair(47, 49, "right_elbow"), repair(50, 52, "left_wrist")]
        canonical, _ = localize_arm_pose(original, candidate, runs, 40)
        reversed_result, _ = localize_arm_pose(original, candidate, runs[::-1], 40)
        duplicate, _ = localize_arm_pose(original, candidate, runs + [runs[0], runs[1]], 40)
        np.testing.assert_array_equal(reversed_result, canonical)
        np.testing.assert_array_equal(duplicate, canonical)

    def test_touching_and_overlapping_same_side_hits_equal_merged_interval(self):
        original, candidate = poses()
        merged, _ = localize_arm_pose(original, candidate, [repair(40, 46)], 40)
        for runs in ([repair(40, 43), repair(43, 46, "right_elbow")],
                     [repair(40, 44), repair(42, 46, "right_elbow")]):
            with self.subTest(runs=runs):
                result, _ = localize_arm_pose(original, candidate, runs, 40)
                np.testing.assert_array_equal(result, merged)

    def test_overlapping_tapers_do_not_double_the_maximum_gain(self):
        original = np.zeros((100, 63))
        candidate = original.copy()
        candidate[:, 50] = 0.4
        runs = [repair(40, 42), repair(47, 49, "right_elbow")]
        result, _ = localize_arm_pose(original, candidate, runs, 40)
        reverse, _ = localize_arm_pose(original, candidate, runs[::-1], 40)
        np.testing.assert_array_equal(result, reverse)
        self.assertGreater(result[45, 50], 0)
        self.assertGreaterEqual(np.min(result[:, 50]), 0)
        self.assertLessEqual(np.max(result[:, 50]), 0.4 * POLICY["correction_gains"]["shoulder"] + 1e-14)

    def test_overlapping_tapers_use_smooth_union_not_pointwise_maximum(self):
        original = np.zeros((100, 63))
        candidate = original.copy()
        candidate[:, 50] = 0.4
        left, right = repair(40, 42), repair(47, 49, "right_elbow")
        first, _ = localize_arm_pose(original, candidate, [left], 40)
        second, _ = localize_arm_pose(original, candidate, [right], 40)
        combined, _ = localize_arm_pose(original, candidate, [left, right], 40)
        peak = first[40, 50]
        expected = peak * (1 - (1 - first[:, 50] / peak) * (1 - second[:, 50] / peak))
        np.testing.assert_allclose(combined[:, 50], expected, atol=1e-14)
        self.assertGreater(combined[44, 50], max(first[44, 50], second[44, 50]))

    def test_isolated_support_uses_c2_quintic_taper(self):
        original = np.zeros((100, 63))
        candidate = original.copy()
        candidate[:, 50] = 0.4
        result, _ = localize_arm_pose(original, candidate, [repair(40, 41)], 40)
        phase = np.linspace(0, 1, 11)
        expected = phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)
        np.testing.assert_allclose(result[30:41, 50] / result[40, 50], expected, atol=1e-14)
        np.testing.assert_allclose(result[40:51, 50] / result[40, 50], expected[::-1], atol=1e-14)

    def test_support_is_time_based_not_fixed_frame_count(self):
        for fps, hit in ((20, 20), (40, 40), (80, 80)):
            with self.subTest(fps=fps):
                original = np.zeros((200, 63))
                candidate = original.copy()
                candidate[:, 50] = 0.4
                result, _ = localize_arm_pose(original, candidate, [repair(hit, hit + 1)], fps)
                nonzero = np.flatnonzero(result[:, 50])
                self.assertGreaterEqual(nonzero[0], hit - int(np.ceil(0.25 * fps)))
                self.assertLessEqual(nonzero[-1], hit + int(np.ceil(0.25 * fps)))
                self.assertGreater(hit - nonzero[0], int(0.1 * fps))

    def test_invalid_pose_contracts_raise_value_error(self):
        original, candidate = poses()
        invalid = [original[:-1], original.reshape(100, 21, 3), original[:, :-1],
                   original[:0], original.astype(np.int32), original.astype(complex),
                   original.astype(str), original * np.nan, original * np.inf]
        for value in invalid:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    localize_arm_pose(value, candidate, [repair()], 40)
                with self.assertRaises(ValueError):
                    localize_arm_pose(original, value, [repair()], 40)

    def test_invalid_fps_raises_value_error(self):
        original, candidate = poses()
        for fps in (0, -1, np.nan, np.inf, True, "40", [40]):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                localize_arm_pose(original, candidate, [repair()], fps)

    def test_invalid_repair_intervals_and_unmapped_joints_raise(self):
        original, candidate = poses()
        invalid = [repair(-1, 3), repair(40, 40), repair(43, 40), repair(99, 101),
                   repair(40.0, 43), repair(True, 43), repair(40, 43.0),
                   repair(40, 43, "neck"), repair(40, 43, "left_ankle"),
                   {"joint": "right_wrist", "start_frame": 40}, {}]
        for run in invalid:
            with self.subTest(run=run), self.assertRaises(ValueError):
                localize_arm_pose(original, candidate, [run], 40)

    def test_missing_two_sided_anchors_is_rejected_not_silently_clipped(self):
        original, candidate = poses()
        for run in (repair(0, 3), repair(98, 100), repair(0, 100)):
            with self.subTest(run=run), self.assertRaises(ValueError):
                localize_arm_pose(original, candidate, [run], 40)


if __name__ == "__main__":
    unittest.main()
