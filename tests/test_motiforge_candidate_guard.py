"""No-model tests for whole-clip source candidate selection."""

import json
import unittest

import numpy as np

from hmr4d.backends.observation_stability import evaluate_repair_candidate


def motion(frames=90):
    # Distinct, nonzero bones; articulated arms have a real elbow angle.
    points = np.array([
        [0, 1, 0], [-.1, 1, 0], [.1, 1, 0], [0, 1.1, 0],
        [-.1, .6, 0], [.1, .6, 0], [0, 1.2, 0], [-.1, .2, 0], [.1, .2, 0],
        [0, 1.3, 0], [-.1, .1, .1], [.1, .1, .1], [0, 1.5, 0],
        [-.05, 1.4, 0], [.05, 1.4, 0], [0, 1.6, 0], [-.2, 1.4, 0], [.2, 1.4, 0],
        [-.5, 1.4, 0], [.5, 1.4, 0], [-.7, 1.4, .2], [.7, 1.4, .2],
    ])
    return np.broadcast_to(points, (frames, 22, 3)).copy()


def repair(frame=30, joint="left_wrist"):
    return {"joint": joint, "start_frame": frame, "stop_frame_exclusive": frame + 1}


class CandidateGuardTests(unittest.TestCase):
    def test_improved_spike_accepted_without_mutation(self):
        candidate = motion()
        original = candidate.copy()
        original[30, 20, 1] += .15
        copies = original.copy(), candidate.copy()
        result = evaluate_repair_candidate(original, candidate, 30, [repair()])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["reasons"], [])
        self.assertTrue(result["windows"][0]["improved_channels"])
        np.testing.assert_array_equal(original, copies[0])
        np.testing.assert_array_equal(candidate, copies[1])
        json.dumps(result, allow_nan=False)

    def test_larger_spike_rejected(self):
        original = motion()
        original[30, 20, 1] += .10
        candidate = original.copy()
        candidate[30, 20, 1] += .10
        report = evaluate_repair_candidate(original, candidate, 30, [repair()])
        self.assertFalse(report["accepted"])
        self.assertIn("temporal_channel_regression", report["reasons"])

    def test_unchanged_fast_motion_not_called_improvement_or_regression(self):
        original = motion()
        original[:, 20, 1] += .1 * np.sin(np.arange(len(original)))
        result = evaluate_repair_candidate(original, original.copy(), 30, [repair()])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reasons"], ["no_local_acceleration_improvement"])
        self.assertEqual(result["violations"], [])

    def test_opposite_limb_cannot_hide_regression(self):
        original = motion()
        original[30, 20, 1] += .2
        candidate = motion()
        candidate[30, 21, 1] += .03
        result = evaluate_repair_candidate(original, candidate, 30, [repair()])
        self.assertFalse(result["accepted"])
        self.assertTrue(result["windows"][0]["improved_channels"])
        self.assertTrue(any(item["channel"] == "root_relative_21" for item in result["violations"]))

    def test_two_windows_require_each_improvement(self):
        original = motion()
        original[20, 20, 1] += .2
        original[60, 20, 1] += .1
        candidate = original.copy()
        candidate[20, 20, 1] -= .2
        result = evaluate_repair_candidate(original, candidate, 30, [repair(20), repair(60)])
        self.assertFalse(result["accepted"])
        self.assertIn("no_local_acceleration_improvement", result["reasons"])

    def test_outside_window_spike_rejected_even_below_old_global_peak(self):
        original = motion()
        original[20, 20, 1] += .2
        candidate = motion()
        candidate[65, 20, 1] += .03
        result = evaluate_repair_candidate(original, candidate, 30, [repair(20)])
        self.assertFalse(result["accepted"])
        self.assertTrue(any(item["region"] == "outside_windows" for item in result["violations"]))

    def test_global_translation_oscillation_not_hidden_by_root_relative_channels(self):
        original = motion()
        original[30, 20, 1] += .2
        candidate = motion()
        candidate[30, :, 1] += .1
        result = evaluate_repair_candidate(original, candidate, 30, [repair()])
        self.assertFalse(result["accepted"])
        self.assertTrue(any(item["channel"] == "root_world" for item in result["violations"]))

    def test_bad_candidate_fails_closed_and_serializes(self):
        original = motion()
        for candidate in (original[:-1], original * np.nan, original[..., :2], original.astype(str),
                          np.zeros_like(original), [], [[[1], [2, 3]]]):
            with self.subTest(shape=np.shape(candidate) if isinstance(candidate, np.ndarray) else "ragged"):
                result = evaluate_repair_candidate(original, candidate, 30, [repair()])
                self.assertEqual(result["reasons"], ["invalid_candidate"])
                self.assertFalse(result["accepted"])
                json.dumps(result, allow_nan=False)

    def test_bad_reference_or_contract_raises(self):
        original = motion()
        for reference in (original * np.nan, original[:0], np.zeros_like(original)):
            with self.assertRaises(ValueError):
                evaluate_repair_candidate(reference, original, 30, [repair()])
        for fps in (0, np.inf, [30], "30"):
            with self.assertRaises(ValueError):
                evaluate_repair_candidate(original, original, fps, [repair()])
        for repairs in ({}, [repair(90)], [repair(-1)], [repair(True)], [repair(30.0)],
                        [{"joint": "neck", "start_frame": 30, "stop_frame_exclusive": 31}]):
            with self.assertRaises(ValueError):
                evaluate_repair_candidate(original, original, 30, repairs)

    def test_insufficient_evidence_rejects_without_fabricating_improvement(self):
        original = motion()
        self.assertEqual(evaluate_repair_candidate(original, original, 30, [])["reasons"], ["no_effective_repairs"])
        short = original[:3]
        result = evaluate_repair_candidate(short, short.copy(), 30, [repair(1)])
        self.assertEqual(result["reasons"], ["insufficient_temporal_evidence"])


if __name__ == "__main__":
    unittest.main()
