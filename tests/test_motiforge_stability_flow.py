"""Adapter mode contract, without checkpoints or a GPU."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from hmr4d.backends import motiforge as backend


def inputs(hit=False, invisible=False):
    kp = torch.ones(30, 17, 3)
    kp[..., :2] = 50
    kp[:, 9, 0] = 10
    kp[:, 10, 0] = 90
    if hit:
        kp[10:12, 9, 0] = 90
        kp[10:12, 9, 2] = 0.4 if invisible else 0.6
    return {"kp2d": kp, "bbx_xys": torch.full((30, 3), 100.)}


class StabilityFlowTests(unittest.TestCase):
    def test_off_skips_audit_and_accepts_legacy_model_data(self):
        data = {"legacy": object()}
        same, report = backend._prepare_observation_data(data, "off")
        self.assertIs(same, data)
        self.assertIsNone(report)

    def test_audit_keeps_original_inputs_even_with_a_repairable_hit(self):
        data = inputs(hit=True)
        before = data["kp2d"].clone()
        same, report = backend._prepare_observation_data(data, "audit")
        self.assertIs(same, data)
        self.assertTrue(torch.equal(data["kp2d"], before))
        self.assertNotIn("keypoint_repair", report)
        self.assertEqual(report["mode"], "audit")

    def test_conservative_repairs_new_tensor_but_not_raw_cache_or_scores(self):
        data = inputs(hit=True)
        before = data["kp2d"].clone()
        changed, report = backend._prepare_observation_data(data, "conservative")
        self.assertIsNot(changed, data)
        self.assertTrue(torch.equal(data["kp2d"], before))
        self.assertTrue(torch.equal(changed["kp2d"][..., 2], before[..., 2]))
        self.assertTrue(torch.all(changed["kp2d"][10:12, 9, 0] == 10))
        self.assertEqual(report["keypoint_repair"]["visible_changed_joint_frames"], 2)

    def test_no_hit_or_invisible_only_returns_identical_model_input(self):
        for data in (inputs(), inputs(hit=True, invisible=True)):
            same, report = backend._prepare_observation_data(data, "conservative")
            self.assertIs(same, data)
            self.assertFalse(report["keypoint_repair"]["applied"])

    def test_repair_outside_native_crop_is_not_an_effective_model_change(self):
        data = inputs(hit=True)
        data["bbx_xys"][:, :2] = 300
        same, report = backend._prepare_observation_data(data, "conservative")
        self.assertIs(same, data)
        self.assertEqual(report["keypoint_repair"]["candidate_joint_frames"], 2)
        self.assertEqual(report["keypoint_repair"]["visible_changed_joint_frames"], 0)

    def test_crop_visibility_transition_is_an_effective_change(self):
        data = inputs(hit=True)  # x=90 inside [50,150], repaired x=10 outside
        changed, report = backend._prepare_observation_data(data, "conservative")
        self.assertIsNot(changed, data)
        self.assertEqual(report["keypoint_repair"]["visible_changed_joint_frames"], 2)

    def test_final_audit_only_adds_metadata_and_never_moves_any_points(self):
        _, report = backend._prepare_observation_data(inputs(), "audit")
        portable = {"pred_w_j3d": np.ones((30, 22, 3)), "motiforge_video": {"fps": 30}}
        original = copy.deepcopy(portable)
        backend._finish_observation_stability(portable, report)
        np.testing.assert_array_equal(portable["pred_w_j3d"], original["pred_w_j3d"])
        json.dumps(report, allow_nan=False)
        self.assertEqual(portable["motiforge_video"]["fps"], 30)
        self.assertIn("human_temporal_audit", report)

    def test_invalid_mode_is_not_silently_disabled(self):
        with self.assertRaises(ValueError):
            backend._prepare_observation_data(inputs(), "aggressive")

    def test_process_item_applies_policy_before_model_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bbox = root / "bbx.pt"
            torch.save({"bbx_xyxy": torch.tensor([[0., 0., 100., 100.]]).repeat(30, 1)}, bbox)
            cfg = SimpleNamespace(output_dir=root, preprocess_dir=root, video_path=root / "input.mp4",
                                  static_cam=True, paths=SimpleNamespace(bbx=bbox))
            data = inputs(hit=True)
            seen = []
            def predict(model_data, **kwargs):
                seen.append(model_data["kp2d"].clone())
                return {}
            demo = SimpleNamespace(get_video_lwh=lambda p: (30, 200, 200),
                                   run_preprocess=lambda c: None, load_data_dict=lambda c: data)
            item = {"output_dir": str(root), "source": "source.mp4", "source_sha256": "sha", "cache_key": "key",
                    "prediction": str(root / "result.npz"), "manifest": str(root / "manifest.json")}
            portable = {"pred_w_j3d": np.ones((30, 22, 3)), "motiforge_video": {"fps": 30}}
            with patch.object(backend, "_compose_config", return_value=cfg), \
                 patch.object(backend, "_normalize_video"), \
                 patch.object(backend, "_portable_prediction", side_effect=lambda **kwargs: copy.deepcopy(portable)), \
                 patch("hmr4d.backends.observation_stability.evaluate_repair_candidate",
                       return_value={"accepted": True, "reasons": []}):
                backend._process_item(item=item, options={"observation_stability": "conservative"},
                                      revision="rev", backend_id="backend", demo=demo,
                                      model=SimpleNamespace(predict=predict), detach_to_cpu=lambda x: x)
            self.assertEqual(len(seen), 2)
            self.assertTrue(torch.all(seen[0][10:12, 9, 0] == 90))
            self.assertTrue(torch.all(seen[1][10:12, 9, 0] == 10))
            self.assertTrue(torch.all(data["kp2d"][10:12, 9, 0] == 90))
            self.assertTrue((root / "observation_stability.json").is_file())
            self.assertTrue((root / "observation_original.npz").is_file())
            self.assertTrue((root / "observation_candidate.npz").is_file())
            with np.load(root / "result.npz", allow_pickle=False) as saved:
                meta = json.loads(saved["motiforge_video_json"].item())
            self.assertEqual(meta["observation_stability"]["mode"], "conservative")
            self.assertTrue(meta["observation_stability"]["keypoint_repair"]["applied"])

    def run_predictions(self, data, mode):
        seen = []
        def predict(value, **kwargs):
            seen.append(value)
            return {"pred_w_j3d": np.full((30, 22, 3), len(seen), dtype=float),
                    "motiforge_video": {"fps": 30}, "other_payload": np.array([len(seen)])}
        result = backend._predict_observation_candidate(
            data=data, mode=mode, image_size=None, boxes=None,
            model=SimpleNamespace(predict=predict), static_cam=True, make_portable=lambda pred: pred,
        )
        return result, seen

    def test_no_effective_change_and_audit_off_predict_once(self):
        for mode, data in (("off", inputs(True)), ("audit", inputs(True)),
                           ("conservative", inputs()), ("conservative", inputs(True, True))):
            with self.subTest(mode=mode):
                (selected, report, original, candidate), seen = self.run_predictions(data, mode)
                self.assertEqual(len(seen), 1)
                self.assertIs(seen[0], data)
                self.assertIs(selected, original)
                self.assertIsNone(candidate)
                if mode == "conservative":
                    self.assertEqual(report["candidate_acceptance"]["prediction_passes"], 1)

    def test_guard_rejection_returns_whole_original_and_corrects_provenance(self):
        with patch("hmr4d.backends.observation_stability.evaluate_repair_candidate",
                   return_value={"accepted": False, "reasons": ["local_regression"]}):
            (selected, report, original, candidate), seen = self.run_predictions(inputs(True), "conservative")
        self.assertEqual(len(seen), 2)
        self.assertIs(selected, original)
        self.assertIsNot(selected, candidate)
        np.testing.assert_array_equal(selected["other_payload"], [1])
        self.assertFalse(report["keypoint_repair"]["applied"])
        self.assertEqual(report["keypoint_repair"]["applied_joint_frames"], 0)
        self.assertEqual(report["candidate_acceptance"]["selected"], "original")
        self.assertIn("keypoint_repair_rolled_back", report["warnings"])
        self.assertNotIn("short_keypoint_repair_applied", report["warnings"])

    def test_guard_acceptance_returns_whole_candidate(self):
        with patch("hmr4d.backends.observation_stability.evaluate_repair_candidate",
                   return_value={"accepted": True, "reasons": []}):
            (selected, report, original, candidate), seen = self.run_predictions(inputs(True), "conservative")
        self.assertEqual(len(seen), 2)
        self.assertIs(selected, candidate)
        self.assertIsNot(selected, original)
        self.assertEqual(report["candidate_acceptance"]["prediction_passes"], 2)
        self.assertTrue(report["keypoint_repair"]["applied"])
        self.assertEqual(report["candidate_acceptance"]["selected"], "candidate")
        np.testing.assert_array_equal(selected["other_payload"], [2])

    def test_candidate_prediction_exception_is_not_silently_accepted(self):
        from unittest.mock import Mock
        model = SimpleNamespace(predict=Mock(side_effect=[{}, RuntimeError("model failed")]))
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            backend._predict_observation_candidate(
                data=inputs(True), mode="conservative", image_size=None, boxes=None,
                model=model, static_cam=True, make_portable=lambda pred: pred,
            )


if __name__ == "__main__":
    unittest.main()
