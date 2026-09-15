"""Public SMPL-X sequence preserves source FK, timeline and original arrays."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hmr4d.backends.smplx_sequence import Y_TO_Z, add_sequence, sequence_evidence
import test_motiforge_body_pose as fixture


class SMPLXSequenceTests(unittest.TestCase):
    def setUp(self):
        self.model = fixture.FakeBodyModel()
        self.model.bm.left_hand_mean = torch.linspace(-.1, .2, 45)
        self.model.bm.right_hand_mean = torch.linspace(.2, -.3, 45)
        self.arrays = fixture.BodyPoseExportTests().arrays()

    def test_standard_parameters_reconstruct_rotated_native_joints_with_pelvis_pivot(self):
        seq, meta = sequence_evidence(self.arrays, self.model)
        with torch.no_grad():
            actual = self.model(
                global_orient=torch.tensor(seq['poses'][:, :3], dtype=torch.float32),
                body_pose=torch.tensor(seq['poses'][:, 3:66], dtype=torch.float32),
                transl=torch.tensor(seq['trans'], dtype=torch.float32),
                betas=torch.tensor(seq['betas'], dtype=torch.float32),
            ).joints.numpy()
        np.testing.assert_allclose(actual, self.arrays['pred_w_j3d'] @ Y_TO_Z.T, atol=5e-7)
        np.testing.assert_allclose(seq['positions'], actual, atol=5e-7)
        naive = self.arrays['smpl_params_global.transl'] @ Y_TO_Z.T
        self.assertGreater(float(np.max(np.abs(seq['trans'] - naive))), .1)
        np.testing.assert_array_equal(seq['poses'][:, 75:120], np.broadcast_to(self.model.bm.left_hand_mean.numpy(), (5, 45)))
        np.testing.assert_array_equal(seq['poses'][:, 120:], np.broadcast_to(self.model.bm.right_hand_mean.numpy(), (5, 45)))
        self.assertEqual(meta['hands'], 'model-mean-unobserved')
        self.assertTrue(meta['full_sequence'])
        # Serialized global and bind frames reconstruct the same world bones.
        q = Rotation.from_quat(seq['rotations'][..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(5, 22, 3, 3)
        bind_r = Rotation.from_quat(seq['bind_rotations'][..., [1, 2, 3, 0]]).as_matrix()
        for j, parent in enumerate(meta['parents'][1:], 1):
            local = bind_r[parent].T @ (seq['bind_positions'][j] - seq['bind_positions'][parent])
            np.testing.assert_allclose(q[:, parent] @ local, seq['positions'][:, j] - seq['positions'][:, parent], atol=5e-7)

    def test_additive_extension_and_double_export_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model.npz'
            model.write_bytes(b'fixture')
            result = add_sequence(self.arrays, model, model_factory=lambda path: self.model)
            for key, value in self.arrays.items():
                if key != 'motiforge_video_json':
                    np.testing.assert_array_equal(result[key], value)
                    self.assertEqual(result[key].dtype, value.dtype)
            old_meta = json.loads(str(self.arrays['motiforge_video_json']))
            new_meta = json.loads(str(result['motiforge_video_json']))
            self.assertEqual({k: new_meta[k] for k in old_meta}, old_meta)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                add_sequence(result, model, model_factory=lambda path: self.model)

    def test_dynamic_shape_rejected_and_limited_source_not_marked_full(self):
        meta = json.loads(str(self.arrays['motiforge_video_json']))
        meta['max_frames'] = 5
        self.arrays['motiforge_video_json'] = np.asarray(json.dumps(meta))
        _, evidence = sequence_evidence(self.arrays, self.model)
        self.assertFalse(evidence['full_sequence'])
        self.arrays['smpl_params_global.betas'][1, 0] += .1
        with self.assertRaisesRegex(ValueError, 'constant betas'):
            sequence_evidence(self.arrays, self.model)


if __name__ == '__main__':
    unittest.main()
