"""Zero-plane grounding must preserve real flight, parameters and disabled mode."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hmr4d.backends.surface_ground import (
    add_surface_ground, export_ground, reference_calibration, surface_correction,
)
from test_motiforge_foot_surface import FootSurfaceExportTests, _FakeBodyModel


class SurfaceGroundTests(unittest.TestCase):
    def test_assume_grounded_projects_every_frame_and_preserves_articulation(self):
        arrays = FootSurfaceExportTests()._arrays(frames=120)
        drift = np.linspace(0, .3, 120).astype(np.float32)
        drift[40:80] += .4 * np.sin(np.linspace(0, np.pi, 40))
        arrays['pred_w_j3d'][..., 1] += drift[:, None]
        arrays['smpl_params_global.transl'][:, 1] += drift
        original = {k: v.copy() for k, v in arrays.items()}
        default = add_surface_ground(arrays, _FakeBodyModel(), enabled=True, model_digest='0'*64)
        result = add_surface_ground(arrays, _FakeBodyModel(), enabled=True,
                                    model_digest='0'*64, assume_grounded=True)
        self.assertGreater(float(default['foot_surface_y'].min(axis=1).max()), .3)
        np.testing.assert_allclose(result['foot_surface_y'].min(axis=1), 0, atol=1e-6)
        delta = result['smpl_params_global.transl']-arrays['smpl_params_global.transl']
        np.testing.assert_allclose(result['pred_w_j3d'], arrays['pred_w_j3d']+delta[:, None], atol=1e-6)
        np.testing.assert_array_equal(delta[:, [0, 2]], 0)
        for name in ('body_pose', 'global_orient', 'betas'):
            np.testing.assert_array_equal(result['smpl_params_global.'+name], arrays['smpl_params_global.'+name])
        np.testing.assert_array_equal(result['smpl_params_incam.transl'], arrays['smpl_params_incam.transl'])
        for key in original:
            np.testing.assert_array_equal(arrays[key], original[key])
        report = json.loads(str(result['motiforge_video_json']))['surface_ground']
        self.assertEqual(report['schema'], 'foot-surface-ground-v4')
        self.assertTrue(report['removes_real_flight'])
        self.assertIsNone(report['correction_speed_limit_mps'])
        self.assertIn('grounded_projection_fast_height_change', report['warnings'])
        self.assertFalse(report['support_anchor_enabled'])

    def test_assume_grounded_conflicts_are_explicit_and_cli_binds(self):
        arrays = FootSurfaceExportTests()._arrays(frames=120)
        for kwargs in ({'enabled': False}, {'enabled': True, 'reference_window': (0, .5)}):
            with self.assertRaisesRegex(ValueError, 'requires ground enabled'):
                add_surface_ground(arrays, _FakeBodyModel(), model_digest='0'*64,
                                   assume_grounded=True, **kwargs)
        from hmr4d.backends.motiforge import main
        with patch('hmr4d.backends.surface_ground.export_ground', return_value={}) as export:
            self.assertEqual(main(['export-ground', 'input.npz', '--output', 'new.npz',
                                   '--asset-root', '.', '--assume-grounded']), 0)
            self.assertTrue(export.call_args.kwargs['assume_grounded'])
            self.assertIsNone(export.call_args.kwargs['reference_window'])

    def test_explicit_constant_calibration_preserves_real_jump_and_swing(self):
        surface = np.full((120, 2), .25)
        surface[:, 1] += .1  # Only one foot needs to be grounded.
        jump = .4 * np.sin(np.linspace(0, np.pi, 20))
        surface[60:80] += jump[:, None]
        offset, report = reference_calibration(surface, 30, (0, .5))
        self.assertEqual(offset, .25)
        self.assertEqual(report['frame_window_half_open'], [0, 15])
        total = surface_correction(surface, np.zeros(120), 30, reference_offset=offset)
        np.testing.assert_allclose(total, .25)
        np.testing.assert_allclose((surface - total[:, None])[60:80, 0], jump, atol=1e-12)
        np.testing.assert_allclose(np.diff(surface - total[:, None], axis=0), np.diff(surface, axis=0))

    def test_reference_is_fixed_robust_and_never_inferred_from_whole_clip(self):
        surface = np.full((120, 2), .15)
        surface[2] = -2  # One bad reference sample must not define the floor.
        surface[30:] = .5  # Later values cannot change the initial calibration.
        offset, _ = reference_calibration(surface, 30, (0, .5))
        self.assertEqual(offset, .15)
        for window in [(0, 0), (-1, .5), (0, 8), (0, .05), (0, np.nan)]:
            with self.subTest(window=window), self.assertRaises(ValueError):
                reference_calibration(surface, 30, window)

    def test_reference_and_one_sided_guard_do_not_pull_later_float_down(self):
        surface = np.full((120, 2), .15)
        surface[40:60] -= .04
        surface[80:] += .2
        total = surface_correction(surface, np.zeros(120), 30, reference_offset=.15)
        self.assertGreaterEqual(float((surface-total[:, None]).min()), -1e-12)
        self.assertLessEqual(float(total.max()), .15000001)
        self.assertLessEqual(float(np.abs(np.diff(total)).max()*30), .20000001)
        np.testing.assert_allclose((surface-total[:, None])[80:], .2, atol=1e-12)

    def test_calibrated_evidence_has_distinct_schema_and_disabled_mode_rejects_it(self):
        arrays = FootSurfaceExportTests()._arrays(frames=120)
        result = add_surface_ground(arrays, _FakeBodyModel(), enabled=True,
                                    model_digest='0'*64, reference_window=(0., .5))
        report = json.loads(str(result['motiforge_video_json']))['surface_ground']
        self.assertEqual(report['schema'], 'foot-surface-ground-v3')
        self.assertFalse(report['support_anchor_enabled'])
        self.assertIn('reference_calibration', report)
        np.testing.assert_array_equal(result['smpl_params_global.body_pose'], arrays['smpl_params_global.body_pose'])
        np.testing.assert_array_equal(result['smpl_params_incam.transl'], arrays['smpl_params_incam.transl'])
        self.assertGreaterEqual(float(result['foot_surface_y'].min()), -1e-6)
        with self.assertRaisesRegex(ValueError, 'disabled'):
            add_surface_ground(arrays, _FakeBodyModel(), enabled=False,
                               model_digest='0'*64, reference_window=(0., .5))

    def test_command_requires_explicit_ground_assumption(self):
        from hmr4d.backends.motiforge import main
        argv = ['export-ground', 'input.npz', '--output', 'new.npz', '--asset-root', '.',
                '--reference-start', '0', '--reference-duration', '.5']
        with self.assertRaises(SystemExit) as result:
            main(argv)
        self.assertEqual(result.exception.code, 2)
        with patch('hmr4d.backends.surface_ground.export_ground', return_value={}) as export:
            self.assertEqual(main(argv + ['--assume-reference-grounded']), 0)
            self.assertEqual(export.call_args.kwargs['reference_window'], (0., .5))

    def test_only_negative_offsets_go_to_zero(self):
        for offset in (-.285, .25):
            surface = np.full((120, 2), offset)
            total = surface_correction(surface, np.zeros(120), 30)
            np.testing.assert_allclose(surface - total[:, None], max(0, offset), atol=1e-12)

    def test_valid_stance_is_exact_noop(self):
        surface = np.zeros((120, 2))
        surface[:, 1] = .1  # Other foot can be in swing.
        total = surface_correction(surface, np.zeros(120), 30)
        np.testing.assert_array_equal(total, 0)

    def test_real_jump_and_positive_baseline_are_not_flattened(self):
        surface = np.full((150, 2), .15)
        jump = .4 * np.sin(np.linspace(0, np.pi, 30))
        surface[60:90] += jump[:, None]
        total = surface_correction(surface, np.zeros(150), 30)
        np.testing.assert_array_equal(surface - total[:, None], surface)

    def test_unknown_positive_heights_are_not_automatic_contact(self):
        surface = np.full((120, 2), .3)
        total = surface_correction(surface, np.zeros(120), 30)
        np.testing.assert_array_equal(total, 0)

    def test_fast_negative_excursion_is_lifted_with_bounded_total_correction(self):
        surface = np.zeros((120, 2))
        surface[50] = -.1
        previous = np.linspace(-.1, .1, 120)
        total = surface_correction(surface, previous, 30)
        self.assertGreaterEqual(float(np.min(surface + previous[:, None] - total[:, None])), -1e-12)
        self.assertLessEqual(float(np.abs(np.diff(total)).max() * 30), .20000001)

    def test_only_world_translation_changes_and_input_is_preserved(self):
        arrays = FootSurfaceExportTests()._arrays(frames=120)
        arrays['left_contact_confidence'][:] = 1
        arrays['right_contact_confidence'][:] = 1
        original = {k: v.copy() for k, v in arrays.items()}
        result = add_surface_ground(arrays, _FakeBodyModel(), enabled=True, model_digest='0' * 64)
        delta = result['smpl_params_global.transl'] - arrays['smpl_params_global.transl']
        np.testing.assert_allclose(result['pred_w_j3d'], arrays['pred_w_j3d'] + delta[:, None], atol=1e-6)
        np.testing.assert_array_equal(delta[:, [0, 2]], 0)
        for key in ('body_pose', 'global_orient', 'betas'):
            np.testing.assert_array_equal(result['smpl_params_global.' + key], arrays['smpl_params_global.' + key])
        np.testing.assert_array_equal(result['smpl_params_incam.transl'], arrays['smpl_params_incam.transl'])
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], original[key])
        self.assertGreaterEqual(float(result['foot_surface_y'].min()), -1e-6)
        with self.assertRaisesRegex(ValueError, 'already evaluated'):
            add_surface_ground(result, _FakeBodyModel(), enabled=True, model_digest='0' * 64)

    def test_disabled_keeps_every_existing_numeric_array(self):
        arrays = FootSurfaceExportTests()._arrays()
        result = add_surface_ground(arrays, _FakeBodyModel(), enabled=False, model_digest='0' * 64)
        for key in arrays:
            if key != 'motiforge_video_json':
                np.testing.assert_array_equal(result[key], arrays[key])
        self.assertFalse(json.loads(str(result['motiforge_video_json']))['surface_ground']['enabled'])

    def test_static_probabilities_cannot_change_heights_or_rejection(self):
        arrays = FootSurfaceExportTests()._arrays(frames=120)
        results = []
        for probability in (0., 1.):
            for side in ('left', 'right'):
                arrays[side + '_contact_confidence'][:] = probability
            result = add_surface_ground(arrays, _FakeBodyModel(), enabled=True, model_digest='0' * 64)
            results.append(result)
            report = json.loads(str(result['motiforge_video_json']))['surface_ground']
            self.assertFalse(report['support_anchor_enabled'])
            self.assertNotIn('static_support_height_residual', report['errors'])
        np.testing.assert_array_equal(results[0]['pred_w_j3d'], results[1]['pred_w_j3d'])
        np.testing.assert_array_equal(results[0]['foot_surface_y'], results[1]['foot_surface_y'])

    def test_stale_geometry_and_invalid_surface_rejected(self):
        arrays = FootSurfaceExportTests()._arrays()
        arrays['smplx.trans'] = np.zeros((5, 3))
        with self.assertRaisesRegex(ValueError, 'derived geometry'):
            add_surface_ground(arrays, _FakeBodyModel(), enabled=True, model_digest='0' * 64)
        with self.assertRaisesRegex(ValueError, 'geometry'):
            surface_correction(np.full((10, 2), np.nan), np.zeros(10), 30)

    def test_cached_export_undoes_legacy_support_correction(self):
        arrays = FootSurfaceExportTests()._arrays()
        metadata = json.loads(str(arrays['motiforge_video_json']))
        metadata['ground_stabilization']['enabled'] = True
        # Dynamic camera bypasses the independent camera reference, isolating
        # whether the old custom support shift is really removed.
        metadata['static_camera'] = False
        arrays['motiforge_video_json'] = np.asarray(json.dumps(metadata))
        expected = arrays['smpl_params_global.transl'].copy()
        expected[:, 1] += arrays['floor_correction_y']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'original.npz', root / 'new.npz'
            np.savez_compressed(source, **arrays)
            original_bytes = source.read_bytes()
            with patch('hmr4d.backends.portable._load_body_model', return_value=_FakeBodyModel()), \
                 patch('hmr4d.backends.portable._sha256', return_value='0' * 64), \
                 patch('hmr4d.backends.smplx_sequence.add_sequence', side_effect=lambda a, *_, **kw: a):
                export_ground(source, target, root, backend_id='no-support-test')
            with np.load(target, allow_pickle=False) as result:
                np.testing.assert_allclose(result['smpl_params_global.transl'], expected)
                np.testing.assert_array_equal(result['floor_correction_y'], 0)
                report = json.loads(str(result['motiforge_video_json']))
                self.assertFalse(report['ground_stabilization']['support_anchor_enabled'])
                self.assertEqual(report['surface_ground_input']['ground_stabilization'],
                                 metadata['ground_stabilization'])
            self.assertEqual(source.read_bytes(), original_bytes)
            arrays.pop('floor_correction_y')
            broken = root / 'missing-correction.npz'
            np.savez_compressed(broken, **arrays)
            with self.assertRaisesRegex(ValueError, 'Cannot undo legacy'):
                export_ground(broken, root / 'invalid.npz', root, backend_id='test')

    def test_explicit_calibration_replaces_fresh_v2_instead_of_stacking(self):
        arrays = FootSurfaceExportTests()._arrays(frames=120)
        metadata = json.loads(str(arrays['motiforge_video_json']))
        metadata['static_camera'] = False
        metadata['ground_stabilization']['enabled'] = True
        arrays['motiforge_video_json'] = np.asarray(json.dumps(metadata))
        v2 = add_surface_ground(arrays, _FakeBodyModel(), enabled=True, model_digest='0'*64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in [('original', arrays), ('v2', v2)]:
                np.savez_compressed(root / (name+'.npz'), **value)
            with patch('hmr4d.backends.portable._load_body_model', return_value=_FakeBodyModel()), \
                 patch('hmr4d.backends.portable._sha256', return_value='0'*64), \
                 patch('hmr4d.backends.smplx_sequence.add_sequence', side_effect=lambda a, *_, **kw: a):
                for name in ('original', 'v2'):
                    export_ground(root/(name+'.npz'), root/(name+'-calibrated.npz'), root,
                                  backend_id='test', reference_window=(0, .5))
                with self.assertRaisesRegex(ValueError, 'already evaluated'):
                    export_ground(root/'v2.npz', root/'no-reference.npz', root, backend_id='test')
                with self.assertRaisesRegex(ValueError, 'already evaluated'):
                    export_ground(root/'v2-calibrated.npz', root/'twice.npz', root,
                                  backend_id='test', reference_window=(0, .5))
                for name in ('original', 'v2', 'v2-calibrated'):
                    export_ground(root/(name+'.npz'), root/(name+'-grounded.npz'), root,
                                  backend_id='test', assume_grounded=True)
                with self.assertRaisesRegex(ValueError, 'already evaluated'):
                    export_ground(root/'v2-grounded.npz', root/'twice-grounded.npz', root,
                                  backend_id='test', assume_grounded=True)
            with np.load(root/'original-calibrated.npz') as a, np.load(root/'v2-calibrated.npz') as b:
                for key in ('pred_w_j3d', 'smpl_params_global.transl', 'floor_correction_y'):
                    np.testing.assert_allclose(a[key], b[key], atol=1e-6)
                report = json.loads(str(b['motiforge_video_json']))
                self.assertEqual(report['surface_ground_input']['surface_ground']['schema'],
                                 'foot-surface-ground-v2')
            with np.load(root/'original-grounded.npz') as expected:
                for name in ('v2', 'v2-calibrated'):
                    with np.load(root/(name+'-grounded.npz')) as actual:
                        for key in ('pred_w_j3d', 'smpl_params_global.transl', 'floor_correction_y', 'foot_surface_y'):
                            np.testing.assert_allclose(expected[key], actual[key], atol=1e-6)
                        self.assertTrue(json.loads(str(actual['motiforge_video_json']))['assume_grounded'])


if __name__ == '__main__':
    unittest.main()
