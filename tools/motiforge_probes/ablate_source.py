"""Isolated GVHMR source-height ablations; does not modify production defaults.

Run with the existing GVHMR Python. Restored native values include native GVHMR
postprocessing and carry float32 inversion uncertainty, not raw-network outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from hmr4d.backends.foot_surface import export_foot_surface
from hmr4d.backends.motiforge import (
    _lipschitz_minorant,
    _stabilize_prediction_ground,
    _stabilize_world_ground,
    _static_camera_height_correction,
    backend_revision,
)
from hmr4d.backends.portable import _validated_prediction, _write_new_artifact


def derive(input_path: Path, output_root: Path, asset_root: Path, case: str) -> dict:
    arrays, metadata, digest = _validated_prediction(input_path)
    if not metadata.get("static_camera") or not metadata["ground_stabilization"].get("applied"):
        raise ValueError("This ablation requires a corrected static-camera source")
    if any(key.startswith("body22_") or key == "foot_surface_y" for key in arrays):
        raise ValueError("Start from plain position-only portable source, not stale evidence")
    stored = arrays["floor_correction_y"]
    raw_world, raw_root = arrays["pred_w_j3d"].copy(), arrays["smpl_params_global.transl"].copy()
    raw_world[:, :, 1] += stored[:, None]
    raw_root[:, 1] += stored
    contacts = tuple(arrays[f"{side}_contact_confidence"] for side in ("left", "right"))
    incam = {
        key.split(".", 1)[1]: value for key, value in arrays.items() if key.startswith("smpl_params_incam.")
    }
    fps = float(metadata["fps"])
    orient = arrays["smpl_params_global.global_orient"]
    combined = _stabilize_prediction_ground(
        raw_world,
        raw_root,
        contacts,
        global_orient=orient,
        incam=incam,
        fps=fps,
        enabled=True,
        static_camera=True,
    )
    delta = float(np.max(np.abs(combined[0] - arrays["pred_w_j3d"])))
    correction_delta = float(np.max(np.abs(combined[2] - stored)))
    if max(delta, correction_delta) > 1e-5:
        raise ValueError(f"Reconstructed combined source differs: {delta}, {correction_delta}")
    camera, camera_info = _static_camera_height_correction(raw_world, raw_root, orient, incam, fps=fps)
    camera = _lipschitz_minorant(camera, 0.2 / fps).astype(np.float32)
    floor = _stabilize_world_ground(raw_world, raw_root, contacts, fps=fps, enabled=True)
    variants = {
        "native": (np.zeros_like(stored), {"reason": "remove_product_height_correction"}),
        "camera-only": (camera, camera_info),
        "floor-only": (floor[2], floor[3]),
    }
    report = {
        "case": case,
        "input": str(input_path.resolve()),
        "input_sha256": digest,
        "backend_revision": backend_revision(),
        "frames": len(stored),
        "fps": fps,
        "combined_reconstruction_max_joints_error_m": delta,
        "combined_reconstruction_max_correction_error_m": correction_delta,
        "native_definition": (
            "Native GVHMR postprocessed source before product height correction; float32 reconstruction"
        ),
        "variants": {},
    }
    for variant, (correction, diagnostics) in variants.items():
        result = {key: value.copy() for key, value in arrays.items()}
        result["pred_w_j3d"] = raw_world.copy()
        result["smpl_params_global.transl"] = raw_root.copy()
        result["pred_w_j3d"][:, :, 1] -= correction[:, None]
        result["smpl_params_global.transl"][:, 1] -= correction
        result["floor_correction_y"] = correction
        info = copy.deepcopy(metadata)
        info["ground_stabilization"] = {
            "version": "research-height-ablation-v1",
            "enabled": variant != "native",
            "applied": bool(np.any(correction)),
            "variant": variant,
            "diagnostics": diagnostics,
        }
        info["source_experiment"] = {
            "kind": "height-stage-ablation",
            "input_artifact": str(input_path.resolve()),
            "input_artifact_sha256": digest,
            "variant": variant,
            "backend_revision": backend_revision(),
            "original_ground_stabilization": metadata["ground_stabilization"],
            "reconstruction_max_error_m": delta,
            "uses_bvh_or_manual_contact": False,
        }
        result["motiforge_video_json"] = np.asarray(json.dumps(info, sort_keys=True))
        for key in arrays:
            if key not in (
                "pred_w_j3d",
                "smpl_params_global.transl",
                "floor_correction_y",
                "motiforge_video_json",
            ):
                if not np.array_equal(result[key], arrays[key]):
                    raise ValueError(f"Unexpected array change: {key}")
        for key in ("pred_w_j3d", "smpl_params_global.transl"):
            if not np.array_equal(result[key][..., [0, 2]], arrays[key][..., [0, 2]]):
                raise ValueError("Horizontal coordinates changed")
        plain = output_root / "plain" / variant / f"{case}.npz"
        enhanced = output_root / "inputs" / variant / f"{case}-surface.npz"
        _write_new_artifact(plain, result)
        evidence = export_foot_surface(plain, enhanced, asset_root, backend_id=backend_revision())
        report["variants"][variant] = {
            "plain": str(plain.resolve()),
            "enhanced": str(enhanced.resolve()),
            "correction_min_m": float(correction.min()),
            "correction_max_m": float(correction.max()),
            "correction_at_14s_m": float(correction[420]) if len(correction) > 420 else None,
            "surface_export": evidence,
        }
        print(case, variant, enhanced, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="case=portable.npz")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for entry in args.source:
        case, path = entry.split("=", 1)
        if not case or Path(case).name != case:
            raise ValueError("Case must be a single path component")
        report = derive(Path(path), args.output_root, args.asset_root, case)
        with (args.output_root / f"{case}-ablation.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
