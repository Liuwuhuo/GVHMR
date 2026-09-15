"""Headless GVHMR backend for the MotiForge video source adapter.

This module owns all GVHMR-specific preprocessing and inference.  MotiForge
only sends a versioned JSON request and consumes the portable ``.npz`` files
listed in the JSON response.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_VERSION = 3
BACKEND_NAME = "gvhmr-motiforge"
TARGET_FPS = 30
PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMMON_ASSETS = (
    "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
    "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
    "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
    "inputs/checkpoints/yolo/yolov8x.pt",
    "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
)
DPVO_ASSETS = ("inputs/checkpoints/dpvo/dpvo.pth",)
RUNTIME_MODULES = (
    "cv2",
    "hydra",
    "omegaconf",
    "pycolmap",
    "pytorch3d",
    "pytorch_lightning",
    "smplx",
    "torch",
    "ultralytics",
)

_STATIC_JOINT_IDS = (7, 10, 8, 11)
_CONTACT_THRESHOLD = 0.8
_MAX_GROUND_CORRECTION_M = 0.25
_MAX_GROUND_SPEED_MPS = 0.20
OBSERVATION_STABILITY_MODES = ("off", "audit", "conservative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backend_revision() -> str:
    """Return an identity for the protocol implementation itself."""

    # All numerical helpers affect cached predictions/diagnostics.
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), Path(__file__).with_name("observation_stability.py"),
                 Path(__file__).with_name("local_arm_repair.py"),
                 Path(__file__).with_name("arm_gap_repair.py"),
                 Path(__file__).with_name("smplx_sequence.py"),
                 Path(__file__).with_name("body_pose.py"),
                 Path(__file__).with_name("portable.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def project_revision() -> str:
    completed = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else "unknown"


def _ffmpeg_command(
    source: Path,
    destination: Path,
    *,
    mirror: bool,
    max_frames: int | None,
) -> list[str]:
    filters = [f"fps={TARGET_FPS}"]
    if mirror:
        filters.append("hflip")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        ",".join(filters),
    ]
    if max_frames is not None:
        command += ["-frames:v", str(max_frames)]
    command += [
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-map_metadata",
        "-1",
        str(destination),
    ]
    return command


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _link_runtime(runtime: Path, asset_root: Path) -> None:
    """Make GVHMR relative and ``PROJ_ROOT`` asset lookups agree."""

    runtime.mkdir(parents=True, exist_ok=True)
    links = {
        "hmr4d": PROJECT_ROOT / "hmr4d",
        "inputs": asset_root / "inputs",
    }
    third_party = PROJECT_ROOT / "third-party"
    if third_party.exists():
        links["third-party"] = third_party
    for name, target in links.items():
        link = runtime / name
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        link.symlink_to(target, target_is_directory=True)


def _activate_runtime(runtime: Path) -> None:
    """Point modules that import ``hmr4d.PROJ_ROOT`` at the asset overlay."""

    import hmr4d

    hmr4d.PROJ_ROOT = runtime
    os.chdir(runtime)


def _load_demo():
    demo_path = PROJECT_ROOT / "tools/demo/demo.py"
    spec = importlib.util.spec_from_file_location("gvhmr_motiforge_demo", demo_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 GVHMR demo：{demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("run_preprocess", "load_data_dict", "get_video_lwh", "register_store_gvhmr")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"GVHMR demo 缺少 backend API：{', '.join(missing)}")
    return module


def diagnose(
    asset_root: Path,
    *,
    use_dpvo: bool = False,
    require_cuda: bool = True,
    check_runtime: bool = True,
) -> dict[str, Any]:
    """Return a complete, non-downloading runtime readiness report."""

    asset_root = asset_root.expanduser().resolve()
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str, **extra: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail, **extra})

    for relative in COMMON_ASSETS + (DPVO_ASSETS if use_dpvo else ()):
        path = asset_root / relative
        ok = path.is_file() and path.stat().st_size > 0
        add(f"asset:{relative}", ok, str(path), size=path.stat().st_size if ok else 0)

    if use_dpvo:
        dpvo_source = PROJECT_ROOT / "third-party/DPVO/dpvo"
        add("dpvo-source", dpvo_source.is_dir(), str(dpvo_source))

    if check_runtime:
        ffmpeg = shutil.which("ffmpeg")
        add("ffmpeg", ffmpeg is not None, ffmpeg or "PATH 中未找到 ffmpeg")
        for module in RUNTIME_MODULES:
            try:
                available = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError) as exc:
                available = False
                detail = f"{type(exc).__name__}: {exc}"
            else:
                detail = "available" if available else "not importable"
            add(f"python:{module}", available, detail)

        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda_detail = f"torch={torch.__version__}, devices={torch.cuda.device_count()}"
        except Exception as exc:  # noqa: BLE001 - doctor must aggregate environment failures
            cuda_available = False
            cuda_detail = f"{type(exc).__name__}: {exc}"
        add("cuda", cuda_available or not require_cuda, cuda_detail)

        try:
            _load_demo()
        except Exception as exc:  # noqa: BLE001 - report API/import drift as one check
            add("backend-api", False, f"{type(exc).__name__}: {exc}")
        else:
            add("backend-api", True, "run_preprocess/load_data_dict contract available")

    errors = [check for check in checks if not check["ok"]]
    return {
        "backend": BACKEND_NAME,
        "protocol": PROTOCOL_VERSION,
        "project_root": str(PROJECT_ROOT),
        "asset_root": str(asset_root),
        "gvhmr_revision": project_revision(),
        "backend_revision": backend_revision(),
        "capabilities": {"observation_stability": list(OBSERVATION_STABILITY_MODES)},
        "ok": not errors,
        "checks": checks,
        "errors": [f"{item['name']}: {item['detail']}" for item in errors],
    }


def _compose_config(demo, output_root: Path, options: dict[str, Any]):
    from hydra import compose, initialize_config_module
    from omegaconf import open_dict

    overrides = [
        "video_name=motiforge",
        f"static_cam={bool(options.get('static_camera', False))}",
        f"use_dpvo={bool(options.get('use_dpvo', False))}",
        "verbose=False",
    ]
    if options.get("focal_mm") is not None:
        overrides.append(f"f_mm={float(options['focal_mm']):g}")
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        demo.register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)
    with open_dict(cfg):
        cfg.output_root = str(output_root)
        cfg.simple_vo_workers = int(options.get("simple_vo_workers", 1))
    return cfg


def _load_model(cfg):
    import hydra

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    return model.eval().cuda()


def _normalize_video(source: Path, destination: Path, options: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = _ffmpeg_command(
        source,
        destination,
        mirror=bool(options.get("mirror", False)),
        max_frames=options.get("max_frames"),
    )
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1:] or [f"退出码 {completed.returncode}"]
        raise RuntimeError(f"ffmpeg 视频标准化失败：{detail[0]}")


def _contact_confidence(pred: dict[str, Any], frame_count: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract the checkpoint's semantic left/right foot contact confidence."""

    net_outputs = pred.get("net_outputs")
    if not isinstance(net_outputs, dict):
        return None
    logits = net_outputs.get("static_conf_logits")
    if logits is None:
        return None
    values = _numpy_array(logits).astype(np.float64, copy=False)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or values.shape[1] < len(_STATIC_JOINT_IDS):
        raise RuntimeError(
            "GVHMR static_conf_logits 形状异常："
            f"expected (T, >=4), got {values.shape}"
        )
    if values.shape[0] == frame_count - 1:
        values = np.concatenate((values, values[-1:]), axis=0)
    if values.shape[0] != frame_count:
        raise RuntimeError(
            "GVHMR 接触置信度帧数异常："
            f"expected {frame_count}, got {values.shape[0]}"
        )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(values[:, :4], -40.0, 40.0)))
    left = np.maximum(probabilities[:, 0], probabilities[:, 1])
    right = np.maximum(probabilities[:, 2], probabilities[:, 3])
    return left.astype(np.float32), right.astype(np.float32)


def _remove_short_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    """Drop isolated contact predictions without eroding sustained support."""

    result = np.asarray(mask, dtype=bool).copy()
    start = None
    for index, active in enumerate(np.append(result, False)):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start < minimum:
                result[start:index] = False
            start = None
    return result


def _smooth_curve(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window -= 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    median = np.asarray(
        [np.median(padded[index : index + window]) for index in range(len(values))]
    )
    weights = np.arange(1, radius + 2, dtype=np.float64)
    weights = np.concatenate((weights, weights[-2::-1]))
    weights /= weights.sum()
    return np.convolve(np.pad(median, (radius, radius), mode="edge"), weights, mode="valid")


def _lipschitz_minorant(values: np.ndarray, max_step: float) -> np.ndarray:
    """Largest two-sided rate-limited curve that never exceeds ``values``."""

    limited = np.asarray(values, dtype=np.float64).copy()
    for index in range(1, len(limited)):
        limited[index] = min(limited[index], limited[index - 1] + max_step)
    for index in range(len(limited) - 2, -1, -1):
        limited[index] = min(limited[index], limited[index + 1] + max_step)
    return limited


def _stabilize_world_ground(
    joints: np.ndarray,
    transl: np.ndarray,
    contacts: tuple[np.ndarray, np.ndarray] | None,
    *,
    fps: float,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove slow vertical world drift between reliable static-foot anchors.

    The upstream static-joint correction is horizontal-only; its separate
    camera-root correction has a dead zone that can leave vertical drift.
    We use the model's own static-foot logits as floor
    observations, interpolate the slow floor component through missing support,
    and apply the same correction to joints and SMPL translation.
    """

    world = np.asarray(joints, dtype=np.float32).copy()
    root = np.asarray(transl, dtype=np.float32).copy()
    frame_count = world.shape[0]
    correction = np.zeros(frame_count, dtype=np.float32)
    diagnostics: dict[str, Any] = {
        "version": "contact-floor-v1",
        "enabled": bool(enabled),
        "applied": False,
        "contact_threshold": _CONTACT_THRESHOLD,
    }
    if contacts is None:
        diagnostics["reason"] = "contact_confidence_unavailable"
        return world, root, correction, diagnostics

    left_confidence, right_confidence = contacts
    left_y = np.minimum(world[:, 7, 1], world[:, 10, 1]).astype(np.float64)
    right_y = np.minimum(world[:, 8, 1], world[:, 11, 1]).astype(np.float64)
    minimum_run = max(2, round(fps * 0.10))
    left_contact = _remove_short_runs(left_confidence >= _CONTACT_THRESHOLD, minimum_run)
    right_contact = _remove_short_runs(right_confidence >= _CONTACT_THRESHOLD, minimum_run)
    support = left_contact | right_contact
    diagnostics.update(
        {
            "contact_frames": {
                "left": int(left_contact.sum()),
                "right": int(right_contact.sum()),
            },
            "static_support_missing_frames": int((~support).sum()),
            # Deprecated compatibility alias: low static probability is not flight.
            "flight_frames": int((~support).sum()),
        }
    )
    if not enabled:
        diagnostics["reason"] = "disabled"
        return world, root, correction, diagnostics
    if support.sum() < max(3, minimum_run):
        diagnostics["reason"] = "insufficient_contact"
        return world, root, correction, diagnostics

    support_y = np.full(frame_count, np.nan, dtype=np.float64)
    only_left = left_contact & ~right_contact
    only_right = right_contact & ~left_contact
    both = left_contact & right_contact
    support_y[only_left] = left_y[only_left]
    support_y[only_right] = right_y[only_right]
    support_y[both] = np.minimum(left_y[both], right_y[both])
    anchor_indices = np.flatnonzero(support)
    anchor_values = support_y[anchor_indices]
    target_height = float(np.percentile(anchor_values, 10.0))
    sampled_correction = anchor_values - target_height
    timeline = np.arange(frame_count)
    interpolated = np.interp(timeline, anchor_indices, sampled_correction)
    window = max(3, round(fps * 0.25) | 1)
    stabilized = _smooth_curve(interpolated, window)
    stabilized = np.clip(
        stabilized,
        -_MAX_GROUND_CORRECTION_M,
        _MAX_GROUND_CORRECTION_M,
    )

    # Never turn a low foot into ground penetration.  This constraint also
    # prevents an erroneous future contact anchor from pulling down a real
    # jump whose current support evidence is absent.  Project both the desired
    # curve and its ceiling onto the same Lipschitz bound so releasing that
    # ceiling at toe-off cannot create a one-frame root jump.
    lowest_foot = np.minimum(left_y, right_y)
    max_step = _MAX_GROUND_SPEED_MPS / fps
    stabilized = np.minimum(
        _lipschitz_minorant(stabilized, max_step),
        _lipschitz_minorant(lowest_foot - target_height, max_step),
    )
    if float(np.max(np.abs(stabilized))) < 0.005:
        diagnostics.update(
            {
                "reason": "correction_below_threshold",
                "target_foot_height_m": target_height,
                "max_abs_correction_m": float(np.max(np.abs(stabilized))),
            }
        )
        return world, root, correction, diagnostics

    correction = stabilized.astype(np.float32)
    world[:, :, 1] -= correction[:, None]
    root[:, 1] -= correction
    corrected_left_y = np.minimum(world[:, 7, 1], world[:, 10, 1])
    corrected_right_y = np.minimum(world[:, 8, 1], world[:, 11, 1])
    corrected_support = np.where(
        left_contact & right_contact,
        np.minimum(corrected_left_y, corrected_right_y),
        np.where(left_contact, corrected_left_y, corrected_right_y),
    )
    residual = np.abs(corrected_support[support] - target_height)
    diagnostics.update(
        {
            "applied": True,
            "target_foot_height_m": target_height,
            "max_abs_correction_m": float(np.max(np.abs(correction))),
            "p95_abs_correction_m": float(np.percentile(np.abs(correction), 95.0)),
            "contact_height_p95_error_m": float(np.percentile(residual, 95.0)),
            "max_correction_speed_mps": float(
                np.max(np.abs(np.diff(correction))) * fps if frame_count > 1 else 0.0
            ),
        }
    )
    return world, root, correction, diagnostics


def _static_camera_height_correction(
    joints: np.ndarray,
    transl: np.ndarray,
    global_orient: np.ndarray,
    incam: dict[str, Any] | None,
    *,
    fps: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Low-pass only the world/raw-camera height discrepancy, not the motion.

    Use the same frame-zero camera rotation as ``pp_static_joint_cam``. A
    vertical motion shared by both representations cancels before filtering,
    so common jumps and squats do not become a camera correction.
    """

    if incam is None:
        raise ValueError("incam parameters unavailable")
    world = np.asarray(joints, dtype=np.float64)
    frame_count = len(world)
    if frame_count < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("static camera height requires at least two frames and positive fps")
    arrays = {}
    for name, value in (
        ("global.transl", transl),
        ("global.global_orient", global_orient),
        ("incam.transl", incam.get("transl")),
        ("incam.global_orient", incam.get("global_orient")),
    ):
        array = np.asarray(_numpy_array(value), dtype=np.float64)
        if array.shape != (frame_count, 3) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape ({frame_count}, 3)")
        arrays[name] = array
    if not np.all(np.isfinite(world)):
        raise ValueError("world joints must be finite")

    def rotation(axis_angle: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(axis_angle))
        if angle == 0.0:
            return np.eye(3)
        x, y, z = axis_angle / angle
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)

    camera_to_world = rotation(arrays["global.global_orient"][0]) @ rotation(
        arrays["incam.global_orient"][0]
    ).T
    pelvis_offset = world[:, 0] - arrays["global.transl"]
    camera_pelvis = pelvis_offset + arrays["incam.transl"]
    reference = camera_pelvis @ camera_to_world.T
    reference += world[0, 0] - reference[0]
    discrepancy = world[:, 0, 1] - reference[:, 1]

    # NumPy equivalent of Gaussian sigma=0.5s, truncate=4, nearest padding.
    # Do not separately smooth incam: that would filter shared real motion.
    sigma = 0.5 * fps
    radius = int(4.0 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    correction = np.convolve(
        np.pad(discrepancy, (radius, radius), mode="edge"), weights, mode="valid"
    )
    correction -= correction[0]
    return correction, {
        "version": "raw-static-incam-height-v1",
        "applied": bool(np.any(correction != 0.0)),
        "reference": "frame_zero_camera_to_world",
        "camera_to_world_rotation": camera_to_world.tolist(),
        "discrepancy_gaussian_sigma_seconds": 0.5,
        "incam_prefilter": "none",
        "max_abs_correction_m": float(np.max(np.abs(correction))),
        "p95_abs_correction_m": float(np.percentile(np.abs(correction), 95.0)),
    }


def _stabilize_prediction_ground(
    joints: np.ndarray,
    transl: np.ndarray,
    contacts: tuple[np.ndarray, np.ndarray] | None,
    *,
    global_orient: np.ndarray,
    incam: dict[str, Any] | None,
    fps: float,
    enabled: bool,
    static_camera: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compose static-camera height and legacy floor corrections in source Y.

    Dynamic cameras, disabled stabilization and missing support keep the exact
    legacy numerical path. Invalid camera parameters also fall back explicitly.
    The single exported correction is applied once to the original prediction
    and its *total* speed, not just each stage's speed, is bounded at 0.2 m/s.
    """

    legacy = _stabilize_world_ground(joints, transl, contacts, fps=fps, enabled=enabled)
    if not enabled or not static_camera:
        return legacy
    if legacy[3].get("reason") in {"contact_confidence_unavailable", "insufficient_contact"}:
        legacy[3]["camera_stage"] = {"applied": False, "reason": legacy[3]["reason"]}
        return legacy
    try:
        camera_correction, camera_diagnostics = _static_camera_height_correction(
            joints, transl, global_orient, incam, fps=fps
        )
    except (TypeError, ValueError, AttributeError) as exc:
        legacy[3]["camera_stage"] = {
            "applied": False,
            "reason": "invalid_or_missing_incam_parameters",
            "detail": str(exc),
        }
        return legacy

    camera_world = np.asarray(joints, dtype=np.float64).copy()
    camera_root = np.asarray(transl, dtype=np.float64).copy()
    camera_world[:, :, 1] -= camera_correction[:, None]
    camera_root[:, 1] -= camera_correction
    _, _, floor_correction, floor_diagnostics = _stabilize_world_ground(
        camera_world, camera_root, contacts, fps=fps, enabled=True
    )
    desired_total = camera_correction + floor_correction
    correction = _lipschitz_minorant(desired_total, _MAX_GROUND_SPEED_MPS / fps).astype(
        np.float32
    )
    world = np.asarray(joints, dtype=np.float32).copy()
    root = np.asarray(transl, dtype=np.float32).copy()
    world[:, :, 1] -= correction[:, None]
    root[:, 1] -= correction

    minimum_run = max(2, round(fps * 0.10))
    left_contact, right_contact = (
        _remove_short_runs(confidence >= _CONTACT_THRESHOLD, minimum_run)
        for confidence in contacts
    )
    left_y = np.minimum(world[:, 7, 1], world[:, 10, 1])
    right_y = np.minimum(world[:, 8, 1], world[:, 11, 1])
    support_y = np.minimum(
        np.where(left_contact, left_y, np.inf), np.where(right_contact, right_y, np.inf)
    )[left_contact | right_contact]
    target_height = floor_diagnostics["target_foot_height_m"]
    diagnostics = {
        "version": "static-camera-contact-floor-v2",
        "enabled": True,
        "applied": bool(np.any(correction != 0.0)),
        "contact_threshold": _CONTACT_THRESHOLD,
        "contact_frames": floor_diagnostics["contact_frames"],
        "static_support_missing_frames": floor_diagnostics["static_support_missing_frames"],
        "flight_frames": floor_diagnostics["flight_frames"],  # Deprecated alias, not flight.
        "camera_stage": camera_diagnostics,
        "floor_stage": floor_diagnostics,
        "correction_composition": "camera_plus_floor_then_lipschitz_minorant",
        "total_speed_limit_mps": _MAX_GROUND_SPEED_MPS,
        "total_projection_max_change_m": float(np.max(np.abs(correction - desired_total))),
        "target_foot_height_m": target_height,
        "max_abs_correction_m": float(np.max(np.abs(correction))),
        "p95_abs_correction_m": float(np.percentile(np.abs(correction), 95.0)),
        "contact_height_p95_error_m": float(
            np.percentile(np.abs(support_y - target_height), 95.0)
        ),
        "max_correction_speed_mps": float(np.max(np.abs(np.diff(correction))) * fps),
    }
    return world, root, correction, diagnostics


def _portable_prediction(
    *,
    pred,
    model,
    detach_to_cpu,
    item: dict[str, Any],
    options: dict[str, Any],
    revision: str,
    backend_id: str,
    normalized_frames: int,
) -> dict[str, Any]:
    global_params = pred["smpl_params_global"]
    batched = {key: value[None] for key, value in global_params.items()}
    pred_w_j3d = model.pipeline.endecoder.fk_v2(**batched)[0].detach().cpu().numpy()
    if pred_w_j3d.ndim != 3 or pred_w_j3d.shape[0] != normalized_frames or pred_w_j3d.shape[-1] != 3:
        raise RuntimeError(
            "GVHMR 世界关节形状异常："
            f"expected ({normalized_frames}, J, 3), got {tuple(pred_w_j3d.shape)}"
        )
    contacts = _contact_confidence(pred, normalized_frames)
    detached_global = detach_to_cpu(pred["smpl_params_global"])
    detached_global = dict(detached_global)
    detached_incam = detach_to_cpu(pred.get("smpl_params_incam", {}))
    pred_w_j3d, stabilized_transl, correction, ground_diagnostics = _stabilize_prediction_ground(
        pred_w_j3d,
        _numpy_array(detached_global["transl"]),
        contacts,
        global_orient=_numpy_array(detached_global["global_orient"]),
        incam=detached_incam,
        fps=float(TARGET_FPS),
        enabled=bool(options.get("ground_stabilization", True)),
        static_camera=bool(options.get("static_camera", False)),
    )
    detached_global["transl"] = stabilized_transl
    portable = {
        "smpl_params_global": detached_global,
        "smpl_params_incam": detached_incam,
        "K_fullimg": detach_to_cpu(pred["K_fullimg"]),
        "pred_w_j3d": pred_w_j3d,
        "floor_correction_y": correction,
        "motiforge_video": {
            "protocol": PROTOCOL_VERSION,
            "source_path": item["source"],
            "source_sha256": item["source_sha256"],
            "fps": float(TARGET_FPS),
            "normalized_num_frames": int(normalized_frames),
            "gvhmr_revision": revision,
            "gvhmr_backend_revision": backend_id,
            "static_camera": bool(options.get("static_camera", False)),
            "use_dpvo": bool(options.get("use_dpvo", False)),
            "simple_vo_workers": int(options.get("simple_vo_workers", 1)),
            "focal_mm": options.get("focal_mm"),
            "mirror": bool(options.get("mirror", False)),
            "max_frames": options.get("max_frames"),
            "ground_stabilization": ground_diagnostics,
        },
    }
    if contacts is not None:
        portable["left_contact_confidence"] = contacts[0]
        portable["right_contact_confidence"] = contacts[1]
    return portable


def _portable_arrays(portable: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten a prediction into an allow_pickle=False NPZ schema."""

    arrays: dict[str, np.ndarray] = {}
    for key, value in portable.items():
        if key == "motiforge_video":
            arrays["motiforge_video_json"] = np.asarray(
                json.dumps(value, ensure_ascii=False, sort_keys=True)
            )
            continue
        if key in {"smpl_params_global", "smpl_params_incam"}:
            for field, field_value in value.items():
                arrays[f"{key}.{field}"] = _numpy_array(field_value)
            continue
        arrays[key] = _numpy_array(value)
    return arrays


def _numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def _write_portable_npz(path: Path, portable: dict[str, Any]) -> None:
    """Atomically write a portable artifact without pickle or Torch loading."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **_portable_arrays(portable))
    os.replace(temporary, path)


def _prepare_observation_data(data, mode="audit", *, image_size=None, boxes=None):
    """Audit raw observations; optionally repair only the validated short-gap policy.

    The native preprocess cache is never changed. Audit and no-effective-hit
    paths return the original model input object, not a resampled/smoothed copy.
    Scores retain their native meaning (visibility > 0.5, not probability).
    """
    if mode not in OBSERVATION_STABILITY_MODES:
        raise ValueError(f"Unknown observation_stability mode: {mode!r}")
    if mode == "off":
        return data, None
    from hmr4d.backends.observation_stability import (
        POLICY,
        audit_observations,
        detect_and_repair,
    )

    original = _numpy_array(data["kp2d"])
    crops = _numpy_array(data["bbx_xys"])
    if crops.shape != (len(original), 3) or not np.isfinite(crops).all():
        raise ValueError("bbx_xys must be finite shape (T, 3)")
    crop_scale = crops[:, 2]
    observations = audit_observations(
        original, crop_scale, float(TARGET_FPS), image_size=image_size, boxes=boxes,
    )
    report = {
        "schema_version": 1,
        "mode": mode,
        "observation_audit": observations,
        "warnings": list(observations["warnings"]),
        "timeline_policy": "preserve_all_frames; no trimming or unanchored gap extrapolation",
        "tracking_evidence": "selected smoothed bbox only; raw detector presence and identity unknown",
    }
    if mode == "conservative":
        candidate, accepted, rejected = detect_and_repair(original, crop_scale, float(TARGET_FPS))
        changed = np.any(candidate[..., :2] != original[..., :2], axis=-1)
        # normalize_kp2d also masks points outside the crop. A visible-to-
        # invisible transition still changes the model input; two invisible
        # positions do not. Match the upstream inclusive crop boundaries.
        lower = crops[:, None, :2] - crop_scale[:, None, None] / 2
        upper = crops[:, None, :2] + crop_scale[:, None, None] / 2
        original_inside = np.all((original[..., :2] >= lower) & (original[..., :2] <= upper), axis=-1)
        candidate_inside = np.all((candidate[..., :2] >= lower) & (candidate[..., :2] <= upper), axis=-1)
        effective = changed & (original[..., 2] > 0.5) & (original_inside | candidate_inside)
        rejection_counts: dict[str, int] = {}
        for entry in rejected:
            for reason in entry["rejection_reasons"]:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        report["keypoint_repair"] = {
            "policy": POLICY,
            "accepted": accepted,
            "effective_runs": [
                entry for entry in accepted
                if effective[entry["start_frame"]:entry["stop_frame_exclusive"], entry["joint_index"]].any()
            ],
            "rejected_run_count": len(rejected),
            "rejection_reason_counts": rejection_counts,
            "candidate_joint_frames": int(changed.sum()),
            "visible_changed_joint_frames": int(effective.sum()),
            "applied_joint_frames": int(changed.sum()) if effective.any() else 0,
            "confidence_unchanged": True,
            "effective_visibility": "score > 0.5 and inside inclusive native crop before or after repair",
            "applied": bool(effective.any()),
            "limitation": "Interpolation is a hypothesis, not observed truth; whole-clip prediction may change.",
        }
        if effective.any():
            import torch

            data = dict(data)
            data["kp2d"] = torch.as_tensor(candidate, dtype=data["kp2d"].dtype, device=data["kp2d"].device)
            report["warnings"].append("short_keypoint_repair_applied")
    return data, report


def _select_observation_candidate(original, candidate, report):
    """Choose one complete human prediction; never splice poses or use a robot."""
    from hmr4d.backends.observation_stability import evaluate_repair_candidate

    repair = report["keypoint_repair"]
    decision = evaluate_repair_candidate(
        original["pred_w_j3d"], candidate["pred_w_j3d"], float(TARGET_FPS), repair["effective_runs"],
    )
    report["candidate_acceptance"] = decision
    repair["applied"] = bool(decision["accepted"])
    report["warnings"] = [code for code in report["warnings"]
                          if code not in {"short_keypoint_repair_applied", "keypoint_repair_rolled_back"}]
    if decision["accepted"]:
        report["warnings"].append("short_keypoint_repair_applied")
    if not decision["accepted"]:
        repair["applied_joint_frames"] = 0
        report["warnings"].append("keypoint_repair_rolled_back")
    return candidate if decision["accepted"] else original


def _arm_pose_prediction(original, pose, model):
    """Replace shared local pose, keeping roots/shape/evidence; recompute native FK."""
    import copy

    import torch

    if not np.array_equal(_numpy_array(original["smpl_params_global"]["body_pose"]),
                          _numpy_array(original["smpl_params_incam"]["body_pose"])):
        raise ValueError("native global/incam body_pose must share the same local rotations")
    localized = copy.deepcopy(original)
    localized["smpl_params_global"]["body_pose"] = pose
    localized["smpl_params_incam"]["body_pose"] = pose.copy()
    decoder = model.pipeline.endecoder
    # Use the already-loaded native FK, not a second body-model installation.
    device = decoder.parents_tensor.device
    params = {key: torch.as_tensor(_numpy_array(value), device=device)[None]
              for key, value in localized["smpl_params_global"].items()}
    with torch.inference_mode():
        localized["pred_w_j3d"] = decoder.fk_v2(**params)[0].detach().cpu().numpy()
    return localized


def _localize_observation_candidate(original, candidate, report, model):
    """Build the existing arm-only model-correction hypothesis, not independent XYZ."""
    from hmr4d.backends.local_arm_repair import localize_arm_pose

    if not np.array_equal(_numpy_array(candidate["smpl_params_global"]["body_pose"]),
                          _numpy_array(candidate["smpl_params_incam"]["body_pose"])):
        raise ValueError("native global/incam body_pose must share the same local rotations")
    pose, evidence = localize_arm_pose(
        _numpy_array(original["smpl_params_global"]["body_pose"]),
        _numpy_array(candidate["smpl_params_global"]["body_pose"]),
        report["keypoint_repair"]["effective_runs"], float(TARGET_FPS),
    )
    report["local_arm_repair"] = evidence
    return _arm_pose_prediction(original, pose, model)


def _interpolate_observation_gaps(original, report, data, model, image_size):
    """Bridge only observed short arm gaps from reliable original-pose neighborhoods."""
    from hmr4d.backends.arm_gap_repair import interpolate_arm_gaps

    pose, evidence = interpolate_arm_gaps(
        _numpy_array(original["smpl_params_global"]["body_pose"]),
        report["keypoint_repair"]["effective_runs"], _numpy_array(data["kp2d"]),
        _numpy_array(data["bbx_xys"]), float(TARGET_FPS), image_size=image_size,
    )
    report["arm_gap_repair"] = evidence
    if not evidence["changed_pose_frames"]:
        return None
    return _arm_pose_prediction(original, pose, model)


def _predict_observation_candidate(*, data, mode, image_size, boxes, model, static_cam, make_portable):
    """One native prediction, plus a candidate only for effective repair proposals.

    Keep an already acceptable whole-model repair. Otherwise try one localized
    arm hypothesis and a short-gap reconstruction, preserving root/shape/ground.
    A gap must pass the original guard and beat any accepted local correction
    under that same guard. The last two returns preserve both FK hypotheses.
    Inference failures remain explicit batch failures.
    """
    prepared, report = _prepare_observation_data(data, mode, image_size=image_size, boxes=boxes)
    original = make_portable(model.predict(data, static_cam=static_cam))
    if prepared is data:
        if mode == "conservative":
            report["candidate_acceptance"] = {
                "accepted": False, "reasons": ["no_effective_input_change"],
                "selected": "original", "prediction_passes": 1,
            }
        return original, report, original, None, None, None
    candidate = make_portable(model.predict(prepared, static_cam=static_cam))
    selected = _select_observation_candidate(original, candidate, report)
    report["model_candidate_acceptance"] = report["candidate_acceptance"]
    localized = gap = None
    if selected is original:
        localized = _localize_observation_candidate(original, candidate, report, model)
        selected = _select_observation_candidate(original, localized, report)
        report["local_candidate_acceptance"] = report["candidate_acceptance"]
        gap = _interpolate_observation_gaps(original, report, data, model, image_size)
        if gap is not None:
            from hmr4d.backends.observation_stability import evaluate_repair_candidate

            repairs = report["keypoint_repair"]["effective_runs"]
            decision = evaluate_repair_candidate(original["pred_w_j3d"], gap["pred_w_j3d"], float(TARGET_FPS), repairs)
            report["gap_candidate_acceptance"] = decision
            accepted = decision["accepted"]
            if accepted and selected is localized:
                comparison = evaluate_repair_candidate(
                    localized["pred_w_j3d"], gap["pred_w_j3d"], float(TARGET_FPS), repairs,
                )
                report["gap_vs_local_acceptance"] = comparison
                accepted = comparison["accepted"]
            if accepted:
                selected = gap
                report["candidate_acceptance"] = decision
    # Final applied state follows the chosen artifact, not an intermediate vote.
    repair = report["keypoint_repair"]
    repair["applied"] = selected is not original
    repair["applied_joint_frames"] = repair["candidate_joint_frames"] if repair["applied"] else 0
    report["warnings"] = [code for code in report["warnings"]
                          if code not in {"short_keypoint_repair_applied", "keypoint_repair_rolled_back"}]
    report["warnings"].append("short_keypoint_repair_applied" if repair["applied"] else "keypoint_repair_rolled_back")
    report["candidate_acceptance"]["prediction_passes"] = 2
    report["candidate_acceptance"]["selected"] = (
        "candidate" if selected is candidate else "gap_candidate" if selected is gap
        else "localized_candidate" if selected is localized else "original"
    )
    return selected, report, original, candidate, localized, gap


def _finish_observation_stability(portable, report):
    """Attach advisory evidence only; never edit human geometry or quality grades."""
    if report is None:
        return
    from hmr4d.backends.observation_stability import audit_world_motion

    temporal = audit_world_motion(portable["pred_w_j3d"], float(TARGET_FPS))
    report["human_temporal_audit"] = temporal
    report["warnings"] = sorted(set(report["warnings"] + temporal["warnings"]))
    portable["motiforge_video"]["observation_stability"] = report


def _process_item(
    *,
    item: dict[str, Any],
    options: dict[str, Any],
    revision: str,
    backend_id: str,
    demo,
    model,
    detach_to_cpu,
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    output_dir = Path(item["output_dir"])
    cfg = _compose_config(demo, output_dir / "native", options)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    _normalize_video(Path(item["source"]), Path(cfg.video_path), options)

    normalized_frames, width, height = map(int, demo.get_video_lwh(cfg.video_path))
    if normalized_frames < 2:
        raise RuntimeError(f"视频标准化后只有 {normalized_frames} 帧，GVHMR 至少需要 2 帧")
    demo.run_preprocess(cfg)
    data = demo.load_data_dict(cfg)
    mode = options.get("observation_stability", "audit")
    boxes = None
    if mode != "off":
        import torch

        boxes = _numpy_array(torch.load(cfg.paths.bbx, map_location="cpu", weights_only=True)["bbx_xyxy"])
    if model is None:
        model = _load_model(cfg)
    def make_portable(pred):
        return _portable_prediction(
            pred=pred, model=model, detach_to_cpu=detach_to_cpu, item=item,
            options=options, revision=revision, backend_id=backend_id,
            normalized_frames=normalized_frames,
        )

    portable, stability, original, candidate, localized, gap = _predict_observation_candidate(
        data=data, mode=mode, image_size=(width, height), boxes=boxes,
        model=model, static_cam=cfg.static_cam, make_portable=make_portable,
    )
    if candidate is not None:
        _write_portable_npz(output_dir / "observation_original.npz", original)
        _write_portable_npz(output_dir / "observation_candidate.npz", candidate)
        stability["candidate_evidence"] = {
            "original": "observation_original.npz", "candidate": "observation_candidate.npz",
            "scope": "complete original and whole-model candidate before temporal selection",
        }
    if localized is not None:
        _write_portable_npz(output_dir / "observation_local_candidate.npz", localized)
        stability["candidate_evidence"]["localized"] = "observation_local_candidate.npz"
        stability["candidate_evidence"]["localized_scope"] = "arm-local SO3/FK fallback before temporal selection"
    if gap is not None:
        _write_portable_npz(output_dir / "observation_gap_candidate.npz", gap)
        stability["candidate_evidence"]["gap"] = "observation_gap_candidate.npz"
        stability["candidate_evidence"]["gap_scope"] = "short anchored local-rotation gap/FK before temporal selection"
    _finish_observation_stability(portable, stability)
    if stability is not None:
        _atomic_json(output_dir / "observation_stability.json", stability)

    prediction = Path(item["prediction"])
    from hmr4d.backends.smplx_sequence import enrich_prediction

    enrich_prediction(portable, Path("inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"))
    _write_portable_npz(prediction, portable)
    manifest = {
        "status": "complete",
        "protocol": PROTOCOL_VERSION,
        "cache_key": item["cache_key"],
        "source": item["source"],
        "source_sha256": item["source_sha256"],
        "prediction": str(prediction),
        "gvhmr_revision": revision,
        "gvhmr_backend_revision": backend_id,
        "options": options,
        "normalized_num_frames": normalized_frames,
        "seconds": time.perf_counter() - started,
    }
    _atomic_json(Path(item["manifest"]), manifest)
    return model, manifest


def run_request(request_path: Path, response_path: Path) -> int:
    response: dict[str, Any] = {
        "backend": BACKEND_NAME,
        "protocol": PROTOCOL_VERSION,
        "results": [],
    }
    items: list[dict[str, Any]] = []
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if request.get("protocol") != PROTOCOL_VERSION:
            raise RuntimeError(f"不支持的 MotiForge/GVHMR 协议：{request.get('protocol')}")
        revision = project_revision()
        backend_id = backend_revision()
        if request.get("gvhmr_revision") != revision:
            raise RuntimeError("请求中的 GVHMR revision 与当前 checkout 不一致")
        if request.get("backend_revision") != backend_id:
            raise RuntimeError("请求中的 GVHMR backend revision 与当前代码不一致")
        asset_root = Path(request["asset_root"]).expanduser().resolve()
        options = dict(request.get("options", {}))
        if options.get("observation_stability", "audit") not in OBSERVATION_STABILITY_MODES:
            raise ValueError("observation_stability must be off, audit or conservative")
        items = list(request.get("items", []))
        if not items:
            raise RuntimeError("GVHMR backend 没有收到视频")

        runtime = Path(items[0]["output_dir"]).parent / f".runtime-{revision[:12]}-{backend_id}"
        _link_runtime(runtime, asset_root)
        _activate_runtime(runtime)
        report = diagnose(asset_root, use_dpvo=bool(options.get("use_dpvo")))
        response["doctor"] = report
        if not report["ok"]:
            raise RuntimeError("GVHMR doctor 未通过：" + "; ".join(report["errors"]))

        demo = _load_demo()
        import torch

        from hmr4d.utils.net_utils import detach_to_cpu

        model = None
        for item in items:
            try:
                model, manifest = _process_item(
                    item=item,
                    options=options,
                    revision=revision,
                    backend_id=backend_id,
                    demo=demo,
                    model=model,
                    detach_to_cpu=detach_to_cpu,
                )
                response["results"].append(
                    {
                        "cache_key": item["cache_key"],
                        "ok": True,
                        "seconds": manifest["seconds"],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - isolate one failed video in a batch
                response["results"].append(
                    {
                        "cache_key": item.get("cache_key", ""),
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8),
                    }
                )
            finally:
                torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - serialize process-boundary failures
        detail = f"{type(exc).__name__}: {exc}"
        response["error"] = detail
        existing = {str(item.get("cache_key")) for item in response["results"]}
        for item in items:
            if str(item.get("cache_key")) not in existing:
                response["results"].append(
                    {"cache_key": item.get("cache_key", ""), "ok": False, "error": detail}
                )
    finally:
        _atomic_json(response_path, response)
    return 0 if "error" not in response else 1


def _doctor_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate the GVHMR MotiForge backend runtime")
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--use-dpvo", action="store_true")
    parser.add_argument("--allow-no-cuda", action="store_true")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="gvhmr-motiforge-doctor-") as temporary:
        runtime = Path(temporary)
        _link_runtime(runtime, args.asset_root.resolve())
        _activate_runtime(runtime)
        report = diagnose(
            args.asset_root,
            use_dpvo=args.use_dpvo,
            require_cuda=not args.allow_no_cuda,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["capabilities"]:
        # No checkpoints, Torch imports or CUDA initialization for negotiation.
        print(json.dumps({
            "backend": BACKEND_NAME,
            "protocol": PROTOCOL_VERSION,
            "backend_revision": backend_revision(),
            "capabilities": {"observation_stability": list(OBSERVATION_STABILITY_MODES),
                             "smplx_sequence": "smplx-sequence-v1"},
        }))
        return 0
    if argv[:1] == ["doctor"]:
        return _doctor_command(argv[1:])
    if argv[:1] == ["export-smplx"]:
        parser = argparse.ArgumentParser(description="Export an engine-independent SMPL-X sequence")
        parser.add_argument("input", type=Path)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--asset-root", required=True, type=Path)
        args = parser.parse_args(argv[1:])
        from hmr4d.backends.smplx_sequence import export_sequence

        try:
            report = export_sequence(args.input, args.output, args.asset_root)
        except Exception as exc:
            print(f"SMPL-X export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if argv[:1] == ["export-body-pose"]:
        parser = argparse.ArgumentParser(description="Add validated SMPL-X body22 pose evidence to a new portable NPZ")
        parser.add_argument("input", type=Path)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--asset-root", required=True, type=Path)
        args = parser.parse_args(argv[1:])
        from hmr4d.backends.body_pose import export_body_pose

        try:
            report = export_body_pose(args.input, args.output, args.asset_root, backend_id=backend_revision())
        except Exception as exc:  # noqa: BLE001 - explicit process-boundary diagnostic
            print(f"body-pose export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if argv[:1] == ["export-foot-surface"]:
        parser = argparse.ArgumentParser(description="Add optional SMPL-X foot-surface evidence to a new portable NPZ")
        parser.add_argument("input", type=Path)
        parser.add_argument("--output", required=True, type=Path)
        parser.add_argument("--asset-root", required=True, type=Path)
        args = parser.parse_args(argv[1:])
        from hmr4d.backends.foot_surface import export_foot_surface

        try:
            report = export_foot_surface(args.input, args.output, args.asset_root, backend_id=backend_revision())
        except Exception as exc:  # noqa: BLE001 - explicit process-boundary diagnostic
            print(f"foot-surface export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if len(argv) == 3 and argv[0] == "run":
        return run_request(Path(argv[1]), Path(argv[2]))
    print(
        "usage: python -m hmr4d.backends.motiforge "
        "capabilities | doctor --asset-root DIR | run REQUEST_JSON RESPONSE_JSON | "
        "export-foot-surface INPUT --output OUTPUT --asset-root DIR | "
        "export-body-pose INPUT --output OUTPUT --asset-root DIR | "
        "export-smplx INPUT --output OUTPUT --asset-root DIR",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
