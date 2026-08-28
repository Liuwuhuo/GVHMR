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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backend_revision() -> str:
    """Return an identity for the protocol implementation itself."""

    return _sha256(Path(__file__).resolve())[:12]


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
    pred_w_j3d = model.pipeline.endecoder.fk_v2(**batched)[0]
    if pred_w_j3d.ndim != 3 or pred_w_j3d.shape[0] != normalized_frames or pred_w_j3d.shape[-1] != 3:
        raise RuntimeError(
            "GVHMR 世界关节形状异常："
            f"expected ({normalized_frames}, J, 3), got {tuple(pred_w_j3d.shape)}"
        )
    return {
        "smpl_params_global": detach_to_cpu(pred["smpl_params_global"]),
        "smpl_params_incam": detach_to_cpu(pred["smpl_params_incam"]),
        "K_fullimg": detach_to_cpu(pred["K_fullimg"]),
        "pred_w_j3d": pred_w_j3d.detach().cpu(),
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
        },
    }


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

    normalized_frames = int(demo.get_video_lwh(cfg.video_path)[0])
    if normalized_frames < 2:
        raise RuntimeError(f"视频标准化后只有 {normalized_frames} 帧，GVHMR 至少需要 2 帧")
    demo.run_preprocess(cfg)
    data = demo.load_data_dict(cfg)
    if model is None:
        model = _load_model(cfg)
    pred = model.predict(data, static_cam=cfg.static_cam)
    portable = _portable_prediction(
        pred=pred,
        model=model,
        detach_to_cpu=detach_to_cpu,
        item=item,
        options=options,
        revision=revision,
        backend_id=backend_id,
        normalized_frames=normalized_frames,
    )

    prediction = Path(item["prediction"])
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
    if argv[:1] == ["doctor"]:
        return _doctor_command(argv[1:])
    if len(argv) == 3 and argv[0] == "run":
        return run_request(Path(argv[1]), Path(argv[2]))
    print(
        "usage: python -m hmr4d.backends.motiforge "
        "doctor --asset-root DIR | run REQUEST_JSON RESPONSE_JSON",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
