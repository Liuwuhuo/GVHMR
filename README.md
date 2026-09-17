# GVHMR: World-Grounded Human Motion Recovery via Gravity-View Coordinates
### [Project Page](https://zju3dv.github.io/gvhmr) | [Paper](https://arxiv.org/abs/2409.06662)

> World-Grounded Human Motion Recovery via Gravity-View Coordinates  
> [Zehong Shen](https://zehongs.github.io/)<sup>\*</sup>,
[Huaijin Pi](https://phj128.github.io/)<sup>\*</sup>,
[Yan Xia](https://isshikihugh.github.io/scholar),
[Zhi Cen](https://scholar.google.com/citations?user=Xyy-uFMAAAAJ),
[Sida Peng](https://pengsida.net/)<sup>†</sup>,
[Zechen Hu](https://zju3dv.github.io/gvhmr),
[Hujun Bao](http://www.cad.zju.edu.cn/home/bao/),
[Ruizhen Hu](https://csse.szu.edu.cn/staff/ruizhenhu/),
[Xiaowei Zhou](https://xzhou.me/)  
> SIGGRAPH Asia 2024

<p align="center">
    <img src=docs/example_video/project_teaser.gif alt="animated" />
</p>

## News 🔥

- [2025-03-08] By default not using DPVO. We implemented a SimpleVO, which is more efficient and compatible with GVHMR.
- [2025-03-08] We added a new option `f_mm` to specify the focal length of the fullframe camera in mm.

## Setup

Please see [installation](docs/INSTALL.md) for details.

## Quick Start

### [<img src="https://i.imgur.com/QCojoJk.png" width="30"> Google Colab demo for GVHMR](https://colab.research.google.com/drive/1N9WSchizHv2bfQqkE9Wuiegw_OT7mtGj?usp=sharing)

### [<img src="https://s2.loli.net/2024/09/15/aw3rElfQAsOkNCn.png" width="20"> HuggingFace demo for GVHMR](https://huggingface.co/spaces/LittleFrog/GVHMR)

### Demo
Demo entries are provided in `tools/demo`. Use `-s` to skip visual odometry if you know the camera is static, otherwise the camera will be estimated by DPVO.
We also provide a script `demo_folder.py` to inference a entire folder.
```shell
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 -s
python tools/demo/demo_folder.py -f inputs/demo/folder_in -d outputs/demo/folder_out -s
```

### MotiForge headless backend

The versioned `hmr4d.backends.motiforge` entry point lets the sibling MotiForge
project orchestrate headless GVHMR inference without sharing Python environments
or vendoring this code. Setup, runtime checks and the portable artifact contract
are documented in [docs/MOTIFORGE_BACKEND.md](docs/MOTIFORGE_BACKEND.md). The
backend also exports the checkpoint's static-foot confidence, but this predicts
low joint speed, NOT physical contact. Custom static-support ground anchors are
disabled; the native network and native postprocessing are unchanged. For a fixed
camera, a separate low-frequency world/raw-in-camera height discrepancy correction
remains, rate-limited to 0.2 m/s. Shared camera/world jumps and squats cancel before
this discrepancy filter. This source correction can be disabled for ablation.

The selected prediction now also uses actual SMPL-X foot surfaces to enforce
the world-zero plane. The surface stage only lifts penetration risks, never pulls
floating feet down; the total vertical correction remains rate-limited. This
flat-ground policy changes only global height, preserves incam and pose/shape,
and rebuilds portable geometry afterwards. Residual penetration is a source error;
elevated feet require review, not rejection based on static-support probabilities.
Use `export-ground INPUT --output NEW_OUTPUT --asset-root DIR` to reprocess an
existing enabled-ground cache without video inference or overwriting its input.
It first undoes the cached custom correction, so old support anchors are not retained.
An explicitly confirmed grounded interval can additionally calibrate ONE fixed
height offset: add `--reference-start 0 --reference-duration 0.5
--assume-reference-grounded`. This is opt-in, not an automatic support detector;
it retains the camera reference and nonpenetration guard and records large
subsequent adjustments for review. Do not assume arbitrary clips start grounded.
Disable source ground stabilization for stairs/object-supported motions; inspect
the resulting source independently. See the backend document for limits and evidence.

For an explicitly declared flat-floor clip with at least one foot grounded in
every frame, the opt-in request option `assume_grounded: true` (or cached
`export-ground --assume-grounded`) projects the lower actual foot surface to zero.
It preserves articulation, shape, horizontal motion and incam, but removes real
flight and may transfer foot-estimation noise into root height. It is NOT enabled
by default or inferred from static-foot confidence. Requires source ground enabled;
cannot combine with a reference window. The v4 report records the assumption and
fast correction warnings. Original/v2/v3 caches may be re-exported into a new path,
after undoing the previous total correction; repeated v4 exports are rejected.

Existing portable predictions can be explicitly augmented with geometric foot
surfaces (`export-foot-surface`) or validated world-space body22 rotations and a
shaped neutral bind (`export-body-pose`). These CPU-only commands preserve the
original motion and do not rerun perception or change default video exports.
The body-pose export rejects dynamic shape and validates SMPL-X FK and serialized
rotation/bind compatibility within 0.1 mm before publishing a new artifact.
It remains an explicit A/B experiment, not default inference or a demonstrated
overall quality improvement: full-clip tests improved tracking/orientation but
increased foot sliding, with physical quality still C. The Feishu Adam Lite test
also increased self-collision; tennis G1 retained 16/313 floating frames. These
results do not establish that floating feet are solved; detailed measurements
are recorded in the backend documentation linked above.

### Reproduce
1. **Test**:
To reproduce the 3DPW, RICH, and EMDB results in a single run, use the following command:
    ```shell
    python tools/train.py global/task=gvhmr/test_3dpw_emdb_rich exp=gvhmr/mixed/mixed ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
    ```
    To test individual datasets, change `global/task` to `gvhmr/test_3dpw`, `gvhmr/test_rich`, or `gvhmr/test_emdb`.

2. **Train**:
To train the model, use the following command:
    ```shell
    # The gvhmr_siga24_release.ckpt is trained with 2x4090 for 420 epochs, note that different GPU settings may lead to different results.
    python tools/train.py exp=gvhmr/mixed/mixed
    ```
    During training, note that we do not employ post-processing as in the test script, so the global metrics results will differ (but should still be good for comparison with baseline methods).

# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

# Acknowledgement

We thank the authors of
[WHAM](https://github.com/yohanshin/WHAM),
[4D-Humans](https://github.com/shubham-goel/4D-Humans),
and [ViTPose-Pytorch](https://github.com/gpastal24/ViTPose-Pytorch) for their great works, without which our project/code would not be possible.
