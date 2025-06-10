# Locally Controlled 3D Generation

Generate 3D assets (left) with locally controlled geometry (right).

<p align="center">
    <img src = "https://github.com/user-attachments/assets/763f685a-528a-47d4-ab9f-2ceb52f12afd" width="80%">
    <img src = "https://github.com/user-attachments/assets/62e69579-3af4-4417-a3ba-4d7a0ba29f7f" width="80%">
    <img src = "https://github.com/user-attachments/assets/65bdf75d-e05c-40d8-be36-6681dcf1b01c" width="80%">
</p>

## Installation

This part is the same as original [MVDream-threestudio](https://github.com/bytedance/MVDream-threestudio). Skip it if you already have installed the environment.

### Install ControlDreamer

Install additional dependency:

```sh
cd threestudio/util/lib_shape_prior
python setup.py build_ext --inplace
```

ControlDreamer using multi-view ControlNet is provided in a different codebase. Install it by:

```sh
export PYTHONPATH=$PYTHONPATH:./extern/MVDream
pip install -e extern/MVDream
```

Further, to provide depth-conditioned MV-ControlNet, download from url or please put midas ckpt file on:
`ControlDreamer/extern/MVDream/mvdream/annotator/ckpts`

## Quickstart

Please download the model from [MV-ControlNet](https://drive.google.com/file/d/1hOdpfVTkKvUXGQStcmeFnzY0P_q4ZSod/view?usp=sharing) under `./extern/MVDream/mvdream/ckpt`

First, generate part segmentations using modified PartField code (the output should be a `.npy` file containing the mask).

Then, feed the source mesh + semantic part segment output as:

```sh
python launch.py --config configs/controldreamernerfstrict-sd21-shading.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$prompt" \
    system.control_renderer.file_path="$file_path" \
    system.control_renderer.masked_segments="$masked_segment" \
    system.control_renderer.local_control=True \
    system.guidance.control_scale=2.0
```

See example use in `generate.sh`

## Evaluation

To convert output into mesh and evaluate chamfer distance + CLIP score, run

```sh
python launch.py --config outputs/controldreamernerfstrict-sd21-shading/$output/configs/parsed.yaml \
    --export --gpu 0 \
    system.exporter.eval=True \
    resume=outputs/controldreamernerfstrict-sd21-shading/$output/ckpts/last.ckpt
```

## Credits

- This code is developed using [threestudio](https://github.com/threestudio-project/threestudio), [MVDream](https://github.com/bytedance/MVDream-threestudi), and [ImageDream](https://github.com/bytedance/ImageDream).

## Citing

If you find ControlDreamer helpful, please consider citing it:

```bibtex
@article{oh2023controldreamer,
  title={ControlDreamer: Stylized 3D Generation with Multi-View ControlNet},
  author={Oh, Yeongtak and Choi, Jooyoung and Kim, Yongsung and Park, Minjun and Shin, Chaehun and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2312.01129},
  year={2023}
}
```
