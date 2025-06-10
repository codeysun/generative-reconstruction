import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.models.exporters.base import ExporterOutput
from threestudio.utils.misc import cleanup, get_device, load_module_weights
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.eval import compute_chamfer_distance, compute_volumetric_iou, compute_clip_similarity

import trimesh
import numpy as np

@threestudio.register("mvdream-strict-control-system")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        refinement: bool = False
        control_renderer_type: str = ""
        control_renderer: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.requires_grad_(False)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        # Load control geometry and renderer
        self.control_renderer = threestudio.find(self.cfg.control_renderer_type)(
            self.cfg.control_renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {"guidance."+k : v for (k,v) in self.guidance.state_dict().items()}
        checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Also render the control signal
        out = self.renderer(**batch)
        with torch.no_grad():
            out_control = self.control_renderer(**batch)
        return out, out_control

    def training_step(self, batch, batch_idx):
        out, control = self(batch)
        batch['idx'] = batch_idx

        guidance_out = self.guidance(
            out["comp_rgb"], self.prompt_utils, control=control, **batch
        )
        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if not self.cfg.refinement:
            if self.C(self.cfg.loss.lambda_orient) > 0:
                ### 1108
                if not "normal" not in out:
                    loss_orient = (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum() / (out["opacity"] > 0).sum()
                    self.log("train/loss_orient", loss_orient)
                    loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
            if hasattr(self.cfg.loss, "lambda_eikonal") and self.C(self.cfg.loss.lambda_eikonal) > 0:
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)
        else:
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness.half() * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out, control = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["depth"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": control["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_rgb" in control
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": control["mask"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "mask" in control
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out, control = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["depth"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": control["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_rgb" in control
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": control["mask"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ]
                if "mask" in control
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )

    def on_predict_epoch_end(self) -> None:
        if self.exporter.cfg.save_video:
            self.on_test_epoch_end()
        exporter_output: List[ExporterOutput] = self.exporter()
        for out in exporter_output:
            save_func_name = f"save_{out.save_type}"
            if not hasattr(self, save_func_name):
                raise ValueError(f"{save_func_name} not supported by the SaverMixin")
            save_func = getattr(self, save_func_name)
            save_func(f"it{self.true_global_step}-export/{out.save_name}", **out.params)
        if self.exporter.cfg.eval:
            control_mesh = self.control_renderer.control_mesh
            # export the control mesh for analysis
            control_mesh.export(os.path.join(self.get_save_dir(), "control.obj"))

            gen_mesh = exporter_output[0].params["mesh"]
            gen_mesh = trimesh.Trimesh(
                vertices=gen_mesh.v_pos.detach().cpu().numpy(),
                faces=gen_mesh.t_pos_idx.detach().cpu().numpy(),
            )

            gt_to_gen_chamfer, gen_to_gt_chamfer = compute_chamfer_distance(control_mesh, gen_mesh)

            volumetric_iou = compute_volumetric_iou(gen_mesh, control_mesh)

            print("Evaluation Metrics:")
            print(f"GT to gen chamfer distance: {gt_to_gen_chamfer:.5f}")
            print(f"Gen to GT chamfer distance: {gen_to_gt_chamfer:.5f}")
            print(f"Volumetric IoU: {volumetric_iou:.5f}")

            # TODO: get CLIP metrics
            # Get renders from 0, 90, 180, 270 angles
            # For simplicity, reuse the rendered image grid
            from PIL import Image
            from torchvision.transforms.functional import pil_to_tensor
            img_dir = f"it{self.true_global_step}-test"
            imgs = []
            for idx in [0, 30, 60, 90]:
                f = f"{idx}.png"
                im = Image.open(os.path.join(self.get_save_dir(), img_dir, f))
                width, height = im.size
                im = im.crop((0, 0, height, height))
                im = pil_to_tensor(im) / 255.0
                imgs.append(im)

            clip_score = compute_clip_similarity(imgs, self.prompt_processor.prompt)
            print(f"CLIP score: {clip_score:.5f}")

            data = {"gt_to_gen_chamfer": np.array(gt_to_gen_chamfer),
                    "gen_to_gt_chamfer": np.array(gen_to_gt_chamfer),
                    "volumetric_iou": np.array(volumetric_iou),
                    "clip_score": np.array(clip_score)}

            self.save_data(
                f"eval",
                data
            )

