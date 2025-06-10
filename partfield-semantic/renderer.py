from dataclasses import dataclass, field
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor
from typing import Dict, Any, Union

import sys
import os
import torch
import torch.nn.functional as F

import os
import trimesh
import numpy as np
import nvdiffrast.torch as dr

def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)

def get_projection_matrix(
    fovy: Float[Tensor, "B"], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "B 4 4"]:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32).to(fovy)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    # proj_mtx[:, 1, 1] = -1.0 / torch.tan(
    #     fovy / 2.0
    # )  # add a negative sign here as the y axis is flipped in nvdiffrast output
    proj_mtx[:, 2, 2] = -(far + near) / (far - near)
    proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
    proj_mtx[:, 3, 2] = -1.0
    return proj_mtx

def get_mvp_matrix(
    c2w: Float[Tensor, "B 4 4"], proj_mtx: Float[Tensor, "B 4 4"]
) -> Float[Tensor, "B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx

def vertex_transform(
    verts: Float[Tensor, "Nv 3"], mvp_mtx: Float[Tensor, "B 4 4"]
) -> Float[Tensor, "B Nv 4"]:
    verts_homo = torch.cat(
        [verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1
    )
    return torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))

def color_mesh_by_mask(mesh, mask, default_color):
    # Convert mask to binary array
    n_faces = mesh.faces.shape[0]
    mask_b = np.zeros(n_faces)
    mask_b[mask] = 1

    # Create an array of colors for all faces (initially black)
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)

    # Set the colors of the masked faces
    face_colors[~mask_b.astype(bool)] = [default_color, default_color, default_color, 255]
    face_colors[mask_b.astype(bool)] = [255, 255, 255, 255]

    return face_colors

class NVDiffRasterizer():
    def __init__(self,
                 root_mesh,
                 color,
                 img_size,
                 vertex_coloring=True):
        device = "cuda"
        self.device = device
        self.ctx = dr.RasterizeGLContext(device=device)
        self.vertex_coloring = vertex_coloring

        self.root_mesh = root_mesh
        face_indices = np.indices(self.root_mesh.faces.shape[:-1]).squeeze()
        face_colors = color_mesh_by_mask(self.root_mesh, face_indices, 0)


        self.v_pos, self.t_pos_idx, self.f_nrm = (
            torch.from_numpy(self.root_mesh.vertices).float().to(device),
            torch.from_numpy(self.root_mesh.faces).long().to(device),
            torch.from_numpy(self.root_mesh.face_normals).float().to(device),
        )  # transform back to torch tensor on CUDA

        if vertex_coloring:
            v_colors = torch.from_numpy(self.root_mesh.visual.vertex_colors).int().to(device)
            self.v_colors = v_colors[:, :3].float() / 255.0

        f_colors = torch.from_numpy(face_colors).int().to(device)
        self.f_colors = f_colors[:, :3].float() / 255.0

        self.ambient_light_color = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32).to(device)
        self.diffuse_light_color = torch.tensor(color, dtype=torch.float32).to(device)

        self.light_positions = torch.tensor([1., 1., 1.]).to(device)

        fovy_deg = 50.
        self.fovy: Float[Tensor, "B"] = torch.tensor([fovy_deg * np.pi / 180]).to(device)
        self.proj: Float[Tensor, "B 4 4"] = get_projection_matrix(self.fovy, 1, 0.1, 1000.0)

        self.height = img_size
        self.width = img_size

    def set_mesh(self, face_indices):
        # if self.vertex_coloring:
        #     v_colors = torch.from_numpy(self.root_mesh.visual.vertex_colors).int().to(self.device)
        #     self.v_colors = v_colors[:, :3].float() / 255.0

        face_colors = color_mesh_by_mask(self.root_mesh, face_indices, 0)
        f_colors = torch.from_numpy(face_colors).int().to(self.device)
        self.f_colors = f_colors[:, :3].float() / 255.0


    def __call__(
        self,
        c2w: Union[
                Float[Tensor, "B 4 4"],
                Float[np.ndarray, "B 4 4"]
            ],
    ) -> Dict[str, Any]:
        # c2w = self.c2w
        if isinstance(c2w, np.ndarray):
            c2w = torch.tensor(c2w, dtype=torch.float).to(self.device)
        batch_size = c2w.shape[0]

        height = self.height
        width = self.width

        # mvp_mtx = self.mvp_mtx
        mvp_mtx = get_mvp_matrix(c2w, self.proj)
        camera_positions: Float[Tensor, "B 3"] = c2w[:, :3, 3]
        # light_positions = camera_positions
        light_positions = self.light_positions.expand(batch_size, -1)

        v_pos_clip = vertex_transform(self.v_pos, mvp_mtx)

        rast, _ = dr.rasterize(self.ctx, v_pos_clip.float(), self.t_pos_idx.int(), (height, width), grad_db=True)
        mask = rast[..., 3:] > 0
        mask_aa = dr.antialias(mask.float(), rast, v_pos_clip.float(), self.t_pos_idx.int())

        out = {"opacity": mask_aa}

        gb_normal = torch.zeros(batch_size, height, width, 3).to(rast)
        gb_normal[mask.squeeze(dim=3)] = self.f_nrm[rast[mask.squeeze(dim=3)][:, 3].int() - 1]
        out.update({"comp_normal": gb_normal})  # in [0, 1]

        selector = mask[..., 0]

        gb_pos, _ = dr.interpolate(
            self.v_pos.float(), rast, self.t_pos_idx.int(), rast_db=None, diff_attrs=None
        )
        gb_light_positions = light_positions[:, None, None, :].expand(
            -1, height, width, -1
        )

        positions = gb_pos[selector]
        shading_normal = gb_normal[selector]

        light_directions: Float[Tensor, "B ... 3"] = F.normalize(
            gb_light_positions[selector] - positions, dim=-1
        )
        diffuse_light: Float[Tensor, "B ... 3"] = (
            torch.abs(dot(shading_normal, light_directions)) * self.diffuse_light_color
        )
        textureless_color = torch.clamp(diffuse_light + self.ambient_light_color, max=1.0)

        rgb_fg = textureless_color

        if self.vertex_coloring:
            gb_color, _ = dr.interpolate(
                self.v_colors.float(), rast, self.t_pos_idx.int(), rast_db=None, diff_attrs=None
            )
            albedo = gb_color[selector]
            rgb_fg = rgb_fg * albedo.clamp(0.0, 1.0)


        gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
        gb_rgb_fg[selector] = rgb_fg

        gb_rgb_bg = torch.ones(batch_size, height, width, 3).to(rgb_fg)
        gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
        gb_rgb_aa = dr.antialias(gb_rgb.float(), rast, v_pos_clip.float(), self.t_pos_idx.int())

        # TODO: We want face colors, not vertex colors
        # control_mask, _ = dr.interpolate(
        #     self.v_colors.float(), rast, self.t_pos_idx.int(), rast_db=None, diff_attrs=None
        # )
        mask = rast[..., 3:] > 0
        control_mask = torch.zeros(batch_size, height, width, 3).to(rast)
        control_mask[mask.squeeze(dim=3)] = self.f_colors[rast[mask.squeeze(dim=3)][:, 3].int() - 1]

        control_mask_aa = dr.antialias(control_mask.float(), rast, v_pos_clip.float(), self.t_pos_idx.int())

        out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg, 
                    "mask": control_mask_aa[..., 0].unsqueeze(-1), "depth": rast[..., 2].unsqueeze(-1)})

        return out

