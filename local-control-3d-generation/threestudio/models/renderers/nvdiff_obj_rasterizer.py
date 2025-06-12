from dataclasses import dataclass, field

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.typing import *
from threestudio.models.mesh import Mesh
from threestudio.utils.ops import dot

import os
import trimesh
import numpy as np

def normalize_mesh(mesh):
    """
    As per MVDream, normalize a mesh by centering it within a bounding box
    between [-0.5, 0.5]
    """
    # Get the vertices of the mesh
    vertices = mesh.vertices  # Shape: (N, 3)

    # Calculate the center of the mesh
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2  # Center point of the bounding box

    # Calculate the scale to fit within the target bounding box [-0.5, 0.5]
    size = mesh.bounds[1] - mesh.bounds[0]  # Size of the current bounding box
    max_dimension = np.max(size)  # Find the maximum dimension
    # scale_factor = 1 / max_dimension  # Scale to fit [-0.5, 0.5]
    # scale_factor = 2 / max_dimension  # Scale to fit [-1, 1]
    scale_factor = 1.8 / max_dimension  # Scale to fit [-0.9, 0.9]
    # scale_factor = 1

    # Normalize the vertices: translate the mesh to center and then scale
    normalized_vertices = (vertices - center) * scale_factor

    # Create a new trimesh object with normalized vertices
    mesh.vertices = normalized_vertices

    return

def color_mesh_by_mask(mesh, mask, default_color):
    # Create an array of colors for all faces (initially black)
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)

    # Set the colors of the masked faces
    face_colors[~mask.astype(bool)] = [default_color, default_color, default_color, 255]
    face_colors[mask.astype(bool)] = [255, 255, 255, 255]

    return face_colors
    # Apply the face colors to the mesh
    if mesh.visual.face_colors is None:
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=face_colors)
    else:
        mesh.visual.face_colors = face_colors # in place replacement

def create_submesh(mesh, mask):
    """Create a submesh by masking faces."""
    faces = mesh.faces
    selected_faces = faces[mask.astype(bool)]
    unique_vertex_indices = np.unique(selected_faces)
    vertex_mapping = {old_index: new_index for new_index, old_index in enumerate(unique_vertex_indices)}
    remapped_faces = np.vectorize(vertex_mapping.get)(selected_faces)
    selected_vertices = mesh.vertices[unique_vertex_indices]
    new_mesh = trimesh.Trimesh(vertices=selected_vertices, faces=remapped_faces)
    return new_mesh

@threestudio.register("nvdiff-obj-rasterizer")
class NVDiffObjRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        file_path: str = ""
        masked_segments: str = ""
        context_type: str = "gl"
        default_control: int = 0
        local_control: bool = True

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)

        device = get_device()
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, device)

        self.trimesh = trimesh.load(self.cfg.file_path, force='mesh', process=False)
        faces = self.trimesh.faces
        masked_faces = np.load(self.cfg.masked_segments)
        # Set the face colors
        face_colors = color_mesh_by_mask(self.trimesh, masked_faces, self.cfg.default_control)

        # Set object pose to align with MVDream
        # opengl_to_blender = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        obj_pose = np.array([
            [0, 0, -1, 0],
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

        # obj_pose = np.array([
        #     [0, 0, 1, 0], 
        #     [1, 0, 0, 0], 
        #     [0, 1, 0, 0], 
        #     [0, 0, 0, 1]
        # ])
        self.trimesh = self.trimesh.apply_transform(obj_pose)

        # Normalize object to be centered in [-0.5, 0.5]
        normalize_mesh(self.trimesh)

        # Create control mesh. Used for chamfer distance evaluation.
        self.control_trimesh = create_submesh(self.trimesh, masked_faces)

        # Texture stuff
        self.texture = None
        self.v_colors = None
        if isinstance(self.trimesh.visual, trimesh.visual.texture.TextureVisuals):
            self.uv = torch.from_numpy(np.array(self.trimesh.visual.uv)).float().to(device)
            self.uv[:, 1] = 1.0 - self.uv[:, 1] # Flip UV. nvdiffrast (0,0) is at bottom-left instead of top-left
            tex = self.trimesh.visual.material.baseColorTexture
            self.texture = torch.from_numpy(np.array(self.trimesh.visual.material.baseColorTexture)).float().to(device) / 255.0
        if isinstance(self.trimesh.visual, trimesh.visual.color.ColorVisuals):
            v_colors = torch.from_numpy(self.trimesh.visual.vertex_colors).int().to(device)
            self.v_colors = v_colors[:, :3].float() / 255.0



        self.v_pos, self.t_pos_idx, self.f_nrm, f_colors = (
            torch.from_numpy(self.trimesh.vertices).float().to(device),
            torch.from_numpy(self.trimesh.faces).long().to(device),
            torch.from_numpy(self.trimesh.face_normals).float().to(device),
            torch.from_numpy(face_colors).int().to(device),
        )  # transform back to torch tensor on CUDA
        self.mesh = Mesh(v_pos=self.v_pos, t_pos_idx=self.t_pos_idx)
        self.f_colors = f_colors[:, :3].float() / 255.0

        # Material options
        self.ambient_light_color = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32).to(device)
        self.diffuse_light_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(device)

    def render_color(
        self,
        v_pos,
        t_pos_idx,
        f_nrm,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]

        height = 256
        width = 256

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, t_pos_idx)

        out = {"opacity": mask_aa}

        gb_normal = torch.zeros(batch_size, height, width, 3).to(rast)
        gb_normal[mask.squeeze(dim=3)] = f_nrm[rast[mask.squeeze(dim=3)][:, 3].int() - 1]
        out.update({"comp_normal": gb_normal})  # in [0, 1]

        selector = mask[..., 0]

        gb_pos, _ = self.ctx.interpolate_one(v_pos, rast, t_pos_idx)
        gb_viewdirs = F.normalize(
            gb_pos - camera_positions[:, None, None, :], dim=-1
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
        textureless_color = diffuse_light + self.ambient_light_color

        rgb_fg = textureless_color

        # Rendering texture (if available)
        if self.texture is not None:
            texc, _ = self.ctx.interpolate_one(self.uv, rast, t_pos_idx)
            albedo = self.ctx.texture(self.texture, texc)
            albedo = albedo[selector][:, :3]
            rgb_fg = rgb_fg * albedo.clamp(0.0, 1.0)
        elif self.v_colors is not None:
            gb_colors, _ = self.ctx.interpolate_one(self.v_colors, rast, t_pos_idx)
            albedo = gb_colors[selector][:, :3]
            rgb_fg = rgb_fg * albedo.clamp(0.0, 1.0)



        gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
        gb_rgb_fg[selector] = rgb_fg

        gb_rgb_bg = torch.ones(batch_size, height, width, 3).to(rgb_fg)
        gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
        gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, t_pos_idx)

        out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg,
                    "depth": rast[..., 2].unsqueeze(-1)})

        return out

    def render_mask(
        self,
        v_pos,
        t_pos_idx,
        f_colors,
        mvp_mtx: Float[Tensor, "B 4 4"],
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]

        height = 256
        width = 256

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, t_pos_idx, (height, width))

        # control_mask, _ = self.ctx.interpolate_one(v_colors, rast, t_pos_idx)
        mask = rast[..., 3:] > 0
        control_mask = torch.zeros(batch_size, height, width, 3).to(rast)
        control_mask[mask.squeeze(dim=3)] = f_colors[rast[mask.squeeze(dim=3)][:, 3].int() - 1]

        control_mask_aa = self.ctx.antialias(control_mask, rast, v_pos_clip, t_pos_idx)

        out = {"mask": control_mask_aa[..., 0].unsqueeze(-1)}
        return out


    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        out = self.render_color(self.v_pos, self.t_pos_idx, self.f_nrm, mvp_mtx, camera_positions, light_positions)
        if self.cfg.local_control:
            out.update(self.render_mask(self.v_pos, self.t_pos_idx, self.f_colors, mvp_mtx))

        return out

    @property
    def control_mesh(self):
        return self.control_trimesh
