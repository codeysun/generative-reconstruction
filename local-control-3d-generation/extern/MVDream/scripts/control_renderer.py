from typing import Any, Dict
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt
from torch import Tensor
from PIL import Image

import argparse
import trimesh
import numpy as np
import torch
import sys
import os
import imageio
import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr

def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)

# def get_camera_nvdiffrast(n_views, eval_elevation_deg, eval_camera_distance):
def get_camera_nvdiffrast(n_views, eval_elevation_deg, fovy):
    print(f"fovy: {fovy}")
    focal_ndc = 1. / torch.tan(fovy / 2.0)
    print(f"focal: {focal_ndc}")
    # eval_camera_distance = 0.5 * focal_ndc[0] * 1.1 # from paper
    eval_camera_distance = focal_ndc[0] * 1.1 # from dataloader
    print(eval_camera_distance)
    # eval_camera_distance = 3.0

    azimuth_deg = torch.linspace(0, 360.0, n_views+1)[:n_views]
    elevation_deg = torch.full_like(
        azimuth_deg, eval_elevation_deg
    )
    camera_distances = torch.full_like(
        elevation_deg, eval_camera_distance
    )

    elevation = elevation_deg * np.pi / 180
    azimuth = azimuth_deg * np.pi / 180

    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    # default scene center at origin
    center = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        None, :
    ].repeat(1, 1)

    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0
    return c2w

def get_projection_matrix(
    fovy: Float[Tensor, "B"], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "B 4 4"]:
    batch_size = fovy.shape[0]
    proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32).to(fovy)
    proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
    proj_mtx[:, 1, 1] = -1.0 / torch.tan(
        fovy / 2.0
    )  # add a negative sign here as the y axis is flipped in nvdiffrast output
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

def create_bbox(bounds):
    min_corner = bounds[0].copy()
    max_corner = bounds[1].copy()

     # Define the vertices of the cube
    v0 = np.array(min_corner)
    v1 = np.array([min_corner[0], min_corner[1], max_corner[2]])  # Front bottom-left
    v2 = np.array([min_corner[0], max_corner[1], min_corner[2]])  # Back bottom-left
    v3 = np.array([min_corner[0], max_corner[1], max_corner[2]])  # Front top-left
    v4 = np.array([max_corner[0], min_corner[1], min_corner[2]])  # Front bottom-right
    v5 = np.array([max_corner[0], min_corner[1], max_corner[2]])  # Back bottom-right
    v6 = np.array([max_corner[0], max_corner[1], min_corner[2]])  # Back top-right
    v7 = np.array(max_corner)                                        # Front top-right

    # Note that nvdiffrast does not support backface culling natively

    # Create an array of vertices
    vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7])

    # Define the faces of the cube
    faces = np.array([[0, 2, 1],
                      [2, 3, 1],
                      [0, 5, 4],
                      [0, 1, 5],
                      [0, 6, 2],
                      [4, 6, 0],
                      [4, 6, 7],
                      [7, 5, 4],
                      [1, 7, 3],
                      [1, 5, 7],
                      [3, 7, 6],
                      [6, 2, 3]])

    # Create trimesh
    bbox_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    color = [255, 255, 255, 255]
    v_colors = np.array([color] * bbox_mesh.vertices.shape[0])
    bbox_mesh.visual.vertex_colors = v_colors

    return bbox_mesh

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

class ControlRasterizer():
    def __init__(
        self,
        file_path,
        masked_segments,
        device,
        fovy,
        default_control=0,
    ):
        self.ctx = dr.RasterizeGLContext(device=device)

        # obj_meshes = []
        # control_meshes = []
        # self.seg_node_map = {}
        # for file in os.listdir(file_path):
        #     if file.endswith('.obj'):
        #     # if file in masked_segments:
        #         obj_file_path = os.path.join(file_path, file)
        #         mesh = trimesh.load(obj_file_path)
        #         if mesh.is_empty:
        #             print(f"Failed to load OBJ file: {obj_file_path}")
        #             continue
        #         obj_meshes.append(mesh)

        #         if masked_segments is None or file in masked_segments:
        #             color = [255, 255, 255, 255]
        #             control_meshes.append(mesh)
        #         else:
        #             color = [default_control, default_control, default_control, 255]
        #         v_colors = np.array([color] * mesh.vertices.shape[0])
        #         mesh.visual.vertex_colors = v_colors


        # self.trimesh = trimesh.util.concatenate(obj_meshes)
        # self.control_trimesh = trimesh.util.concatenate(control_meshes)

        # # Scale object
        # obj_scale = np.eye(4)
        # obj_scale[:3, :3] *= 0.6
        # self.trimesh = self.trimesh.apply_transform(obj_scale)
        # self.control_trimesh = self.control_trimesh.apply_transform(obj_scale)
        self.trimesh = trimesh.load(file_path, force='mesh', process=False)
        faces = self.trimesh.faces
        masked_faces = np.load(masked_segments)
        # Set the face colors
        face_colors = color_mesh_by_mask(self.trimesh, masked_faces, default_control)

        # Set object pose to align with MVDream
        obj_pose = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

        self.trimesh = self.trimesh.apply_transform(obj_pose)
        # self.control_trimesh = self.control_trimesh.apply_transform(obj_pose)

        # Normalize object to be centered in [-0.5, 0.5]
        normalize_mesh(self.trimesh)
        # normalize_mesh(self.control_trimesh)

        # Create control mesh. Used for chamfer distance evaluation.
        self.control_trimesh = create_submesh(self.trimesh, masked_faces)

        # TODO: texture stuff
        self.uv = torch.from_numpy(np.array(self.trimesh.visual.uv)).float().to(device)
        self.uv[:, 1] = 1.0 - self.uv[:, 1] # Flip UV. nvdiffrast (0,0) is at bottom-left instead of top-left
        tex = self.trimesh.visual.material.baseColorTexture
        self.texture = torch.from_numpy(np.array(tex)).float().to(device) / 255.0


        self.v_pos, self.t_pos_idx, self.f_nrm, f_colors = (
            torch.from_numpy(self.trimesh.vertices).float().to(device),
            torch.from_numpy(self.trimesh.faces).long().to(device),
            torch.from_numpy(self.trimesh.face_normals).float().to(device),
            torch.from_numpy(face_colors).int().to(device),
        )  # transform back to torch tensor on CUDA
        self.f_colors = f_colors[:, :3].float() / 255.0

        # # Get bounding boxes
        # bounds = self.trimesh.bounds
        # self.bbox_trimesh = create_bbox(bounds)
        # self.bbox_v_pos, self.bbox_t_pos_idx, self.bbox_f_nrm = (
        #     torch.from_numpy(self.bbox_trimesh.vertices).float().to(device),
        #     torch.from_numpy(self.bbox_trimesh.faces).long().to(device),
        #     torch.from_numpy(self.bbox_trimesh.face_normals).float().to(device),
        # )  # transform back to torch tensor on CUDA

        # Material options
        self.ambient_light_color = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32).to(device)
        self.diffuse_light_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).to(device)

        # Camera
        # fovy_deg: Float[Tensor, "B"] = torch.tensor([40.]).to(device)
        # fovy = fovy_deg * np.pi / 180
        self.proj: Float[Tensor, "B 4 4"] = get_projection_matrix(fovy, 1, 0.1, 1000.0)

    def render_color(
        self,
        c2w: Float[Tensor, "B 4 4"],
        v_pos,
        t_pos_idx,
        f_nrm,
        background=None,
        diffuse_light_color=None
    ):
        batch_size = c2w.shape[0]

        height = 256
        width = 256

        mvp_mtx = get_mvp_matrix(c2w, self.proj)
        camera_positions: Float[Tensor, "B 3"] = c2w[:, :3, 3]
        light_positions = camera_positions

        v_pos_clip = vertex_transform(v_pos, mvp_mtx)

        rast, _ = dr.rasterize(self.ctx, v_pos_clip.float(), t_pos_idx.int(), (height, width), grad_db=True)
        mask = rast[..., 3:] > 0
        mask_aa = dr.antialias(mask.float(), rast, v_pos_clip.float(), t_pos_idx.int())

        out = {"opacity": mask_aa}

        gb_normal = torch.zeros(batch_size, height, width, 3).to(rast)
        gb_normal[mask.squeeze(dim=3)] = f_nrm[rast[mask.squeeze(dim=3)][:, 3].int() - 1]
        out.update({"comp_normal": gb_normal})  # in [0, 1]

        selector = mask[..., 0]

        gb_pos, _ = dr.interpolate(
            v_pos.float(), rast, t_pos_idx.int(), rast_db=None, diff_attrs=None
        )
        gb_light_positions = light_positions[:, None, None, :].expand(
            -1, height, width, -1
        )

        positions = gb_pos[selector]
        shading_normal = gb_normal[selector]

        # TODO: try rendering texture (if available)
        texc, _ = dr.interpolate(
            self.uv.float(), rast, t_pos_idx.int(), rast_db=None, diff_attrs=None
        )
        albedo = dr.texture(self.texture[None, ...], texc)
        albedo = albedo[selector][:, :3]

        light_directions: Float[Tensor, "B ... 3"] = F.normalize(
            gb_light_positions[selector] - positions, dim=-1
        )
        if diffuse_light_color is None:
            diffuse_light_color = self.diffuse_light_color
        diffuse_light: Float[Tensor, "B ... 3"] = (
            torch.abs(dot(shading_normal, light_directions)) * diffuse_light_color
        )
        textureless_color = diffuse_light + self.ambient_light_color

        # rgb_fg = textureless_color
        rgb_fg = albedo.clamp(0.0, 1.0) * textureless_color
        # rgb_fg = albedo.clamp(0.0, 1.0)

        gb_rgb_fg = torch.zeros(batch_size, height, width, 3).to(rgb_fg)
        gb_rgb_fg[selector] = rgb_fg

        if background is None:
            gb_rgb_bg = torch.ones(batch_size, height, width, 3).to(rgb_fg)
        else:
            gb_rgb_bg = background
        gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
        gb_rgb_aa = dr.antialias(gb_rgb.float(), rast, v_pos_clip.float(), t_pos_idx.int())

        out.update({"comp_rgb": gb_rgb_aa, "comp_rgb_bg": gb_rgb_bg,
                    "depth": rast[..., 2].unsqueeze(-1)})

        return out

    def render_mask(
        self,
        c2w: Float[Tensor, "B 4 4"],
        v_pos,
        t_pos_idx,
        f_colors
    ):
        batch_size = c2w.shape[0]

        height = 256
        width = 256

        mvp_mtx = get_mvp_matrix(c2w, self.proj)
        camera_positions: Float[Tensor, "B 3"] = c2w[:, :3, 3]

        v_pos_clip = vertex_transform(v_pos, mvp_mtx)

        rast, _ = dr.rasterize(self.ctx, v_pos_clip.float(), t_pos_idx.int(), (height, width), grad_db=True)

        # control_mask, _ = dr.interpolate(
        #     v_colors.float(), rast, t_pos_idx.int(), rast_db=None, diff_attrs=None
        # )
        mask = rast[..., 3:] > 0
        control_mask = torch.zeros(batch_size, height, width, 3).to(rast)
        control_mask[mask.squeeze(dim=3)] = f_colors[rast[mask.squeeze(dim=3)][:, 3].int() - 1]

        control_mask_aa = dr.antialias(control_mask.float(), rast, v_pos_clip.float(), self.t_pos_idx.int())

        out = {"mask": control_mask_aa[..., 0].unsqueeze(-1)}

        return out

    def __call__(
        self,
        c2w: Float[Tensor, "B 4 4"],
    ) -> Dict[str, Any]:
        # out_bbox = self.render_color(c2w, self.bbox_v_pos, self.bbox_t_pos_idx, self.bbox_f_nrm)

        # out = self.render_color(c2w, self.v_pos, self.t_pos_idx, self.f_nrm, background=out["comp_rgb"])
        out = self.render_color(c2w, self.v_pos, self.t_pos_idx, self.f_nrm)
        out.update(self.render_mask(c2w, self.v_pos, self.t_pos_idx, self.f_colors))

        # out.update({"bbox_mask": out_bbox["opacity"]})

        return out

if __name__ == "__main__":
    device = "cuda"
    image_size = 256

    fovy_deg: Float[Tensor, "B"] = torch.tensor([40.]).to(device)
    fovy = fovy_deg * np.pi / 180

    # c2ws = get_camera_nvdiffrast(4, 0.0, 3.0).to(device)
    c2ws = get_camera_nvdiffrast(4, 0.0, fovy).to(device)
    control_obj_path = "/home/codeysun/git/data/PartNet/data_v0/8677/objs"
    control_masked_segments = ["original-3.obj"]
    # control_obj_path = "/home/codeysun/git/data/PartNet/data_v0/14102/objs"
    # control_masked_segments = ["new-0.obj", "new-1.obj"]
    default_control = 0

    control_rasterizer = ControlRasterizer(control_obj_path, control_masked_segments, device, fovy, default_control)
    control_dict = control_rasterizer(c2ws)
    control_rgb = control_dict["comp_rgb"]

    B, H, W, C = control_rgb.shape
    control_rgb = control_rgb.permute(0, 2, 1, 3) # B W H C
    concatenated_image = control_rgb.reshape(W*B, H, C)
    concatenated_image = concatenated_image.permute(1, 0, 2)

    # Convert to NumPy array and scale to [0, 255] if necessary
    concatenated_image_np = (concatenated_image.cpu().detach().numpy() * 255).astype(np.uint8)

    # Save the image using PIL
    pil_image = Image.fromarray(concatenated_image_np).save(f"test.png")

    # Export mesh
    control_rasterizer.trimesh.export('new_mesh.obj', file_type='obj')
    print("Done.")
