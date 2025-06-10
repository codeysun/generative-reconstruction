import trimesh
import numpy as np
import math

from subprocess import call

def load_mesh_util(input_fname) -> trimesh.Trimesh:
    mesh = trimesh.load(input_fname, force='mesh', process=False)
    return mesh

def run_script(cmd):
    ret = call(cmd, shell=True)
    if ret != 0:
        raise Exception(f"Failed to run {cmd}")

def render_mesh(rast, c2w):
    # Render using nvdiff rast
    # Output both image rendering + GT segmentation

    control_dict = rast(c2w)

    rgb = control_dict["comp_rgb"].squeeze().detach().cpu().numpy() # H W C
    mask = control_dict["mask"].squeeze().detach().cpu().numpy() # H W

    return rgb, mask

def sample_camera_pos(n_samples, radius, min_elev, max_elev) -> list[np.ndarray]:
    """
    Generate a list of camera positions covering a sphere in a spiral
    """
    min_elev_rad = np.radians(min_elev)
    max_elev_rad = np.radians(max_elev)

    elevations = np.linspace(min_elev_rad, max_elev_rad, n_samples)
    azimuths = np.linspace(0, 4 * np.pi, n_samples)

    positions: list = []

    for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
        x = radius * np.cos(elev) * np.cos(azim)
        y = radius * np.sin(elev)
        z = radius * np.cos(elev) * np.sin(azim)
        positions.append(np.array([x, y, z]))

    return positions

def convert_opengl_to_blender(camera_matrix):
    flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return np.dot(flip_yz, camera_matrix)

def gen_camera_traj(n_samples, radius=3, min_elev=-90, max_elev=90, blender=False) -> list[np.ndarray]:
    positions: list[np.ndarray] = sample_camera_pos(n_samples, radius, min_elev, max_elev) # [B, 3]

    center = np.array([0, 0, 0])
    up_world = np.array([0, 1, 0])

    c2ws: list = []
    for cam_pos in positions:
        lookat = (center - cam_pos)
        lookat = lookat / np.linalg.norm(lookat)
        right = np.cross(lookat, up_world)
        right = right / np.linalg.norm(right)
        up = np.cross(right, lookat)
        up = up / np.linalg.norm(up)
        c2w3x4 = np.concatenate([np.stack([right, up, -lookat], axis=1), cam_pos[:, np.newaxis]], axis=1)
        c2w = np.concatenate([c2w3x4, np.zeros_like(c2w3x4[:1, :])], axis=0)
        c2w[3, 3] = 1.0

        if blender:
            c2w = convert_opengl_to_blender(c2w)

        c2ws.append(c2w)

    return c2ws

def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing="xy",
    )
    directions = np.stack(
        [(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], -1
    ) 

    return directions

def gen_pcd(depth, c2w_opengl, camera_angle_x):

    h, w = depth.shape
    
    depth_valid = depth < 65500.0
    depth = depth[depth_valid]
    focal = (
        0.5 * w / math.tan(0.5 * camera_angle_x)
    )  # scaled focal length
    ray_directions = get_ray_directions(w, h, focal, focal, w // 2, h // 2)
    points_c = ray_directions[depth_valid] * depth[:, None]
    points_c_homo = np.concatenate(
        [points_c, np.ones_like(points_c[..., :1])], axis=-1
    )
    org_points = (points_c_homo @ c2w_opengl.T)[..., :3]

    return org_points
