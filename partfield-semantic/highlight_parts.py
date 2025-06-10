import argparse
import numpy as np
import torch
import trimesh
import json
import cv2
import os
import math
from os.path import join
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
# import pointops
# import sys
# sys.path.append(os.path.abspath(".."))

# from pointcept.datasets.sampart3d_util import *
from partfield.utils import *

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

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

def cal_mapping_2d_3d(render_dir, mesh_path):
    """
    Maps pixels to mesh faces
    """
    mesh = load_mesh_util(mesh_path)


    # samples, face_index = trimesh.sample.sample_surface(mesh, 100000, sample_color=False) 
    # # samples, face_index, colors = sample_surface(mesh, 50000, sample_color=True)
    # face_index = torch.from_numpy(face_index).int()
    # face_index = torch.concat([face_index, torch.tensor([-1]).int()])

    meta_data = json.load(open(join(render_dir, "meta.json")))
    mesh_scale = meta_data["scaling_factor"]
    mesh_center_offset = meta_data["mesh_offset"]

    # object_org_coord = samples
    object_org_coord = mesh.vertices
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]])
    object_org_coord = np.dot(object_org_coord, rotation_matrix)
    object_org_coord = object_org_coord * mesh_scale + mesh_center_offset
    mesh.vertices = object_org_coord

    # Using proximity query is a bit slower but is more accurate
    pquery = trimesh.proximity.ProximityQuery(mesh)

    # object_org_coord = torch.from_numpy(object_org_coord).to("cuda").contiguous().float()
    # obj_offset = torch.tensor(object_org_coord.shape[0]).to("cuda")

    mapping_list = []
    camera_angle_x = meta_data['camera_angle_x']
    for i, c2w_opengl in enumerate(meta_data["transforms"]):
        c2w_opengl = np.array(c2w_opengl)
        rgb_path = join(render_dir, f"render_{i:04d}.webp")
        img = np.array(Image.open(rgb_path))
        if img.shape[-1] == 4:
            mask_img = img[..., 3] == 0
            img[mask_img] = [255, 255, 255, 255]
            img = img[..., :3]
            img = Image.fromarray(img.astype('uint8'))

        # Calculate mapping
        depth_path = join(render_dir, f"depth_{i:04d}.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth[..., 0]
        depth_valid = depth < 65500.0
        # depth_valid = torch.tensor(depth < 65500.0)

        org_points = gen_pcd(depth, c2w_opengl, camera_angle_x)
        org_points = torch.from_numpy(org_points)
        points_tensor = org_points.to("cuda").contiguous().float()
        offset = torch.tensor(points_tensor.shape[0]).to("cuda")

        _, distances, indices = pquery.on_surface(org_points)
        mapping = np.zeros((depth.shape[0], depth.shape[1]), dtype=int) - 1
        mask_dis = distances < 0.03
        indices[~mask_dis] = -1
        mapping[depth_valid] = indices
        # indices, distances = pointops.knn_query(1, object_org_coord, obj_offset, points_tensor, offset)
        # mapping = torch.zeros((depth.shape[0], depth.shape[1]), dtype=torch.int) - 1
        #
        # mask_dis = distances[..., 0] < 0.03
        # indices[~mask_dis] = -1
        # mapping[depth_valid] = face_index[indices.cpu().flatten()]

        mapping_list.append(mapping)
        # mapping_list.append(mapping.cpu().numpy())
    return np.stack(mapping_list)


def highlight_parts_in_multi_views(render_dir, mesh_path, results_dir, save_dir, uid, img_num=1):

    print(f"Processing {mesh_path}")
    obj_mapping = cal_mapping_2d_3d(render_dir, mesh_path)
    # scale_list = ["0.0", "0.5", "1.0", "1.5", "2.0"]
    NUM_CLUSTER = 20
    scale_list = [f"{i+1:02d}" for i in range(NUM_CLUSTER)]
    for scale in scale_list:
        print(scale)
        # ins_pred = np.load(join(results_dir, f"mesh_{scale}.npy"))
        ins_pred = np.load(join(results_dir, f"{uid}_0_{scale}.npy")) # label for each mesh face [F,]

        unique_labels = np.unique(ins_pred)
        label_to_id = {k:v for v, k in enumerate(unique_labels)}
        lti_vectorized = np.vectorize(label_to_id.get)


        # Get the number of images and the number of classes
        num_images = obj_mapping.shape[0]
        # num_classes = np.max(ins_pred) + 1
        num_classes = len(unique_labels) + 1

        # Initialize an array to store the pixel count for each class in each image
        pixel_count = np.zeros((num_images, num_classes), dtype=np.int32)
        # Iterate over each image
        for i in range(num_images):
            # Get the group numbers for each pixel in the image
            valid_areas = obj_mapping[i] != -1
            groups = lti_vectorized(ins_pred[obj_mapping[i][valid_areas]])
            # Count the number of pixels for each group
            pixel_count[i], _ = np.histogram(groups, bins=np.arange(num_classes + 1) - 0.5)
        # Find the top 1 images for each class
        top_image_ids = np.argsort(-pixel_count, axis=0)[:img_num]
        # top_image_ids = np.stack([top_image_ids[0, :], top_image_ids[2, :], top_image_ids[4, :]])

        save_path = join(save_dir, scale)
        os.makedirs(save_path, exist_ok=True)
        # for part_id in range(ins_pred.max()+1):
        for part_id, label in zip(range(num_classes), unique_labels):
            img_id_list = top_image_ids[:, part_id]
            for topj, img_id in enumerate(img_id_list):
                image = np.array(Image.open(join(render_dir, f"render_{img_id:04d}.webp")))
                if image.shape[-1] == 4:
                    mask_img = image[..., 3] == 0
                    image[mask_img] = [255, 255, 255, 255]
                    image = image[..., :3]
                image = Image.fromarray(image)
                valid_areas = obj_mapping[img_id] != -1
                mask = np.zeros_like(obj_mapping[img_id], dtype=bool)
                mask[valid_areas] = (ins_pred[obj_mapping[img_id][valid_areas]] == label)

                # Find the edges of the mask
                # edges = ndi.binary_dilation(mask, iterations=1) ^ mask

                # Draw a red circle around the edges
                draw = ImageDraw.Draw(image)
                # for y, x in np.argwhere(edges):
                for y, x in np.argwhere(mask):
                    # draw.ellipse([x-2, y-2, x+2, y+2], fill='red')
                    draw.point((x, y), fill='red')
                image.save(join(save_path, f"{label}-{topj}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_dir', default= "renders/00200996b8f34f55a2dd2f44d316d107", type=str)
    parser.add_argument('--mesh_path', default= "/home/codeysun/git/data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh/00200996b8f34f55a2dd2f44d316d107.glb", type=str)
    parser.add_argument('--results_dir', default= "exp_results/clustering/partobjtiny/cluster_out", type=str)

    args = parser.parse_args()

    uid = "00200996b8f34f55a2dd2f44d316d107"
    render_dir = args.render_dir
    mesh_path = args.mesh_path
    results_dir = args.results_dir
    save_dir = os.path.join(args.render_dir, "highlights")
    # render_dir = "data_root/knight"
    # mesh_path = "mesh_root/knight.glb"
    # results_dir = "exp/sampart3d/knight/results/last"
    # save_dir = "highlights/knight"
    highlight_parts_in_multi_views(render_dir, mesh_path, results_dir, save_dir, uid)
