"""
Given tree and canonical views, find the mask corresponding to each submesh
"""

import os
import sys
import argparse
import numpy as np
import json
import trimesh

import cv2
from PIL import Image, ImageDraw
from partfield.utils import run_script, gen_pcd, load_mesh_util
from tree import load_tree, save_tree, PartTree

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def render_blender(blender_path, mesh_path, output_path, types="glb"):
    """
    If canonical views are not rendered yet, render them with Blender
    """
    try:
        cmd = f"{blender_path} -b -P blender_render_views.py {mesh_path} {types} {output_path}"
        run_script(cmd)
    except Exception as e:
        print(e)


def cal_mapping_2d_3d(render_dir, mesh_path):
    """
    Maps pixels to mesh faces
    """
    mesh = load_mesh_util(mesh_path)

    meta_data = json.load(open(os.path.join(render_dir, "meta.json")))
    mesh_scale = meta_data["scaling_factor"]
    mesh_center_offset = meta_data["mesh_offset"]

    object_org_coord = mesh.vertices
    rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    object_org_coord = np.dot(object_org_coord, rotation_matrix)
    object_org_coord = object_org_coord * mesh_scale + mesh_center_offset
    mesh.vertices = object_org_coord

    # Using proximity query is a bit slower but is more accurate
    pquery = trimesh.proximity.ProximityQuery(mesh)

    mapping_list = []
    camera_angle_x = meta_data["camera_angle_x"]
    for i, c2w_opengl in enumerate(meta_data["transforms"]):
        c2w_opengl = np.array(c2w_opengl)
        rgb_path = os.path.join(render_dir, f"render_{i:04d}.webp")
        img = np.array(Image.open(rgb_path))
        if img.shape[-1] == 4:
            mask_img = img[..., 3] == 0
            img[mask_img] = [255, 255, 255, 255]
            img = img[..., :3]
            img = Image.fromarray(img.astype("uint8"))

        # Calculate mapping
        depth_path = os.path.join(render_dir, f"depth_{i:04d}.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth[..., 0]
        depth_valid = depth < 65500.0

        org_points = gen_pcd(depth, c2w_opengl, camera_angle_x)

        _, distances, indices = pquery.on_surface(org_points)
        mapping = np.zeros((depth.shape[0], depth.shape[1]), dtype=int) - 1
        mask_dis = distances < 0.03
        indices[~mask_dis] = -1
        mapping[depth_valid] = indices

        mapping_list.append(mapping)
    return np.stack(mapping_list)


def render_masks(render_dir, mesh_path, tree: PartTree) -> PartTree:
    """
    Given a parttree and canonical renders, render a mask for each node's submesh on the canonical views
    Args:
        render_dir: path to canonical renderings
        mesh_path: path to original mesh file
        tree: PartTree of submesh hierarchy. Each node contains faces of a submesh
    Return:
        parttree: same input parttree but with added image paths
    """

    # For the canonical views, calculate the mapping b/w pixels and mesh faces
    obj_mapping = cal_mapping_2d_3d(render_dir, mesh_path)
    save_dir = os.path.join(render_dir, "masks")

    def highlight_parts(node):
        faces = node.mesh

        num_images = obj_mapping.shape[0]

        masks = []
        for i in range(num_images):
            # Get the face idx for each pixel in the image
            masks.append(np.isin(obj_mapping[i], faces))

        save_path = os.path.join(save_dir)
        os.makedirs(save_path, exist_ok=True)

        image_paths = []

        # Create the query images
        for i in range(num_images):
            # Create the query image how you want
            image = Image.open(os.path.join(render_dir, f"render_{i:04d}.webp"))
            if image.mode != "RGBA":
                image = image.convert("RGBA")

            image_np = np.array(image)
            image_np[..., 3][~masks[i]] = 0
            image = Image.fromarray(image_np)

            img_dir = os.path.join(save_path, str(node.label))
            os.makedirs(img_dir, exist_ok=True)
            image_path = os.path.join(img_dir, f"{i:04d}.png")
            image.save(image_path)
            image_paths.append(image_path)

        node.set_image(image=None, image_path=image_paths)

        for child in node.children:
            highlight_parts(child)

    highlight_parts(tree.root)
    return tree


def main(args):
    out_dir = args.out_dir
    source_dir = args.source_dir
    blender_path = args.blender_path

    dataset = os.path.basename(os.path.normpath(out_dir))
    print(dataset)
    if dataset == "partobjtiny":
        types = "glb"
    elif dataset == "partnet":
        types = "obj"
    else:
        types = "obj"

    render_dir = os.path.join(out_dir, "render")
    tree_dir = os.path.join(out_dir, "tree")
    all_files = os.listdir(tree_dir)

    print(all_files)
    for f in all_files:
        uid = f.split("_")[0]
        # uid = f.rsplit("_", 2)[0]

        tree_path = os.path.join(tree_dir, f, "tree.pkl")
        mesh_path = os.path.join(source_dir, f"{uid}.{types}")

        tree = load_tree(tree_path)

        render_path = os.path.join(render_dir, f)
        if dataset == "partnet":
            tree.generate_images_from_mesh(render_path)
        else:
            if not os.path.exists(os.path.join(render_path, "meta.json")):
                # Canonical renderings do not exist. Render with Blender
                render_blender(blender_path, mesh_path, render_path, types)

            # Render masks for each tree node
            tree = render_masks(render_path, mesh_path, tree)

        save_tree(tree, tree_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="", type=str)
    parser.add_argument("--source_dir", default="", type=str)
    parser.add_argument("--blender_path", default="", type=str)

    args = parser.parse_args()
    main(args)
