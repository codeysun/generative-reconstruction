import torch
import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree

from threestudio.utils.typing import *
from threestudio.utils.openclip_utils import OpenCLIPNetwork, OpenCLIPNetworkConfig

import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "./lib_shape_prior/")))
from threestudio.utils.lib_shape_prior.core.models.utils.occnet_utils.utils.libmesh.inside_mesh import check_mesh_contains

def compute_chamfer_distance(gt_mesh, gen_mesh, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
            compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
            method (see compute_metrics.py for more)

    """

    gt_points_sampled = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]
    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    # gt_points_np = gt_points_sampled.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_sampled)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_sampled)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer, gen_to_gt_chamfer

def compute_volumetric_iou(mesh1, mesh2, voxel_size = 1./16):

    inside_mask = check_mesh_contains(mesh1, mesh2.vertices)
    return inside_mask.mean()

def compute_clip_similarity(images, prompt):
    """
    Uses open CLIP model to determine image-text similarity
    Args:
        images: list[torch.tensor]
        prompt: str

    Returns:
        metric: float averaging CLIP score of all given views
    """
    # Load CLIP model
    clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)

    # Optional: Add directionality to the prompts
    directions = ['a front view of', 'a side view of', 'a back view of', 'a side view of']
    prompts = []
    for direction in directions:
        prompts.append(direction + prompt)

    tiles = torch.stack(images).to('cuda')
    image_embeds = clip_model.encode_image(tiles)
    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
    image_embeds = image_embeds.half()

    text_embeds = clip_model.encode(prompts).to(image_embeds)

    relevancy = torch.mm(text_embeds, image_embeds.T)

    score = 0
    for i in range(len(images)):
        score += relevancy[i, i].item()

    return score / len(images)



    