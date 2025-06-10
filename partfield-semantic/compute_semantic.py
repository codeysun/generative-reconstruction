import numpy as np
import argparse
import json
import os
import re
from os.path import join
from typing import List
from collections import defaultdict
from partfield.utils import load_mesh_util
from tqdm import tqdm
import trimesh


def mean_stats(list_of_stats: list) -> dict:
    # Extract the keys from the first dictionary (assuming all dicts have the same keys)
    keys = list_of_stats[0].keys()

    # Initialize an empty dictionary to store the sums of each metric
    sums = {key: 0.0 for key in keys}  # Use float for potential decimal averages

    # Calculate the sum of each metric across all dictionaries
    num_nonzero = 0
    for stats in list_of_stats:
        if stats['iou'] > 0:
            num_nonzero += 1
        for key in keys:
            sums[key] += stats[key]

    # Calculate the average for each metric
    num_stats = len(list_of_stats)
    averages = {key: sums[key] / num_stats for key in keys}

    return averages

def compute_stats(pred, gt):
    # compute iou, precision, recall, f1score
    # Calculate TP, FP, FN
    TP = np.logical_and(pred, gt).sum() # True Positive
    FP = np.logical_and(pred, np.logical_not(gt)).sum()  # False Positive - predicted positive, was actually negative
    FN = np.logical_and(np.logical_not(pred), gt).sum()  # False Negative - predicted negative, was actually positive
    intersection = TP
    union = np.logical_or(pred, gt).sum()

    # Calculate IoU
    iou = (intersection / union) * 100 if union != 0 else 0

    # Calculate Precision
    precision = (TP / (TP + FP)) * 100 if (TP + FP) != 0 else 0

    # Calculate Recall
    recall = (TP / (TP + FN)) * 100 if (TP + FN) != 0 else 0

    # Calculate F1-score
    f1score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    metrics = {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1score": f1score,
        "TP":TP,
        "FP":FP,
        "FN":FN
    }
    return metrics

def save_submeshes(label_list, masks, mesh_path, output_dir):
    """
    Save the submeshes to ply
    """
    # TODO: Alternatively, show all mask levels and color code
    mesh = load_mesh_util(mesh_path)
    faces = mesh.faces
    os.makedirs(output_dir, exist_ok=True)

    for label, mask in zip(label_list, masks):
        selected_faces = faces[mask.astype(bool)]

        if len(selected_faces) == 0: # no prediction for this
            continue

        # Extract unique vertex indices from the selected faces
        unique_vertex_indices = np.unique(selected_faces)
        
        # Create a mapping from original vertex indices to new indices
        vertex_mapping = {old_index: new_index for new_index, old_index in enumerate(unique_vertex_indices)}
        
        # Map the selected faces to new vertex indices
        remapped_faces = np.vectorize(vertex_mapping.get)(selected_faces)
        
        # Extract vertices corresponding to unique vertex indices
        selected_vertices = mesh.vertices[unique_vertex_indices]
        
        # Create a new trimesh object with selected vertices and remapped faces
        new_mesh = trimesh.Trimesh(vertices=selected_vertices, faces=remapped_faces)
        
        # sanitize label name
        invalid_chars = r'[\\/*?:"<>|]'
        label = re.sub(invalid_chars, '_', label)

        # Save the new mesh to a .ply file
        output_path = os.path.join(output_dir, f"{label}.ply")
        new_mesh.export(output_path)

        # Also save the mask by itself
        mask_path = os.path.join(output_dir, f"{label}.npy")
        np.save(mask_path, mask)


def eval_per_shape_mean_iou(
    part_name_list: List[str],  # The name list of the shape
    pred_sem: np.ndarray,   # Predicted semantic labels, continuous natural numbers, each number is the index of the part_name_list
                            # if hierarchical: [L, B, N] B is number of predicted parts, binary label
                            # else: [N,] semantic labels
    gt_sem: np.ndarray,  # Ground truth semantic labels.
                         # if hierarchical: [L, N], binary label
                         # else: [N,] semantic labels
    visualize=False,
    mesh_path=None,
    output_dir = None
    ) -> dict:
    if visualize and not (mesh_path and output_dir):
        raise ValueError("mesh_path must be defined if visualize is set to True.")

    best_stats_all = []
    best_masks = []
    if len(pred_sem.shape) == 3:
        for i in range(len(part_name_list)):
            # best_iou = float('-inf')
            best_stats = {'iou': float('-inf')}
            best_mask = None

            if len(gt_sem.shape) == 2:
                gt_mask = gt_sem[i]
            else:
                gt_mask = gt_sem == i

            if gt_mask.sum() == 0:
                continue
            for mask in pred_sem[i]:
                stats = compute_stats(mask, gt_mask)
                if stats['iou'] > best_stats['iou']:
                    best_stats = stats
                    best_mask = mask
            best_stats_all.append(best_stats)
            best_masks.append(best_mask)

    else:
        for i in range(len(part_name_list)):
            if len(gt_sem.shape) == 2:
                gt_mask = gt_sem[i]
            else:
                gt_mask = gt_sem == i

            if gt_mask.sum() == 0:
                continue
            best_stats_all.append(compute_stats(pred_sem == i, gt_mask))
            best_masks.append(pred_sem == i)

    if visualize:
        save_submeshes(part_name_list, best_masks, mesh_path, output_dir)

    return mean_stats(best_stats_all)

def eval_all_shape_mean_iou(meta_path, pred_sem_path, gt_sem_path, **kwargs):

    meta_data = json.load(open(meta_path, 'r'))
    total_stats = []
    cate_stats = defaultdict(list)

    for cate in tqdm(meta_data.keys()):
        for uid in tqdm(meta_data[cate]):
            if not os.path.isfile(join(pred_sem_path, f"{uid}.npy")):
                continue
            print(f"Evaluating {uid}")
            part_name_list = meta_data[cate][uid]
            pred_sem = np.load(join(pred_sem_path, f"{uid}.npy"))
            gt_sem = np.load(join(gt_sem_path, f"{uid}.npy"))

            shape_kwargs = {'visualize': kwargs['visualize']}
            if kwargs['visualize'] and kwargs['mesh_dir']:
                mesh_path = os.path.join(kwargs['mesh_dir'], f"{uid}.{kwargs['types']}")
                shape_kwargs.update({'mesh_path': mesh_path, 
                                     'output_dir': os.path.join(pred_sem_path, uid)})

            obj_stats = eval_per_shape_mean_iou(part_name_list, pred_sem, gt_sem, **shape_kwargs)

            total_stats.append(obj_stats)
            print(f"miou: {obj_stats['iou']}")
            with open("eval_sem_results.txt", "a") as f:
                f.write(f"{uid}: {obj_stats['iou']}\n")
            cate_stats[cate].append(obj_stats)
    
    for cate in cate_stats.keys():
        mean_cate_stats = mean_stats(cate_stats[cate])
        print(f"{cate} miou: {mean_cate_stats['iou']}")
        print(f"{cate} prec: {mean_cate_stats['precision']}")
        print(f"{cate} recall: {mean_cate_stats['recall']}")
        print(f"{cate} f1: {mean_cate_stats['f1score']}")
        with open("eval_sem_results.txt", "a") as f:
            f.write(f"{cate} miou: {mean_cate_stats['iou']}\n")
            
    mean_total_stats = mean_stats(total_stats)
    print(f"Total miou: {mean_total_stats['iou']}")
    print(f"Total prec: {mean_total_stats['precision']}")
    print(f"Total recall: {mean_total_stats['recall']}")
    print(f"Total f1: {mean_total_stats['f1score']}")
    with open("eval_sem_results.txt", "a") as f:
        f.write(f"Total miou: {mean_total_stats['iou']}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval semantic IoU')
    parser.add_argument('--meta_path', default= "../data/PartObjaverse-Tiny/PartObjaverse-Tiny_semantic.json", type=str)
    parser.add_argument('--pred_sem_path', default= "exp_results/clustering/partobjtiny/sem_preds", type=str)
    parser.add_argument('--gt_sem_path', default= "../data/PartObjaverse-Tiny/PartObjaverse-Tiny_semantic_gt", type=str)
    parser.add_argument('--mesh_dir', default= "../data/PartObjaverse-Tiny/PartObjaverse-Tiny_mesh", type=str)
    parser.add_argument('--types', default= "glb", type=str)
    # parser.add_argument('--meta_path', default= "./data/partnet/partnet_semantic.json", type=str)
    # parser.add_argument('--pred_sem_path', default= "./exp_results/clustering/partnet/sem_preds", type=str)
    # parser.add_argument('--gt_sem_path', default= "./data/partnet/semantic_gt", type=str)
    # parser.add_argument('--mesh_dir', default= "./data/partnet/mesh", type=str)
    # parser.add_argument('--types', default= "obj", type=str)
    parser.add_argument('--viz', action='store_true', help='Visualize submeshes')
    args = parser.parse_args()

    meta_path = args.meta_path
    pred_sem_path = args.pred_sem_path
    gt_sem_path = args.gt_sem_path
    eval_all_shape_mean_iou(meta_path, pred_sem_path, gt_sem_path, visualize=args.viz, mesh_dir=args.mesh_dir, types=args.types)
