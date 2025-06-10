import gc
import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
import sys
import json
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_DIR)

from subprocess import call
from tqdm import tqdm
from tree import PartTree, save_tree, load_tree
from openclip_utils import OpenCLIPNetwork, OpenCLIPNetworkConfig
from eval_clip import evaluate_iou
from tree_query_qwen import query_tree, load_model_and_processor
from sentence_transformers import SentenceTransformer


def read_selected_directories(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]

USE_LMDEPLOY = False


def run_script(cmd):
    ret = call(cmd, shell=True)
    if ret != 0:
        raise Exception(f"Failed to run {cmd}")


def build_trees(num_clusters=10):
    try:
        cmd = f"python tree_vis.py --root exp_results/partfield_features/partnet --dump_dir exp_results/clustering/partnet --source_dir data/partnet --use_agglo True --max_num_clusters {num_clusters} --option 0"
        run_script(cmd)
    except Exception as e:
        print(e)


def label_trees(pred_dir):
    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    model, processor = load_model_and_processor(
        DEFAULT_MODEL, use_lmdeploy=USE_LMDEPLOY
    )

    tree_dir = os.path.join(pred_dir, "tree")
    render_dir = os.path.join(pred_dir, "render")

    all_files = os.listdir(tree_dir)
    for f in tqdm(all_files):
        uid = f.split("_")[0]

        if os.path.isfile(os.path.join(tree_dir, f, "tree_labeled.pkl")):
            continue

        print(uid)

        tree_path = os.path.join(tree_dir, f, "tree.pkl")
        render_path = os.path.join(render_dir, f)

        tree = load_tree(tree_path)
        tree.set_render_dir(render_path)
        query_tree(tree, model, processor, use_lmdeploy=USE_LMDEPLOY)
        tree.render_tree(os.path.join(os.path.dirname(tree_path), "tree_with_labels"))
        save_tree(tree, os.path.join(os.path.dirname(tree_path), "tree_labeled.pkl"))


def embed_tree(model, tree: PartTree, file):
    """
    If tree already has captions, embed the captions in text encoder
    """
    assert tree.root.caption, "Tree has no captions. Run tree through VLM first."
    nodes = tree.get_nodes()

    texts = []
    for node in nodes:
        caption = node.caption
        # assert caption, f"Node {node} of file {file} has no caption"
        texts.append(caption)

    embed = model.encode(texts, convert_to_tensor=True)

    for idx, node in enumerate(nodes):
        node.set_embed(embed[idx])


def get_semantic(model_name, out_dir, gt_path, k=None):
    """
    Embed the PartTree and create the proper numpy array for evaluation.
    Ret:
        out (np.ndarray): shape [L,B,N] of semantic label, mask (from top k), faces (binary label)
    """
    # Get the text embedding model
    torch.cuda.empty_cache()
    if model_name == "clip":
        model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    elif model_name == "gte":
        torch.set_default_dtype(torch.float16)
        model = SentenceTransformer(
            "Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True, device="cuda"
        )
        model.max_seq_length = 4096
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    tree_dir = os.path.join(out_dir, "tree")
    all_files = os.listdir(tree_dir)
    preds_dir = os.path.join(out_dir, "sem_preds")

    os.makedirs(preds_dir, exist_ok=True)

    # Parse ground truth. Not interested in categories, so merge all the uids
    meta_data = json.load(open(gt_path, "r"))
    gt_labels = {}
    for category, uids in meta_data.items():
        gt_labels.update(uids)

    for f in tqdm(all_files):
        torch.cuda.empty_cache()
        uid = f.split("_")[0]

        tree_path = os.path.join(tree_dir, f, "tree_labeled.pkl")
        tree = load_tree(tree_path)

        # Get the embeddings for each node
        with torch.no_grad():
            embed_tree(model, tree, f)

        # Put all the nodes + embedding into a list
        nodes = tree.get_nodes()

        # Calculate relevancy map
        labels = gt_labels[uid]
        embeds = []
        for node in nodes:
            embed = node.embed
            assert embed is not None, f"Node {node} has no semantic embedding"
            embeds.append(embed)
        embeds = torch.stack(embeds).to("cuda")  # nodes x 512

        if isinstance(model, OpenCLIPNetwork):
            model.set_positives(labels)
            relevancy = (
                model.get_max_across(embeds).detach().cpu().numpy()
            )  # phrases x nodes
        else:
            query_embeds = model.encode(
                labels, prompt_name="query", convert_to_tensor=True
            )
            relevancy = (
                (query_embeds @ embeds.T).detach().cpu().numpy()
            )  # phrases x nodes

        # Get top k tree node activations
        n_phrases, n_nodes = relevancy.shape
        if k is None or k > n_nodes:
            k = n_nodes  # save all results
        n_faces = len(tree.root.mesh)
        top_k_indices = np.argsort(relevancy, axis=1)[:, -k:][:, ::-1]

        preds = []
        preds = np.zeros((n_phrases, k, n_faces), dtype=int)
        for i in range(n_phrases):
            # For each top k, combine all the faces
            accumulated_faces = set()
            for j, node_idx in enumerate(top_k_indices[i]):
                faces = nodes[node_idx].mesh
                if isinstance(faces, tuple):  # np.where returns a tuple
                    faces = faces[0]
                accumulated_faces.update(faces.tolist())
                preds[i, j, list(accumulated_faces)] = 1

        # Save the numpy array
        pred_sem_path = os.path.join(preds_dir, f"{uid}.npy")
        np.save(pred_sem_path, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate tree models with different embeddings"
    )

    # Model selection arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["clip", "gte"],
        default="gte",
        help="Model type to use for evaluation (default: gte)",
    )
    # Evaluation options
    parser.add_argument(
        "--out_dir", default="", type=str, help="Directory with clustering outputs"
    )
    parser.add_argument("--gt_path", default="", type=str, help="Path of GT json data")
    parser.add_argument(
        "--label_trees", action="store_true", help="Label trees before evaluation"
    )
    parser.add_argument("--eval", action="store_true", help="Run evaluation")

    args = parser.parse_args()

    if args.label_trees:
        print("Labeling trees...")
        label_trees(args.out_dir)
        gc.collect()
        torch.cuda.empty_cache()

    if args.eval:
        stats = get_semantic(args.model, args.out_dir, args.gt_path)
