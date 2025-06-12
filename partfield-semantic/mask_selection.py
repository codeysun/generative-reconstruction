import gc
import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
import sys
import json
import matplotlib.pyplot as plt
import trimesh

from tqdm import tqdm
from tree import PartTree, save_tree, load_tree
from openclip_utils import OpenCLIPNetwork, OpenCLIPNetworkConfig
from sentence_transformers import SentenceTransformer


def create_submesh(mesh: trimesh.Trimesh, face_indices) -> trimesh.Trimesh:
    new_faces = mesh.faces[face_indices]

    # Find the unique vertices that are used in the new faces.  This is *crucial*
    # because the face indices refer to the indices in the *original* vertex array.
    # We need to remap these to the new vertex array.
    unique_vertex_indices = np.unique(new_faces.flatten())
    new_vertices = mesh.vertices[unique_vertex_indices]

    # Create a mapping from original vertex index to new vertex index.
    # This lets us reconstruct the faces with the correct references to the new vertex array.
    vertex_mapping = {
        old_index: new_index
        for new_index, old_index in enumerate(unique_vertex_indices)
    }

    # Remap the face indices to refer to the new vertex array.
    remapped_faces = np.array([[vertex_mapping[v] for v in face] for face in new_faces])

    submesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
    return submesh


def embed_tree(model, tree: PartTree):
    """
    If tree already has captions, embed the captions in text encoder
    """
    assert tree.root.caption, "Tree has no captions. Run tree through VLM first."
    nodes = tree.get_nodes()

    texts = []
    for node in nodes:
        caption = node.caption
        # assert caption, f"Node {node} has no caption"
        texts.append(caption)

    embed = model.encode(texts, convert_to_tensor=True)

    for idx, node in enumerate(nodes):
        node.set_embed(embed[idx])


def get_semantic(model_name, out_dir, uid, labels, k=None):
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

    preds_dir = os.path.join(out_dir, "sem_preds")
    submesh_dir = os.path.join(preds_dir, uid)

    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(submesh_dir, exist_ok=True)

    torch.cuda.empty_cache()

    tree_dir = os.path.join(out_dir, "tree")
    tree = load_tree(os.path.join(tree_dir, uid, "tree_labeled.pkl"))

    # Get the embeddings for each node
    with torch.no_grad():
        embed_tree(model, tree)

    # Put all the nodes + embedding into a list
    nodes = tree.get_nodes()

    # Calculate relevancy map
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
        query_embeds = model.encode(labels, prompt_name="query", convert_to_tensor=True)
        relevancy = (query_embeds @ embeds.T).detach().cpu().numpy()  # phrases x nodes

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

            # TODO: save the mesh
            submesh = create_submesh(tree.root_mesh, list(accumulated_faces))
            submesh.export(os.path.join(submesh_dir, f"{labels[i]}_{j}.ply"))

            np.save(
                os.path.join(submesh_dir, f"{labels[i]}_{j}.npy"),
                preds[i, j, :],
            )

    # # Save the numpy array
    # pred_sem_path = os.path.join(preds_dir, f"{uid}.npy")
    # np.save(pred_sem_path, preds)


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

    parser.add_argument(
        "--out_dir", default="", type=str, help="Directory with clustering outputs"
    )

    args = parser.parse_args()

    # TODO: Change here!
    uid = "0c3ca2b32545416f8f1e6f0e87def1a6_0" # object ID
    labels = ["bowl", "apples", "stem"] # Semantic queries

    stats = get_semantic(args.model, args.out_dir, uid, labels)
