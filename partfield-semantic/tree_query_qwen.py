import torch
import numpy as np
import scipy.sparse as sp
import struct
import imageio
import os
import argparse
import sys
import tqdm
import glob
import json
import base64
import re
import hashlib
import requests
import decord
from decord import VideoReader, cpu


from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from tree import PartTree, save_tree, load_tree
from prompts2 import (
    root_system_prompt,
    system_prompt,
    root_query_prompt,
    query_prompt,
    verification_prompt,
)


# Constants
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_OUTPUT_TOKENS = 1024
MAX_IMAGE_SIZE = (1120, 1120)


def load_model_and_processor(
    model_name: str, finetuning_path: str = None, use_lmdeploy=True
):
    """Load model and processor with optional LoRA adapter"""
    print(f"Loading model: {model_name}")

    if use_lmdeploy:
        from lmdeploy import pipeline, TurbomindEngineConfig
        from lmdeploy.vl import load_image
        from lmdeploy.vl.constants import IMAGE_TOKEN

        backend_config = TurbomindEngineConfig(
            session_len=2048, max_prefill_token_num=1024, cache_max_entry_count=0.8
        )
        pipe = pipeline(model_name, backend_config=backend_config)

        # gen_config = GenerationConfig(
        #     do_sample=True,
        #     temperature=0.2,
        #     max_new_tokens=2048)

        return pipe, None
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        processor = AutoProcessor.from_pretrained(model_name)

        return model, processor


def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir=".cache"):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode("utf-8")).hexdigest()
    if video_path.startswith("http://") or video_path.startswith("https://"):
        video_file_path = os.path.join(cache_dir, f"{video_hash}.mp4")
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f"{video_hash}_{num_frames}_frames.npy")
    timestamps_cache_file = os.path.join(
        cache_dir, f"{video_hash}_{num_frames}_timestamps.npy"
    )

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)

    return video_file_path, frames, timestamps


def process_image(image_path: str = None, image=None) -> Image.Image:
    """Process and validate image input"""
    if image is not None:
        if isinstance(image, np.ndarray):
            return Image.fromarray(image, "RGB")
        return image.convert("RGB")
    if image_path and os.path.exists(image_path):
        im = np.array(Image.open(image_path))
        # Make transparency -> white background
        if im.shape[-1] == 4:
            mask = im[..., 3] != 0
            im[~mask] = [255, 255, 255, 255]
            im = im[..., :3]
        im = Image.fromarray(im.astype("uint8"))
        return im.convert("RGB")
    raise ValueError("No valid image provided")


class CaptionNotFoundError(Exception):
    """Custom exception raised when the <caption> tag is not found."""

    pass


def get_caption(result: str) -> str:
    label_pattern = r"<caption>(.*?)</caption>"
    label_match = re.search(label_pattern, result, re.DOTALL)
    if not label_match:
        raise CaptionNotFoundError("<caption> tag not found")

    label_text = label_match.group(1)

    return label_text


def generate_text_from_image(model, processor, images, conversation):
    """Generate text from image using model"""
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    # print("Input Prompt:\n", prompt)

    # # Images
    # image_inputs, video_inputs = process_vision_info([conversation])
    # inputs = processor(text=[prompt], images=image_inputs, padding=True, return_tensors="pt")
    # inputs = inputs.to('cuda')

    # Video
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [conversation], return_video_kwargs=True
    )
    fps_inputs = video_kwargs["fps"]
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        fps=fps_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKENS)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Free GPU memory
    del inputs
    torch.cuda.empty_cache()

    return output_text[0]


def get_video(node, num_images):
    video = []
    for i in range(num_images):
        image = process_image(image_path=node.query_image_paths[i])

        # If image is too small then skip this node
        w, h = image.size
        if w == 1 or h == 1:
            return

        video.append(image)

        # # Visualize the image
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 8))
        # plt.imshow(image)
        # plt.title(f"Image {i} of {num_images}")
        # plt.axis('on')  # Show axes for size reference
        # plt.tight_layout()
        # plt.show()  # This displays the plot in notebooks or interactive sessions
    return video


def label_tree(model, processor, tree, use_lmdeploy=True, **kwargs):
    images = []
    object_label = None
    object_video = None

    if kwargs["traversal"] == "dfs":

        def label_node_dfs(node, parent_caption=None):
            nonlocal object_label, object_video

            num_images = len(node.query_image_paths)
            video = get_video(node, num_images)
            formatted_content = []
            if node is tree.root:
                conversation = [
                    {"role": "system", "content": root_system_prompt(**kwargs)},
                ]
                formatted_content = root_query_prompt(video1=video, **kwargs)

                object_video = video
            else:
                conversation = [
                    {"role": "system", "content": system_prompt(**kwargs)},
                ]
                formatted_content = query_prompt(
                    video1=object_video,
                    video2=video,
                    object_caption=object_label,
                    parent_caption=parent_caption,
                    **kwargs,
                )

            conversation.append(
                {
                    "role": "user",
                    "content": formatted_content,
                }
            )

            label = ""
            for attempt in range(2):
                try:
                    if use_lmdeploy:
                        result = model(conversation)
                    else:
                        result = generate_text_from_image(
                            model, processor, images, conversation
                        )

                    print(result)
                    print()

                    if (node is tree.root and kwargs["individual"]) or kwargs["naive"]:
                        label = result
                    else:
                        label = get_caption(result)
                    break
                except CaptionNotFoundError as e:
                    print(f"Attempt {attempt+1}: {e}. Retrying")
                    label = result

            if node is tree.root:
                object_label = label

            # Verification step
            if kwargs["verify"] and node is not tree.root:
                verif_conversation = [
                    {
                        "role": "user",
                        "content": verification_prompt(video1=object_video, video2=video, caption=label)
                    },
                ]

                print("Verification:\n")
                for attempt in range(2):
                    try:
                        verif_result = generate_text_from_image(
                            model, processor, images, verif_conversation
                        )

                        if (node is tree.root and kwargs["individual"]) or kwargs["naive"]:
                            label =verif_result 
                        else:
                            label = get_caption(verif_result)
                        break
                    except CaptionNotFoundError as e:
                        print(f"Attempt {attempt+1}: {e}. Retrying")

            caption = f"{label}"

            node.set_caption(caption)

            for child in node.children:
                label_node_dfs(child, label)

        label_node_dfs(tree.root)


def query_tree(tree: PartTree, model, processor, use_lmdeploy=True):
    # tree.query_preprocess(crop=True, pad=True, resize=(224, 224))
    tree.query_preprocess(highlight=True)
    # tree.query_preprocess()
    label_tree(
        model,
        processor,
        tree,
        traversal="dfs",
        individual=True,
        naive=False,
        verify=False,
        use_lmdeploy=use_lmdeploy,
    )
    # label_tree(model, processor, tree, traversal='dfs', individual=True)
    # label_tree(model, processor, tree, traversal='bfs')
    # label_tree(model, processor, tree, traversal='bottomup')


def main(args):
    """Main execution flow"""
    model, processor = load_model_and_processor(args.model_name)
    tree = load_tree(args.tree_path)
    query_tree(tree, model, processor)
    tree.render_tree(os.path.join(os.path.dirname(args.tree_path), "tree_with_labels"))
    save_tree(tree, os.path.join(os.path.dirname(args.tree_path), "tree_labeled.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-modal inference with optional Gradio UI and LoRA support"
    )
    parser.add_argument("--tree_path", type=str, help="Path to part hierarchy tree")
    parser.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL, help="Model name"
    )

    args = parser.parse_args()
    torch.cuda.empty_cache()
    main(args)
