# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
OmniGen Image Generation Script

This script loads a dataset of prompts and reference images, and uses the OmniGen pipeline
to generate images based on text and image inputs. It supports optional pose priors and
configurable generation parameters such as image size, guidance scales, and random seed.

Usage:
    python generate_images.py --dataset_json_file path/to/metadata.json --output_path results/
"""

import argparse
import json
import os
import torch
from OmniGen import OmniGenPipeline

def parse_args():
    """
    Parse command-line arguments for the image generation script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run OmniGen image generation pipeline.")
    parser.add_argument("--dataset_json_file", default="data/multihuman_metadata.json", help="Path to the input metadata JSON file.")
    parser.add_argument("--output_path", default="results/omnigen/", help="Directory to store generated images.")
    parser.add_argument("--image_guidance_scale", type=float, default=2.0, help="Guidance scale for image conditioning.")
    parser.add_argument("--text_guidance_scale", type=float, default=2.5, help="Guidance scale for text conditioning.")
    parser.add_argument("--model_name", default="Shitao/OmniGen-v1", help="Model name or path for the OmniGen pipeline.")
    parser.add_argument("--pose_prior", action="store_true", help="Whether to use pose priors.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--seed", type=int, default=878, help="Random seed for reproducibility.")
    return parser.parse_args()


def load_metadata(filepath):
    """
    Load the dataset metadata from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        list: List of metadata entries.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def generate_images(args):
    """
    Generate images using the OmniGen pipeline based on the provided arguments.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    pipe = OmniGenPipeline.from_pretrained(args.model_name)
    pipe.to(device)

    metadata = load_metadata(args.dataset_json_file)
    os.makedirs(args.output_path, exist_ok=True)

    for test_id, item in enumerate(metadata):
        filename = os.path.join(args.output_path, f"{test_id}.png")

        for i in range(len(item["people"])):
            filepath_str = item["people"][i]
            item["people"][i] = filepath_str

        if args.pose_prior:
            final_prompt = item["omnigen_pose_prompt"]
            image_inputs = item["people"]
            image_inputs.append(pose_image)
            input_size = 1024
        else:
            final_prompt = item["omnigen_prompt"]
            image_inputs = item["people"]
            input_size = 200

        images = pipe(
            prompt=final_prompt,
            input_images=image_inputs,
            height=args.height,
            width=args.width,
            guidance_scale=args.text_guidance_scale,
            img_guidance_scale=args.image_guidance_scale,
            max_input_image_size=input_size,
            seed=args.seed,
            use_kv_cache=False,
        )

        images[0].save(filename)


def main():
    args = parse_args()
    generate_images(args)


if __name__ == "__main__":
    main()
