# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All rights reserved.

import json
import os
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from tqdm import tqdm


def load_lookup_table(filepath):
    """
    Load the dataset ID lookup table from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: A dictionary mapping dataset names to lists of image IDs.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def prepare_output_directory(directory):
    """
    Create the output directory if it doesn't exist.

    Args:
        directory (str): Path to the output directory.
    """
    os.makedirs(directory, exist_ok=True)


def download_celeba_images(lookup, out_dir):
    """
    Download and save CelebAHQ images based on the provided lookup IDs.

    Args:
        lookup (dict): Dictionary containing image IDs for CelebAHQ.
        out_dir (str): Directory to save the images.
    """
    celebahq_ids = set(map(int, lookup["celebahq"]))
    data_train = load_dataset("FrsECM/CelebAHQ_mask", split="train", streaming=True)
    data_val = load_dataset("FrsECM/CelebAHQ_mask", split="test", streaming=True)
    data_celebhq = concatenate_datasets([data_train, data_val])

    print("Iterating through CelebAHQ dataset to find images")
    for item in tqdm(data_celebhq):
        im_id = int(item["image_id"])
        if im_id in celebahq_ids:
            image = item["image"].resize((512, 512))
            image.save(f"{out_dir}/celeba_hq_{im_id}.jpg")


def download_ffhq_images(lookup, out_dir):
    """
    Download and save FFHQ images based on the provided lookup IDs.

    Args:
        lookup (dict): Dictionary containing image IDs for FFHQ.
        out_dir (str): Directory to save the images.
    """
    ffhq_ids = set(map(int, lookup["ffhq"]))
    data_ffhq = load_dataset("bitmind/ffhq-256", split="train", streaming=True)

    print("Iterating through FFHQ dataset to find images")
    for idx, item in tqdm(enumerate(data_ffhq)):
        if idx in ffhq_ids:
            image = item["image"].resize((512, 512))
            image.save(f"{out_dir}/ffhq_{idx}.jpg")


def download_sfhq_images(lookup, out_dir):
    """
    Download and save SFHQ images based on the provided lookup IDs.

    Args:
        lookup (dict): Dictionary containing image IDs for SFHQ.
        out_dir (str): Directory to save the images.
    """
    sfhq_ids = set(map(int, lookup["sfhq"]))
    data_sfhq = load_dataset("canva999888/SFHQ-Tiny-512-Part1", split="validation", streaming=True)

    print("Iterating through SFHQ dataset to find images")
    for item in tqdm(data_sfhq):
        ID = int(item["__key__"].split("_")[2])
        if ID in sfhq_ids:
            image = item[".jpg"].resize((512, 512))
            image.save(f"{out_dir}/sfhq_{ID}.jpg")


def main():
    """
    Main function to orchestrate downloading and saving images from CelebAHQ, FFHQ, and SFHQ datasets.
    """
    out_dir = "data/faces"
    lookup_path = "data_generation/dataset_ids.json"

    prepare_output_directory(out_dir)
    lookup = load_lookup_table(lookup_path)

    download_celeba_images(lookup, out_dir)
    download_ffhq_images(lookup, out_dir)
    download_sfhq_images(lookup, out_dir)


if __name__ == "__main__":
    main()
