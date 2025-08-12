# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import time
from typing import Union
import facer
import hpsv2
import huggingface_hub
import torch
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.utils import hps_version_map
from huggingface_hub import snapshot_download
from onnx2torch import convert
from PIL import Image

############# Initialize HPS model
def initialize_hps_model(device):
    """
    Initializes the HPSv2 model with pretrained weights and tokenizer.

    Args:
        device (torch.device): The device to load the model onto.

    Returns:
        tuple: (hps_model, tokenizer, preprocess_val) for inference.
    """
    hps_model_dict = {}
    if not hps_model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
            cache_dir="models/",
        )
        hps_model_dict["model"] = model
        hps_model_dict["preprocess_val"] = preprocess_val

    hps_model = hps_model_dict["model"]
    preprocess_val = hps_model_dict["preprocess_val"]
    hps_version = "v2.1"
    cp = huggingface_hub.hf_hub_download(
        "xswu/HPSv2", hps_version_map[hps_version], local_dir="models/"
    )
    hps_checkpoint = torch.load(cp, map_location=device)
    hps_model.load_state_dict(hps_checkpoint["state_dict"])
    tokenizer = get_tokenizer("ViT-H-14")
    hps_model = hps_model.to(device)
    hps_model.eval()
    return hps_model, tokenizer, preprocess_val

########### Init peripherals
def init_peripherals(device):
    """
    Initializes all required peripherals for multi-human evaluation, including
    face detection, identity embedding, and HPS scoring.

    Args:
        device (torch.device): The device to load models onto.

    Returns:
        tuple: Initialized (face_detector, ant_model, hps_model, cosine_sim, tokenizer, preprocess_val).
    """
    face_detector = facer.face_detector("retinaface/mobilenet", device=device)
    snapshot_download("DIAMONIK7777/antelopev2", local_dir="models/antelopev2")
    ant_model = convert("models/antelopev2/glintr100.onnx").eval().to(device)
    for param in ant_model.parameters():
        param.requires_grad_(False)
    hps_model, tokenizer, preprocess_val = initialize_hps_model(device)
    cosine_sim = torch.nn.CosineSimilarity(dim=0)
    return face_detector, ant_model, hps_model, cosine_sim, tokenizer, preprocess_val

########### Detect Faces
def face_detect(face_detector, image_file, device):
    """
    Detects faces in an image using the specified face detector.

    Args:
        face_detector: The face detection model.
        image_file (str): Path to the image file.
        device (torch.device): The device for inference.

    Returns:
        tuple: (faces, face_image) where faces is the detection result and face_image is the tensor.
    """
    face_image = facer.hwc2bchw(facer.read_hwc(image_file)).to(
        device=device
    )  # image: 1 x 3 x h x w
    with torch.inference_mode():
        faces = face_detector(face_image)
    return faces, face_image

########### HPS Score
def hps_score_function(
    img_path: Union[list, str, Image.Image],
    prompt: str,
    tokenizer,
    preprocess_val,
    device,
    hps_model,
) -> list:
    """
    Computes the Human Preference Score (HPS) between an image and a prompt.

    Args:
        img_path (Union[list, str, Image.Image]): Input image(s) for evaluation.
        prompt (str): Text prompt describing the image.
        tokenizer: Tokenizer for processing the prompt.
        preprocess_val: Preprocessing function for the image.
        device (torch.device): Device for model inference.
        hps_model: The HPS model.

    Returns:
        list: List of HPS scores for each image.
    """
    if isinstance(img_path, list):
        result = []
        for one_img_path in img_path:
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                image = (
                    preprocess_val(one_img_path)
                    .unsqueeze(0)
                    .to(device=device, non_blocking=True)
                )
                # Process the prompt
                text = tokenizer([prompt]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.cuda.amp.autocast():
                    outputs = hps_model(image, text)
                    image_features, text_features = (
                        outputs["image_features"],
                        outputs["text_features"],
                    )
                    logits_per_image = image_features @ text_features.T
                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            result.append(hps_score[0])
        return result
