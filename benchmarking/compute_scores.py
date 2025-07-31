# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All rights reserved.

import argparse
import json
import os
import pathlib

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils import (MLLM_AWAKE, crop_face, face_detect, gather_and_print_scores,
                   hps_score_function, hungarian_algorithm, init_peripherals,
                   mllm_vqa)


def iterate_and_dump(args):
    """
    Processes a dataset of prompts and associated images to evaluate and log metrics.

    This function performs the following steps:
    1. Loads metadata from our JSON file.
    2. Initializes necessary models and peripherals (e.g., face detector, HPS model, etc.).
    3. Iterates through each test case, performing:
        - Face detection on generated and reference images.
        - HPS (Human Preference Score) evaluation.
        - Identity matching using cosine similarity and the Hungarian algorithm.
        - Optional evaluation using a Multimodal Large Language Model (MLLM) for VQA.
    4. Aggregates and logs metrics such as accuracy, face similarity, HPS, and MLLM scores.
    5. Writes per-sample results to a JSONL file and prints summary statistics for simple, 
       complex, and all prompts.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - stored_result_path (str): Path to the directory containing generated images and 
              where the output JSONL file will be saved.
            - dataset_json_file (str): Path to the JSON file containing metadata for the test cases.
            - mllm_metrics (bool): Flag indicating whether to compute MLLM-based metrics.

    Outputs:
        - A JSONL file named 'scores.jsonl' containing per-sample evaluation results.
        - Printed summary statistics for different prompt categories (simple, complex, all).
    """
    # Open a json file to dump scores
    out_file = open(f"{args.stored_result_path}/scores.jsonl", "w")

    # Metadata json file for testbench
    with open(args.dataset_json_file, "r") as file:
        metadata = json.load(file)

    # Init peripherals
    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_detector, ant_model, hps_model, cosine_sim, tokenizer, preprocess_val = (
        init_peripherals(device)
    )

    # Initialize scores
    face_sim = 0
    accuracy = 0
    hps = 0
    total = 0
    split_metrics = {
        1: [0, 0, 0],
        2: [0, 0, 0],
        3: [0, 0, 0],
        4: [0, 0, 0],
        5: [0, 0, 0],
    }
    total_people = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    if MLLM_AWAKE and args.mllm_metrics:
        lmm_score = 0
        num_lmm = 0

    num_simple_prompts = 1500
    total_simple = 0
    num_lmm_simple = 0
    
    # Iterating over all prompts, including 1500 simple prompts and 300 complex prompts.
    for test_id, item in tqdm(enumerate(metadata), total=len(metadata)):
        prompt = item["prompt"]
        people = item["people"]
        questions = item["vlm_questions"]
        image_file = f"{args.stored_result_path}/{test_id}.png"

        num_people = len(people)
        total += 1
        total_people[num_people] += 1

        # Skip if the file doesn't exist
        if not os.path.exists(image_file):
            total -= 1
            total_people[num_people] -= 1
            continue
        pil_image = Image.open(image_file)

        # Detect Faces
        faces, face_image = face_detect(face_detector, image_file, device)

        # HPS scores
        hps_score = hps_score_function(
            [pil_image], prompt, tokenizer, preprocess_val, device, hps_model
        )[0]
        hps += hps_score
        split_metrics[num_people][0] += hps_score

        if MLLM_AWAKE and args.mllm_metrics:
            answers = mllm_vqa(questions, pil_image)
            try:
                answers = answers.replace("\n", "")
                num_questions = len(questions)
                for q_id in range(num_questions):
                    score = 1 + int(answers[q_id])
                    lmm_score += score
                    num_lmm += 1
            except Exception as e:  # Catch the exception and assign it to 'e'
                print(f"Can't parse MLLM answers for {test_id}. Error: {e}")

        # Filter images with no faces
        no_gen = len(faces.keys()) == 0 or faces["rects"].nelement() == 0
        if not no_gen:
            # Find Generated Face Arcface ID
            num_genned_faces = len(faces["rects"])
            gen_embeds = []
            for g_face in range(num_genned_faces):
                gen_embeds.append(ant_model(crop_face(faces, face_image, g_face)))

            # Find Reference Face Arcface ID
            person_embeds = []
            for p_face, person in enumerate(people):
                person_face, person_image = face_detect(face_detector, person, device)
                person_embeds.append(ant_model(crop_face(person_face, person_image, 0)))

            # Construct a cost matrix for matching
            cost_matrix = np.zeros((len(person_embeds), num_genned_faces))
            for i_p, p_ in enumerate(person_embeds):
                for i_g, g_ in enumerate(gen_embeds):
                    cost_matrix[i_p, i_g] = (
                        cosine_sim(p_.flatten(), g_.flatten()).cpu().numpy()
                    )

            # Hungarian Algorithm
            _, dict_assignments = hungarian_algorithm(cost_matrix)

            # Compute the Hungarian ID similarity
            matched_similarity = 0.0
            all_ids = []
            for person_id in range(len(person_embeds)):
                assignment_id = dict_assignments[person_id]
                if num_genned_faces > assignment_id:
                    matched_similarity += cost_matrix[person_id, assignment_id]
                    all_ids.append(cost_matrix[person_id, assignment_id])
                else:
                    all_ids.append(0)

            face_sim += matched_similarity / num_people
            id_match = str(matched_similarity / num_people)
            split_metrics[num_people][1] += matched_similarity / num_people

            # Compute the Accuracy
            accuracy += int(num_people == num_genned_faces)
            split_metrics[num_people][2] += int(num_people == num_genned_faces)

        else:
            num_genned_faces = 0
            id_match = 0
            all_ids = num_people * [0]

        # Construct dump dictionary
        data_dict = {
            "idx": test_id,
            "accuracy": int(num_people == num_genned_faces),
            "hps": str(hps_score),
            "id_match": id_match,
            "all_ids": all_ids,
        }
        if MLLM_AWAKE and args.mllm_metrics:
            data_dict["scores"] = answers

        json.dump(data_dict, out_file)
        out_file.write("\n")

        # Print scores for 1500 Simple action prompts
        if test_id == num_simple_prompts-1:
            print("------------------Gathering Simple Prompt Scores------------------")
            gather_and_print_scores(
                accuracy, face_sim, hps, split_metrics, total, total_people
            )
            if MLLM_AWAKE and args.mllm_metrics:
                print(f"MLLM Action-S score: {lmm_score/num_lmm}")
                lmm_simple = 0.0 + lmm_score
                num_lmm_simple = 0.0 + num_lmm

            accuracy_simple = 0.0 + accuracy
            face_sim_simple = 0.0 + face_sim
            hps_simple = 0.0 + hps
            total_simple = 0.0 + total

    # Check if simple scores were cached
    if total_simple>0:    
        # Print scores for Complex action prompts
        print("------------------Gathering Complex prompt scores------------------")
        print(f"Found {total-total_simple} Images")
        print(
            f"Count Accuracy: {(accuracy-accuracy_simple)/(total-total_simple)}, Face Similarity: {(face_sim-face_sim_simple)/(total-total_simple)}, HPS: {(hps-hps_simple)/(total-total_simple)}"
        )
        if MLLM_AWAKE and args.mllm_metrics:
            print(f"MLLM Action-C score: {(lmm_score-lmm_simple)/(num_lmm-num_lmm_simple)}")

    # Print scores for all prompts
    print("------------------Gathering All prompt scores------------------")
    print(f"Found {total} Images")
    gather_and_print_scores(accuracy, face_sim, hps, split_metrics, total, total_people)

    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument(
        "--dataset_json_file",
        help="Path to the data json file",
        default="data/multihuman_metadata.json",
    )
    parser.add_argument(
        "--stored_result_path",
        help="Path to the stored results",
        default="results/omnigen",
    )
    parser.add_argument(
        "--mllm_metrics",
        help="If we want to generate action scores using VQA (Requires Gemini ID)",
        action="store_true",
    )

    args = parser.parse_args()

    iterate_and_dump(args)
