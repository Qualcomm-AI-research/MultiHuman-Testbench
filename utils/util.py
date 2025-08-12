# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import pathlib

import facer
import numpy as np
import torch
import torch.nn.functional as F


########### Hungarian Matching
def hungarian_algorithm(cost_matrix):
    """
    Applies the Hungarian algorithm to a given cost matrix.  Assumes that
    a higher value in the cost_matrix indicates a *better* assignment
    (i.e., it tries to maximize the total cost).

    Args:
        cost_matrix: A 2D NumPy array representing the cost matrix.
                       It must be square (N x N).  If it's not square,
                       the function will pad it with zeros to make it square.

    Returns:
        A tuple containing:
            - A list of tuples, where each tuple (row_index, col_index)
              represents an assignment (i.e., the row is matched to the column).
              The list will contain exactly N tuples, where N is the number of rows
              in the original cost matrix. If a perfect matching isn't possible,
              some rows may be assigned to dummy columns (from padding).
            - The maximum total cost of the assignment.

    Raises:
        ValueError: If the input cost matrix is not 2D.
        TypeError: If the input is not a NumPy array.
    """
    if not isinstance(cost_matrix, np.ndarray):
        raise TypeError("Input cost matrix must be a NumPy array.")
    if cost_matrix.ndim != 2:
        raise ValueError("Input cost matrix must be 2-dimensional.")

    # Make a copy to avoid modifying the original matrix
    cost_matrix = cost_matrix.copy()
    n_rows, n_cols = cost_matrix.shape
    original_n_rows = n_rows  # Store original number of rows

    # Ensure the matrix is square.  Pad with zeros if necessary.
    if n_rows != n_cols:
        max_dim = max(n_rows, n_cols)
        padded_matrix = np.zeros((max_dim, max_dim), dtype=cost_matrix.dtype)
        padded_matrix[:n_rows, :n_cols] = cost_matrix
        cost_matrix = padded_matrix
        n_rows = max_dim
        n_cols = max_dim

    # The Hungarian algorithm is designed for *minimization*.  To use it
    # for a maximization problem, we can either:
    # 1.  Invert the signs of all the costs.
    # 2.  Subtract all costs from the maximum cost in the matrix.
    # We'll use the second approach, as it's numerically more stable.
    max_cost = np.max(cost_matrix)
    cost_matrix = max_cost - cost_matrix

    # Step 1: Reduce the matrix (find minimums in rows and columns)
    # Subtract the minimum value of each row from all elements of the row
    for i in range(n_rows):
        cost_matrix[i, :] -= np.min(cost_matrix[i, :])

    # Subtract the minimum value of each column from all elements of the column
    for j in range(n_cols):
        cost_matrix[:, j] -= np.min(cost_matrix[:, j])

    # Step 2: Cover all zeros with a minimum number of lines
    row_covers = np.zeros(n_rows, dtype=bool)  # Tracks covered rows
    col_covers = np.zeros(n_cols, dtype=bool)  # Tracks covered columns
    n_lines = 0

    while True:
        # Find the maximum number of independent zeros (zeros
        # where no two are in the same row or column).  This is
        # equivalent to finding a maximum matching in a bipartite graph.
        # We'll do this iteratively.
        row_covers.fill(False)
        col_covers.fill(False)
        n_lines = 0
        assignments = []  # List to store the (row, col) assignments

        def find_assignment():
            for r in range(n_rows):
                if not row_covers[r]:
                    for c in range(n_cols):
                        if not col_covers[c] and cost_matrix[r, c] == 0:
                            assignments.append((r, c))
                            row_covers[r] = True
                            col_covers[c] = True
                            return True
            return False

        while find_assignment():
            pass

        n_lines = np.sum(row_covers) + np.sum(col_covers)
        if n_lines >= n_rows:  # Or n_cols, since it's square
            break  # We've found an optimal solution

        # Step 3: Adjust the matrix if we haven't found enough lines
        min_val = np.inf
        for r in range(n_rows):
            if not row_covers[r]:
                for c in range(n_cols):
                    if not col_covers[c]:
                        min_val = min(min_val, cost_matrix[r, c])

        for r in range(n_rows):
            if not row_covers[r]:
                cost_matrix[r, :] -= min_val
        for c in range(n_cols):
            if col_covers[c]:
                cost_matrix[:, c] += min_val

    # Calculate the maximum cost of the optimal assignment
    max_total_cost = 0
    final_assignments = []
    for r, c in assignments:
        max_total_cost += max_cost - cost_matrix[r, c]  # Use original cost
        final_assignments.append((r, c))

    # Ensure we have exactly original_n_rows assignments
    if len(final_assignments) < n_rows:
        # Find unmatched rows:
        assigned_rows = set(r for r, _ in final_assignments)
        unassigned_rows = [r for r in range(n_rows) if r not in assigned_rows]

        # Find already assigned columns
        assigned_cols = set(c for _, c in final_assignments)
        # Find all available columns (including dummy ones from padding)
        available_cols = list(range(n_cols))  # Use n_cols, not original
        unassigned_cols = [c for c in available_cols if c not in assigned_cols]

        if len(unassigned_cols) >= len(unassigned_rows):
            unassigned_cols = unassigned_cols[: len(unassigned_rows)]

        if len(unassigned_cols) == len(unassigned_rows):
            for rix, cix in zip(unassigned_rows, unassigned_cols):
                final_assignments.append((rix, cix))

    dict_assignments = {}
    for assignment in final_assignments:
        if assignment[0] not in dict_assignments.keys():
            dict_assignments[assignment[0]] = assignment[1]

    return final_assignments, dict_assignments


########### Crop function
def crop_face(faces, facer_image, idx):
    """
    Crops a face from the input image tensor based on detected face bounding boxes.

    Args:
        faces (dict): Dictionary containing face detection results, including 'rects'.
        facer_image (torch.Tensor): The input image tensor in BCHW format.
        idx (int): Index of the face to crop.

    Returns:
        torch.Tensor: Cropped and resized face image tensor (112x112).
    """
    if len(faces.keys()) > 0 and faces["rects"].nelement() != 0:
        rects = torch.clamp(faces["rects"][idx], min=0)
        rects = rects.long()
        image_cropped = (
            facer_image[:, :, rects[1] : rects[3], rects[0] : rects[2]] / 255
        )
    else:
        image_cropped = facer_image / 255
    image_cropped = F.interpolate(image_cropped, size=(112, 112))
    return image_cropped


########### Gathering scores
def gather_and_print_scores(
    accuracy, face_sim, hps, split_metrics, total, total_people
):
    """
    Computes and prints overall and per-group evaluation metrics for multi-human generation.

    Args:
        accuracy (float): Total count of correct face count matches.
        face_sim (float): Cumulative face similarity score.
        hps (float): Cumulative Human Preference Score.
        split_metrics (dict): Dictionary mapping number of people to [HPS, face_sim, accuracy].
        total (int): Total number of evaluated samples.
        total_people (dict): Dictionary mapping number of people to sample counts.
    """
    accuracy /= total
    face_sim /= total
    hps /= total

    print(f"Count Accuracy: {accuracy}, Face Similarity: {face_sim}, HPS: {hps}")

    for num_people in range(1, 6):
        if total_people[num_people]==0:
            print(f"Not enough samples for {num_people} people.")
        print(
            f"People={num_people} Count Accuracy: {split_metrics[num_people][2]/(total_people[num_people])}, "
            f"Face Similarity: {split_metrics[num_people][1]/(total_people[num_people])}, "
            f"HPS: {split_metrics[num_people][0]/(total_people[num_people])}"
        )
