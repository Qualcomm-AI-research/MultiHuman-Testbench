# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.dwpose_utils import draw_bodypose, draw_handpose, draw_facepose


def draw_pose(pose: Dict[str, Any], H: int, W: int) -> np.ndarray:
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = draw_bodypose(canvas, candidate, subset)
    canvas = draw_handpose(canvas, hands)
    canvas = draw_facepose(canvas, faces)

    return canvas


# ---------- Helpers ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def compute_canvas_size_from_pose(
    pose: Dict[str, Any],
    fallback: Optional[Tuple[int, int]] = None
) -> Tuple[int, int]:
    """
    Determine canvas size (W, H).
    Priority:
      1) Use explicit pose['width']/pose['height'] if present.
      2) Compute from bodies['candidate'] extents (x,y) with padding.
      3) Use fallback if given, else 1024x1024.
    """
    # Explicit size provided?
    for w_key, h_key in (("width", "height"), ("image_width", "image_height")):
        if w_key in pose and h_key in pose:
            try:
                W = int(pose[w_key])
                H = int(pose[h_key])
                if W > 0 and H > 0:
                    return W, H
            except Exception:
                pass

    # Compute from candidate x,y if available
    try:
        bodies = pose.get("bodies", {})
        candidate = bodies.get("candidate", None)
        if candidate is not None:
            xs, ys = [], []

            # candidate can be:
            # - list of [x,y,score,id] rows
            # - numpy-like list of lists
            # - flat list (rare): [x,y,c,id, x,y,c,id, ...]
            if isinstance(candidate, list):
                if len(candidate) > 0 and isinstance(candidate[0], (list, tuple)):
                    # list of rows
                    for row in candidate:
                        if len(row) >= 2:
                            xs.append(float(row[0]))
                            ys.append(float(row[1]))
                else:
                    # fallback: assume flat 4-step structure
                    for i in range(0, len(candidate), 4):
                        if i + 1 < len(candidate):
                            xs.append(float(candidate[i]))
                            ys.append(float(candidate[i+1]))

            if xs and ys:
                pad = 20
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                W = int(np.ceil(maxx - minx + 2 * pad))
                H = int(np.ceil(maxy - miny + 2 * pad))
                # Ensure reasonable bounds
                if W <= 0 or H <= 0:
                    raise ValueError("Non-positive canvas from extents")
                # Also store offsets so draw_bodypose can render in view if you later add shifting logic.
                # For now, draw_* utilities typically expect original coordinates already within canvas.
                return max(W, 64), max(H, 64)
    except Exception:
        pass

    # Fallback
    if fallback is not None:
        return fallback
    return 1024, 1024


def normalize_pose_entry(val: Any) -> List[Dict[str, Any]]:
    """
    Ensure we always iterate a list of 'pose' dicts.
    - If val is a dict -> [val]
    - If val is a list -> filter dicts
    - Else -> []
    """
    if isinstance(val, dict):
        return [val]
    if isinstance(val, list):
        return [v for v in val if isinstance(v, dict)]
    return []


# ---------- Main pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Render DWpose images from a dumped JSON.")
    ap.add_argument("--json", required=True, help="Path to the DWpose merged JSON (dumped earlier).")
    ap.add_argument("--out_dir", required=True, help="Directory to save pose images.")
    ap.add_argument("--prefix", default="", help="Optional prefix to add to output filenames.")
    ap.add_argument("--fallback_canvas", nargs=2, type=int, default=None,
                    help="Fallback canvas size W H if JSON lacks size and extents can't be computed.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    fallback = tuple(args.fallback_canvas) if args.fallback_canvas else None

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    for key, val in data.items():
        entries = normalize_pose_entry(val)
        if not entries:
            print(f"[WARN] Skipping '{key}': expected dict or list-of-dicts.")
            continue

        for i, pose in enumerate(entries):
            # Basic validation for required fields
            if not isinstance(pose, dict) or "bodies" not in pose:
                print(f"[WARN] {key}[{i}] missing 'bodies' field; skipping.")
                continue
            bodies = pose.get("bodies", {})
            if "candidate" not in bodies or "subset" not in bodies:
                print(f"[WARN] {key}[{i}] 'bodies' lacks 'candidate' or 'subset'; skipping.")
                continue

            # Ensure 'faces' and 'hands' exist (even if empty), as your draw_pose expects them
            pose.setdefault("faces", [])
            pose.setdefault("hands", [])

            # Determine canvas size
            W, H = compute_canvas_size_from_pose(pose, fallback=fallback)

            try:
                canvas = draw_pose(pose, 1024, 1024)  # Your renderer
            except Exception as e:
                print(f"[ERROR] Failed rendering {key}[{i}]: {e}")
                continue

            suffix = ""
            out_name = f"{args.prefix}{key}{suffix}.png"
            out_path = os.path.join(args.out_dir, out_name)

            plt.imsave(out_path, canvas)

    print(f"[DONE] Saved {total} image(s) to {args.out_dir}")


if __name__ == "__main__":
    main()