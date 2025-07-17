#!/usr/bin/env python3
"""build_imagenet_validation_json.py

Generate an "imagenet_validation.json" file similar in structure to the provided
`validation.json`, but for ImageNet validation images.

Requirements (as per user request):
1. Read the file list located at
   `/share/project/datasets/ImageNet/val/filelist.txt`.
2. Count the total number of lines (images) and print the count.
3. Randomly select **up to 1000** images.
4. For each selected image, create a JSON entry with the following keys:
   - image_path: absolute path to the image file
   - num_inference_steps: 50 (fixed)
   - height: 480
   - width: 480
   - num_frames: 1
   - key_frames_indices: [[0]]
5. Write the output JSON to `imagenet_validation.json` in the same directory as
   this script.

Usage:
    python build_imagenet_validation_json.py
"""

import json
import random
from pathlib import Path

FILELIST_PATH = Path("/share/project/datasets/ImageNet/val/filelist.txt")
BASE_DIR = FILELIST_PATH.parent  # /share/project/datasets/ImageNet/val
OUTPUT_JSON =  Path("data/imagenet/imagenet_validation.json")

# For reproducibility you may want to fix a seed
random.seed(42)


def main() -> None:
    if not FILELIST_PATH.is_file():
        raise FileNotFoundError(f"File list not found: {FILELIST_PATH}")

    # Read all relative image paths
    with FILELIST_PATH.open("r", encoding="utf-8") as fp:
        relative_paths = [line.strip() for line in fp if line.strip()]

    total_images = len(relative_paths)
    print(f"Total images found: {total_images}")

    # Sample up to 1000 images
    sample_size = min(1000, total_images)
    sampled_paths = random.sample(relative_paths, sample_size)
    print(f"Randomly selected {sample_size} images for validation set.")

    # Build JSON structure
    data_entries = []
    for rel_path in sampled_paths:
        abs_path = str((BASE_DIR / rel_path).resolve())
        entry = {
            "image_path": abs_path,
            "num_inference_steps": 50,
            "height": 480,
            "width": 480,
            "num_frames": 1,
            "key_frames_indices": [[0]],
        }
        data_entries.append(entry)

    output_obj = {"data": data_entries}

    with OUTPUT_JSON.open("w", encoding="utf-8") as fp:
        json.dump(output_obj, fp, ensure_ascii=False, indent=2)
    print(f"Validation JSON written to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main() 