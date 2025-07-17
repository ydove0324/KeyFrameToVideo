import argparse
import os
import json
import glob
from typing import Dict, Any

import datasets  # HuggingFace Datasets supports streaming WebDataset
import webdataset as wds
from tqdm import tqdm


def load_flow_scores(path: str, threshold: float):
    """Return a set of sample keys whose flow score is above the threshold."""
    with open(path, "r") as f:
        scores: Dict[str, float] = json.load(f)
    # Keep only the keys we need
    keep_keys = {k for k, v in scores.items() if v > threshold}
    print(f"Loaded {len(scores)} flow scores → {len(keep_keys)} keys with score > {threshold}")
    return keep_keys


def filter_webdataset(dataset_dir: str, flow_json: str, out_dir: str, threshold: float = 20.0,
                       maxcount: int = 10_000, maxsize: float = 10e9):
    """Filter `dataset_dir` shards by flow score and write a new WebDataset in `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load flow scores and build lookup
    keep_keys = load_flow_scores(flow_json, threshold)
    if not keep_keys:
        print("[WARN] No keys satisfy the threshold. Nothing to do.")
        return

    # 2. Prepare input shards list
    shard_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.tar")))
    if not shard_paths:
        raise FileNotFoundError(f"No .tar shards found in {dataset_dir}")
    print(f"Found {len(shard_paths)} shards")

    # 3. Build streaming dataset (iterator yields raw bytes)
    ds = datasets.load_dataset(
        "webdataset",
        data_files={"train": shard_paths},
        split="train",
        streaming=True,
    )

    # 4. Set up ShardWriter for output shards
    sink = wds.ShardWriter(os.path.join(out_dir, "dataset_%06d.tar"), maxcount=maxcount, maxsize=maxsize)

    kept = 0
    total = 0
    for sample in tqdm(ds, desc="Filtering"):
        total += 1
        key = sample["__key__"]
        if key not in keep_keys:
            continue

        # HF dataset returns Python objects; ensure JSON is str/bytes and other binary fields stay bytes.
        out_sample: Dict[str, Any] = {"__key__": key}
        for k, v in sample.items():
            if k == "__key__" or k == "__url__":
                continue  # Skip duplicate key/__url__ in output
            if k == "json":
                # Ensure JSON is a string of bytes
                if isinstance(v, (dict, list)):
                    v = json.dumps(v)
            out_sample[k] = v
        sink.write(out_sample)
        kept += 1

    sink.close()
    print(f"Completed: kept {kept}/{total} samples (threshold={threshold}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a WebDataset by optical flow score.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing input .tar shards")
    parser.add_argument("--flow_json", type=str, required=True, help="Path to flow_scores.json file")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to write filtered shards")
    parser.add_argument("--threshold", type=float, default=20.0, help="Keep samples with flow_score > threshold")
    parser.add_argument("--maxcount", type=int, default=10_000, help="Max samples per output shard")
    parser.add_argument("--maxsize", type=float, default=10e9, help="Max shard size in bytes (approx.)")
    args = parser.parse_args()

    filter_webdataset(
        dataset_dir=args.dataset_dir,
        flow_json=args.flow_json,
        out_dir=args.out_dir,
        threshold=args.threshold,
        maxcount=args.maxcount,
        maxsize=args.maxsize,
    ) 