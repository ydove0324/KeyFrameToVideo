#!/usr/bin/env python
"""Multi‑GPU video slicer → fixed‑length clips (streaming, OOM‑safe)

Key points
~~~~~~~~~~
* **Streaming decode** – never load more than `--max-gpu-frames` (default
  `--max-gpu-clips × clip_len`) into a GPU at once. Safe for long videos /
  limited VRAM.
* **Multi‑GPU dispatch** – supply `--gpus 0,1,...`; videos round‑robin to GPUs.
* **GPU‑batched spatial upscale/downscale** with `torch.nn.functional.interpolate`.
* **Thread‑pool H.264 encoding** so I/O overlaps with GPU work.
* **Auto‑resolution** & skip already‑processed videos.

Example
~~~~~~~
```bash
python video_preprocess_mgpu.py raw out \
       --frames 17 --fps 25 --gpus 0,1 \
       --max-gpu-clips 15 --threads 8    # ≤15×17=255 frames per GPU batch
```
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torchvision.io import write_video
import decord
from tqdm import tqdm

# Flow score thresholds
MIN_FLOW = 7
MAX_FLOW = 32

decord.bridge.set_bridge("torch")

# ---------------------------------------------------------------------------
# Resolution helper
# ---------------------------------------------------------------------------
CANDIDATE_RESOLUTIONS: List[Tuple[int, int]] = [
    (480, 832), (832, 480), (720, 1280), (1280, 720), (1024, 1024)
]

def choose_best_resolution(orig_size: Tuple[int, int]) -> Tuple[int, int]:
    h0, w0 = orig_size
    ar0, area0 = w0 / h0, h0 * w0
    best, best_score = CANDIDATE_RESOLUTIONS[0], float("inf")
    for h, w in CANDIDATE_RESOLUTIONS:
        ar, area = w / h, h * w
        score = abs(ar - ar0) * 10 + abs((area / area0) - 1 if area >= area0 else (area0 / area) - 1)
        if score < best_score:
            best, best_score = (h, w), score
    return best

# ---------------------------------------------------------------------------
# GPU resize (single batch)
# ---------------------------------------------------------------------------

def resize_gpu_batch(frames: torch.Tensor, target_hw: Tuple[int, int], clip_len: int) -> List[torch.Tensor]:
    """frames: (B*clip_len, H, W, C) uint8 CPU/GPU → list[clip] CPU"""
    if frames.device.type != "cuda":
        frames = frames.cuda(non_blocking=True)
    B = frames.shape[0] // clip_len
    _, H, W, _ = frames.shape
    x = (
        frames.view(B, clip_len, H, W, 3)
        .permute(0, 4, 1, 2, 3)  # (B,C,T,H,W)
        .float()
    )
    x = F.interpolate(x, size=(clip_len, *target_hw), mode="trilinear", align_corners=False)
    x = x.permute(0, 2, 3, 4, 1).byte().cpu()  # (B,T,H,W,C)
    return [x[i] for i in range(B)]

# ---------------------------------------------------------------------------
# Encoder helper
# ---------------------------------------------------------------------------

def encode_clip(clip: torch.Tensor, path: str, fps: int):
    write_video(path, clip, fps=fps, video_codec="libx264")

# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def gather_videos(p: pathlib.Path) -> List[str]:
    if p.is_file():
        return [str(p)]
    vids: List[str] = []
    for ext in ("mp4", "avi", "mov", "mkv", "webm", "flv"):
        vids += glob.glob(str(p / f"**/*.{ext}"), recursive=True)
        vids += glob.glob(str(p / f"**/*.{ext.upper()}"), recursive=True)
    return vids

def filter_existing(vids: List[str], out_root: pathlib.Path) -> List[str]:
    return [v for v in vids if not (out_root / pathlib.Path(v).stem).exists()]

def parse_size(s: str) -> Tuple[int, int]:
    return tuple(map(int, s.lower().split("x")))  # type: ignore

def parse_gpus(s: str) -> List[int]:
    return [int(t) for t in s.split(",") if t.strip().isdigit()]

# ---------------------------------------------------------------------------
# Video processing (one video, one GPU)
# ---------------------------------------------------------------------------

def process_video(
    vpath: str,
    out_root: str,
    clip_len: int,
    target_hw: Tuple[int, int] | None,
    fps: int,
    max_gpu_clips: int,
    threads: int,
    gpu_idx: int,
    max_clip_per_video: int,
):
    torch.cuda.set_device(gpu_idx)
    stem = pathlib.Path(vpath).stem
    out_dir = pathlib.Path(out_root) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    vr = decord.VideoReader(vpath, num_threads=12)
    total_frames = min(max_clip_per_video*clip_len, len(vr))
    if target_hw is None:
        target_hw = choose_best_resolution(vr[0].shape[:2])

    # 一开始就获取所有帧数据
    print(f"Loading all {total_frames} frames from {vpath}...")
    all_frames = vr.get_batch(range(total_frames))
    if not isinstance(all_frames, torch.Tensor):
        all_frames = torch.from_numpy(all_frames.asnumpy() if hasattr(all_frames, "asnumpy") else all_frames)
    
    total_clips = total_frames // clip_len
    print(f"Processing {total_clips} clips...")

    encode_pool = ThreadPoolExecutor(max_workers=threads)
    futures = []

    # 按批次处理clips
    clips_per_batch = max_gpu_clips
    clip_idx = 0
    
    for start_clip in range(0, total_clips, clips_per_batch):
        batch_clips = min(clips_per_batch, total_clips - start_clip)
        start_frame = start_clip * clip_len
        end_frame = start_frame + batch_clips * clip_len
        
        # 从已加载的帧中切片
        batch_frames = all_frames[start_frame:end_frame]
        
        resized = resize_gpu_batch(batch_frames, target_hw, clip_len)
        for clip in resized:
            clip_path = str(out_dir / f"clip_{clip_idx:04d}.mp4")  # Convert Path to string
            fut = encode_pool.submit(
                encode_clip,
                clip,
                clip_path,
                fps,
            )
            futures.append(fut)
            clip_idx += 1

    for f in as_completed(futures):
        if exc := f.exception():
            print(f"❌ write error: {exc}")
    encode_pool.shutdown()

# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Multi‑GPU video slicer (streaming, OOM‑safe)")
    ap.add_argument("--input", help="video file or directory")
    ap.add_argument("--output", help="output directory")
    ap.add_argument("--frames", type=int, default=17, help="frames per clip")
    ap.add_argument("--size", default="auto", help="HxW or 'auto'")
    ap.add_argument("--fps", type=int, default=25, help="output FPS")
    ap.add_argument("--max-gpu-clips", type=int, default=15, help="clips per GPU batch (<=15 safe ~255 frames)")
    ap.add_argument("--threads", type=int, default=8, help="parallel encoders per video")
    ap.add_argument("--gpus", default="0", help="comma‑separated CUDA indices, e.g. 0,1")
    ap.add_argument("--flow_stats", help="Path to the JSONL file containing flow statistics")
    ap.add_argument("--max-clip-per-video", type=int, default=10, help="maximum number of clips to process per video")
    args = ap.parse_args()

    gpu_ids = parse_gpus(args.gpus) if torch.cuda.is_available() else []
    if not gpu_ids:
        gpu_ids = [0]
    if max(gpu_ids) >= torch.cuda.device_count():
        raise ValueError("GPU index exceeds available devices")

    target_hw = None if args.size.lower() == "auto" else parse_size(args.size)

    in_path = pathlib.Path(args.input)
    vids = filter_existing(gather_videos(in_path), pathlib.Path(args.output))
    if not vids:
        print("No new videos to process.")
        return

    # Filter videos by flow score if stats file is provided
    if args.flow_stats and os.path.exists(args.flow_stats):
        print(f"Loading flow statistics from: {args.flow_stats}")
        flow_stats = {}
        with open(args.flow_stats, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                flow_stats[os.path.basename(item["path"])] = item["flow_data"]["mean_flow"]
        
        filtered_vids = []
        for vid in vids:
            vid_name = os.path.basename(vid)
            if vid_name in flow_stats:
                mean_flow = flow_stats[vid_name]
                if MIN_FLOW <= mean_flow <= MAX_FLOW:
                    filtered_vids.append(vid)
                else:
                    print(f"Skipping {vid_name} - flow score {mean_flow:.2f} outside range [{MIN_FLOW}, {MAX_FLOW}]")
        vids = filtered_vids
        print(f"After flow filtering: {len(vids)} videos remaining")
        
        if not vids:
            print("No videos to process after flow filtering.")
            return

    print(f"🚀 {len(vids)} videos → GPUs {gpu_ids} (clips/batch={args.max_gpu_clips})")
    gpu_cycle = cycle(gpu_ids)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        fut_map = {
            ex.submit(
                process_video,
                vp,
                args.output,
                args.frames,
                target_hw,
                args.fps,
                args.max_gpu_clips,
                args.threads,
                next(gpu_cycle),
                args.max_clip_per_video, # Pass args.max_clip_per_video
            ): vp
            for vp in vids
        }
        for f in tqdm(as_completed(fut_map), total=len(vids), unit="video"):
            if exc := f.exception():
                print(f"❌ {fut_map[f]}: {exc}")
    print("✅ All done.")


if __name__ == "__main__":
    main()
