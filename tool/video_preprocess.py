import argparse
import os
import pathlib
from typing import List, Tuple, Union
import numpy as np
import torch
import cv2
from tqdm import tqdm
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

import decord
decord.bridge.set_bridge("torch")

def resize_video_frames(frames: Union[torch.Tensor, np.ndarray], target_size: Tuple[int, int], device: str = "cpu", use_torch: bool = False) -> torch.Tensor:
    """
    Resize video frames to target size
    Args:
        frames: torch.Tensor or NDArray of shape (T, H, W, C)
        target_size: (height, width)
        device: "cpu" or "cuda"
        use_torch: whether to use torch operations (GPU-friendly) or cv2 (CPU-optimized)
    Returns:
        resized frames: torch.Tensor of shape (T, target_H, target_W, C)
    """
    # Convert to torch.Tensor if input is NDArray
    if not isinstance(frames, torch.Tensor):
        frames = torch.from_numpy(frames.asnumpy() if hasattr(frames, 'asnumpy') else np.array(frames))
    
    T, H, W, C = frames.shape
    target_H, target_W = target_size
    
    if use_torch and device == "cuda" and torch.cuda.is_available():
        # GPU-accelerated resizing using torch
        frames = frames.to(device)
        # Permute to (T, C, H, W) for torch operations
        frames_permuted = frames.permute(0, 3, 1, 2).float()
        resized_frames = torch.nn.functional.interpolate(
            frames_permuted, 
            size=(target_H, target_W), 
            mode='bilinear', 
            align_corners=False
        )
        # Permute back to (T, H, W, C)
        resized_frames = resized_frames.permute(0, 2, 3, 1).to(torch.uint8)
        return resized_frames.cpu()
    else:
        # CPU-optimized resizing using cv2
        frames_np = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy() if hasattr(frames, 'asnumpy') else np.array(frames)
        frames_np = frames_np.astype(np.uint8)
        resized_frames = []
        
        for i in range(T):
            frame = frames_np[i]
            resized_frame = cv2.resize(frame, (target_W, target_H), interpolation=cv2.INTER_LANCZOS4)
            resized_frames.append(resized_frame)
        
        resized_frames = np.stack(resized_frames, axis=0)
        return torch.from_numpy(resized_frames)

def extract_video_clips(video_path: str, clip_length: int = 17, overlap: int = 1, device: str = "cpu") -> List[torch.Tensor]:
    """
    Extract video clips with specified length and overlap
    Args:
        video_path: path to video file
        clip_length: number of frames per clip
        overlap: number of overlapping frames between adjacent clips
        device: "cpu" or "cuda"
    Returns:
        list of video clips, each clip is torch.Tensor of shape (clip_length, H, W, C)
    """
    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        
        if total_frames < clip_length:
            print(f"Warning: Video {video_path} has only {total_frames} frames, less than required {clip_length}")
            # If video is too short, duplicate frames to reach clip_length
            all_frames = vr.get_batch(list(range(total_frames)))
            # Convert to torch.Tensor if needed
            if not isinstance(all_frames, torch.Tensor):
                all_frames = torch.from_numpy(all_frames.asnumpy() if hasattr(all_frames, 'asnumpy') else np.array(all_frames))
                
            if total_frames == 1:
                # Single frame video, repeat the frame
                frames = all_frames[0:1].repeat(clip_length, 1, 1, 1)
                return [frames]
            else:
                # Duplicate the last frame to reach clip_length
                last_frame = all_frames[-1:]
                padding_frames = last_frame.repeat(clip_length - total_frames, 1, 1, 1)
                frames = torch.cat([all_frames, padding_frames], dim=0)
                return [frames]
        
        clips = []
        step = clip_length - overlap
        
        for start_idx in range(0, total_frames - clip_length + 1, step):
            end_idx = start_idx + clip_length
            frame_indices = list(range(start_idx, end_idx))
            clip = vr.get_batch(frame_indices)
            # Convert to torch.Tensor if needed
            if not isinstance(clip, torch.Tensor):
                clip = torch.from_numpy(clip.asnumpy() if hasattr(clip, 'asnumpy') else np.array(clip))
            clips.append(clip)
        
        # Handle the last clip if there are remaining frames
        if (total_frames - clip_length) % step != 0:
            start_idx = total_frames - clip_length
            frame_indices = list(range(start_idx, total_frames))
            clip = vr.get_batch(frame_indices)
            # Convert to torch.Tensor if needed
            if not isinstance(clip, torch.Tensor):
                clip = torch.from_numpy(clip.asnumpy() if hasattr(clip, 'asnumpy') else np.array(clip))
            clips.append(clip)
            
        return clips
        
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return []

def save_video_clip(clip: Union[torch.Tensor, np.ndarray], output_path: str, fps: int = 25):
    """
    Save video clip to file
    Args:
        clip: torch.Tensor or NDArray of shape (T, H, W, C)
        output_path: output file path
        fps: frames per second
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    T, H, W, C = clip.shape
    # Convert to numpy array regardless of input type
    if isinstance(clip, torch.Tensor):
        clip_np = clip.numpy().astype(np.uint8)
    else:
        clip_np = clip.asnumpy().astype(np.uint8) if hasattr(clip, 'asnumpy') else np.array(clip).astype(np.uint8)
    
    # Use cv2.VideoWriter to save video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for i in range(T):
        frame = clip_np[i]
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()

def get_device_info():
    """Get available device information"""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU available: {device_name} ({memory_gb:.1f}GB)")
    else:
        device = "cpu"
        cpu_count = multiprocessing.cpu_count()
        print(f"💻 Using CPU with {cpu_count} cores")
    
    return device

def get_resolution_name(resolution: Tuple[int, int]) -> str:
    """Get a name for the resolution"""
    h, w = resolution
    if (h, w) == (480, 832):
        return "480x832"
    elif (h, w) == (832, 480):
        return "832x480"
    elif (h, w) == (720, 1280):
        return "720x1280"
    elif (h, w) == (1280, 720):
        return "1280x720"
    elif (h, w) == (1024, 1024):
        return "1024x1024"
    else:
        return f"{h}x{w}"

def choose_best_resolution(original_size: Tuple[int, int], candidate_resolutions: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Choose the best resolution from candidates based on original video size
    Args:
        original_size: (height, width) of original video
        candidate_resolutions: list of (height, width) candidate resolutions
    Returns:
        best matching resolution: (height, width)
    """
    orig_h, orig_w = original_size
    orig_aspect_ratio = orig_w / orig_h
    orig_area = orig_h * orig_w
    
    best_resolution = candidate_resolutions[0]
    best_score = float('inf')
    
    for target_h, target_w in candidate_resolutions:
        target_aspect_ratio = target_w / target_h
        target_area = target_h * target_w
        
        # Calculate aspect ratio difference
        aspect_diff = abs(orig_aspect_ratio - target_aspect_ratio)
        
        # Calculate area ratio (prefer similar or slightly larger area)
        area_ratio = target_area / orig_area
        if area_ratio < 1:
            area_score = 1 / area_ratio  # Penalize downscaling heavily
        else:
            area_score = area_ratio  # Moderate penalty for upscaling
        
        # Combined score: prioritize aspect ratio match, then area similarity
        score = aspect_diff * 10 + abs(area_score - 1)
        
        if score < best_score:
            best_score = score
            best_resolution = (target_h, target_w)
    
    return best_resolution

def process_single_clip_resize(args):
    """Helper function for multiprocessing clip resizing"""
    clip, resolution, device, use_torch = args
    return resize_video_frames(clip, resolution, device, use_torch)

def process_clips_multithread(clips: List[torch.Tensor], 
                            resolution: Tuple[int, int], 
                            device: str = "cpu", 
                            use_torch: bool = False, 
                            num_workers: int = None) -> List[torch.Tensor]:
    """Process multiple clips with multithreading"""
    if num_workers is None:
        num_workers = min(len(clips), multiprocessing.cpu_count())
    
    if num_workers == 1 or len(clips) == 1:
        # Single-threaded processing
        return [resize_video_frames(clip, resolution, device, use_torch) for clip in clips]
    
    # Multi-threaded processing
    resized_clips = [None] * len(clips)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(resize_video_frames, clip, resolution, device, use_torch): i 
            for i, clip in enumerate(clips)
        }
        
        # Collect results
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                resized_clips[index] = future.result()
            except Exception as e:
                print(f"Error processing clip {index}: {str(e)}")
                resized_clips[index] = clips[index]  # Use original clip as fallback
    
    return resized_clips

def preprocess_video_file(
    video_path: str,
    output_dir: str,
    target_resolutions: List[Tuple[int, int]],
    clip_length: int = 17,
    overlap: int = 1,
    fps: int = 25,
    device: str = "auto",
    use_torch: bool = False,
    num_workers: int = None
):
    """
    Preprocess a single video file
    Args:
        video_path: path to input video
        output_dir: output directory
        target_resolutions: list of (height, width) tuples
        clip_length: number of frames per clip
        overlap: overlap between clips
        fps: output fps
        device: "cpu", "cuda", or "auto" for automatic detection
        use_torch: whether to use torch operations for resizing (GPU-friendly)
        num_workers: number of worker threads for parallel processing
    """
    video_name = pathlib.Path(video_path).stem
    
    # Auto-detect device
    if device == "auto":
        device = get_device_info()
    elif device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # Set default num_workers based on device
    if num_workers is None:
        if device == "cuda":
            num_workers = 2  # Conservative for GPU to avoid memory issues
        else:
            num_workers = min(multiprocessing.cpu_count(), 8)  # Limit to avoid too many threads
    
    print(f"🎬 Processing video: {video_path}")
    print(f"⚙️  Device: {device.upper()}, Workers: {num_workers}, Torch resize: {use_torch}")
    
    # Get original video resolution
    try:
        vr = decord.VideoReader(video_path)
        original_height, original_width = vr[0].shape[:2]
        original_size = (original_height, original_width)
        print(f"📐 Original resolution: {original_width}x{original_height}")
    except Exception as e:
        print(f"❌ Error reading video resolution: {str(e)}")
        return
    
    # Choose the best matching resolution
    best_resolution = choose_best_resolution(original_size, target_resolutions)
    resolution_name = get_resolution_name(best_resolution)
    print(f"🎯 Selected best matching resolution: {resolution_name} {best_resolution}")
    
    # Extract clips
    clips = extract_video_clips(video_path, clip_length, overlap, device)
    
    if not clips:
        print(f"❌ No clips extracted from {video_path}")
        return
    
    print(f"✂️  Extracted {len(clips)} clips from {video_name}")
    
    # Process the selected resolution
    resolution = best_resolution
    resolution_name = get_resolution_name(resolution)
    resolution_dir = os.path.join(output_dir, resolution_name, video_name)
    
    print(f"🔄 Processing resolution {resolution_name} with {num_workers} workers...")
    
    # Multi-threaded clip resizing
    with tqdm(total=len(clips), desc=f"Resizing clips for {resolution_name}") as pbar:
        resized_clips = process_clips_multithread(
            clips, resolution, device, use_torch, num_workers
        )
        pbar.update(len(clips))
    
    # Save clips
    print(f"💾 Saving {len(resized_clips)} clips...")
    for i, resized_clip in enumerate(tqdm(resized_clips, desc=f"Saving clips for {resolution_name}")):
        output_path = os.path.join(resolution_dir, f"clip_{i:04d}.mp4")
        save_video_clip(resized_clip, output_path, fps)
    
    # Save metadata
    metadata = {
        "original_video": video_path,
        "original_resolution": original_size,
        "clip_length": clip_length,
        "overlap": overlap,
        "target_resolution": resolution,
        "num_clips": len(clips),
        "fps": fps,
        "device": device,
        "use_torch": use_torch,
        "num_workers": num_workers
    }
    metadata_path = os.path.join(resolution_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Completed {resolution_name} - saved to {resolution_dir}")

def preprocess_videos(
    input_dir: str,
    output_dir: str,
    target_resolutions: List[Tuple[int, int]] = None,
    clip_length: int = 17,
    overlap: int = 1,
    fps: int = 25,
    video_extensions: List[str] = None,
    device: str = "auto",
    use_torch: bool = False,
    num_workers: int = None,
    max_concurrent_videos: int = 1
):
    """
    Preprocess all videos in a directory
    Args:
        input_dir: input directory containing videos
        output_dir: output directory
        target_resolutions: list of (height, width) tuples
        clip_length: number of frames per clip
        overlap: overlap between clips
        fps: output fps
        video_extensions: supported video file extensions
        device: "cpu", "cuda", or "auto" for automatic detection
        use_torch: whether to use torch operations for resizing (GPU-friendly)
        num_workers: number of worker threads for parallel processing per video
        max_concurrent_videos: maximum number of videos to process simultaneously
    """
    if target_resolutions is None:
        target_resolutions = [
            (480, 832),   # Portrait 480x832
            (832, 480),   # Landscape 832x480
            (720, 1280),  # Portrait 720x1280
            (1280, 720),  # Landscape 1280x720
            (1024, 1024)  # Square 1024x1024
        ]
    
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    input_path = pathlib.Path(input_dir)
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.rglob(f"*{ext}"))
        video_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    print(f"📁 Found {len(video_files)} video files")
    print(f"🏭 Processing with max {max_concurrent_videos} concurrent videos")
    
    # Process videos concurrently
    def process_single_video(video_file):
        try:
            preprocess_video_file(
                str(video_file),
                output_dir,
                target_resolutions,
                clip_length,
                overlap,
                fps,
                device,
                use_torch,
                num_workers
            )
            return str(video_file), True, None
        except Exception as e:
            error_msg = f"Error processing {video_file}: {str(e)}"
            print(f"❌ {error_msg}")
            return str(video_file), False, error_msg
    
    if max_concurrent_videos == 1:
        # Sequential processing
        for video_file in video_files:
            process_single_video(video_file)
    else:
        # Concurrent processing
        with ThreadPoolExecutor(max_workers=max_concurrent_videos) as executor:
            # Submit all video processing tasks
            future_to_video = {
                executor.submit(process_single_video, video_file): video_file 
                for video_file in video_files
            }
            
            # Track results
            successful = 0
            failed = 0
            
            # Process completed tasks
            for future in tqdm(as_completed(future_to_video), total=len(video_files), desc="Processing videos"):
                video_file = future_to_video[future]
                try:
                    video_path, success, error = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"❌ Unexpected error with {video_file}: {str(e)}")
                    failed += 1
            
            print(f"\n📊 Processing complete: {successful} successful, {failed} failed")

def main():
    parser = argparse.ArgumentParser(description="Preprocess videos into clips with different resolutions (CPU/GPU + Multi-threading)")
    parser.add_argument("--input_dir", type=str, help="Input directory containing videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--clip_length", type=int, default=17, help="Number of frames per clip")
    parser.add_argument("--overlap", type=int, default=1, help="Number of overlapping frames")
    parser.add_argument("--fps", type=int, default=25, help="Output FPS")
    parser.add_argument("--single_video", type=str, help="Process a single video file instead of directory")
    
    # Performance options
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                       help="Processing device: auto (detect), cpu, or cuda")
    parser.add_argument("--use_torch", action="store_true", 
                       help="Use torch operations for resizing (GPU-friendly, slower on CPU)")
    parser.add_argument("--num_workers", type=int, default=None, 
                       help="Number of worker threads for parallel clip processing")
    parser.add_argument("--max_concurrent_videos", type=int, default=1, 
                       help="Maximum number of videos to process simultaneously")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.single_video and not args.input_dir:
        parser.error("Either --single_video or --input_dir must be specified")
    
    # Define target resolutions
    target_resolutions = [
        (480, 832),   # Portrait 480x832
        (832, 480),   # Landscape 832x480
        (720, 1280),  # Portrait 720x1280
        (1280, 720),  # Landscape 1280x720
        (1024, 1024)  # Square 1024x1024
    ]
    
    # Print configuration
    print("🚀 Video Preprocessing Configuration:")
    print(f"   Device: {args.device}")
    print(f"   Use Torch: {args.use_torch}")
    print(f"   Workers per video: {args.num_workers or 'auto'}")
    print(f"   Max concurrent videos: {args.max_concurrent_videos}")
    print(f"   Clip length: {args.clip_length}")
    print(f"   Overlap: {args.overlap}")
    print(f"   Output FPS: {args.fps}")
    print()

    if args.single_video:
        # Process single video
        preprocess_video_file(
            args.single_video,
            args.output_dir,
            target_resolutions,
            args.clip_length,
            args.overlap,
            args.fps,
            args.device,
            args.use_torch,
            args.num_workers
        )
    else:
        # Process directory
        preprocess_videos(
            args.input_dir,
            args.output_dir,
            target_resolutions,
            args.clip_length,
            args.overlap,
            args.fps,
            video_extensions=None,
            device=args.device,
            use_torch=args.use_torch,
            num_workers=args.num_workers,
            max_concurrent_videos=args.max_concurrent_videos
        )
    
    print("🎉 Preprocessing completed!")

if __name__ == "__main__":
    main() 