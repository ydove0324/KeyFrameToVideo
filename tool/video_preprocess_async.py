import argparse
import os
import pathlib
from typing import List, Tuple, Union, Generator
import numpy as np
import torch
import cv2
from tqdm import tqdm
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial
import queue
import time
from threading import Event

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

def resize_clip_temporal(clip: torch.Tensor, target_frames: int = 17) -> torch.Tensor:
    """
    Resize clip in temporal dimension to target number of frames
    Args:
        clip: torch.Tensor of shape (T, H, W, C)
        target_frames: target number of frames
    Returns:
        resized clip: torch.Tensor of shape (target_frames, H, W, C)
    """
    T, H, W, C = clip.shape
    
    if T == target_frames:
        return clip
    
    if T == 1:
        # Single frame, repeat to target_frames
        return clip.repeat(target_frames, 1, 1, 1)
    
    # Use interpolation to resize temporal dimension
    # Convert to float for interpolation
    clip_float = clip.float()
    
    # Create indices for interpolation
    original_indices = torch.linspace(0, T - 1, T)
    target_indices = torch.linspace(0, T - 1, target_frames)
    
    # Interpolate each pixel across time
    resized_frames = []
    for t in target_indices:
        # Find the two nearest frames
        t_floor = int(torch.floor(t))
        t_ceil = int(torch.ceil(t))
        
        if t_floor == t_ceil:
            # Exact frame
            resized_frames.append(clip_float[t_floor])
        else:
            # Interpolate between two frames
            weight = t - t_floor
            frame = (1 - weight) * clip_float[t_floor] + weight * clip_float[t_ceil]
            resized_frames.append(frame)
    
    resized_clip = torch.stack(resized_frames, dim=0)
    return resized_clip.to(torch.uint8)

def extract_video_clips_generator(video_path: str, target_clip_length: int = 17, overlap: int = 1) -> Generator[Tuple[int, torch.Tensor], None, None]:
    """
    🚀 Generator版本：逐个生成clips而不是一次性加载所有clips到内存
    Args:
        video_path: path to video file
        target_clip_length: target number of frames per clip after resizing
        overlap: number of overlapping frames between adjacent clips (in terms of seconds)
    Yields:
        (clip_index, clip): 每次yield一个clip
    """
    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        
        # Calculate frames per second (1 second clip length in original video)
        frames_per_second = int(round(original_fps))
        
        print(f"📹 Video FPS: {original_fps:.2f}, using {frames_per_second} frames per 1-second clip")
        
        if total_frames < frames_per_second:
            print(f"⚠️  Video {video_path} has only {total_frames} frames, less than 1 second ({frames_per_second} frames)")
            # If video is too short, use all frames and resize to target length
            all_frames = vr.get_batch(list(range(total_frames)))
            # Convert to torch.Tensor if needed
            if not isinstance(all_frames, torch.Tensor):
                all_frames = torch.from_numpy(all_frames.asnumpy() if hasattr(all_frames, 'asnumpy') else np.array(all_frames))
            
            # Resize temporal dimension to target_clip_length
            resized_clip = resize_clip_temporal(all_frames, target_clip_length)
            yield 0, resized_clip
            return
        
        # Calculate overlap in frames (overlap is in seconds, convert to frames)
        overlap_frames = int(overlap * frames_per_second)
        step = frames_per_second - overlap_frames
        
        clip_index = 0
        for start_idx in range(0, total_frames - frames_per_second + 1, step):
            end_idx = start_idx + frames_per_second
            frame_indices = list(range(start_idx, end_idx))
            
            # 🚀 关键优化：每次只读取一个clip，立即yield
            clip = vr.get_batch(frame_indices)
            # Convert to torch.Tensor if needed
            if not isinstance(clip, torch.Tensor):
                clip = torch.from_numpy(clip.asnumpy() if hasattr(clip, 'asnumpy') else np.array(clip))
            
            # Resize temporal dimension to target_clip_length
            resized_clip = resize_clip_temporal(clip, target_clip_length)
            yield clip_index, resized_clip
            clip_index += 1
        
        # Handle the last clip if there are remaining frames
        remaining_frames = total_frames - (clip_index * step)
        if remaining_frames >= frames_per_second // 2:  # If at least half a second remains
            start_idx = total_frames - frames_per_second
            if start_idx < 0:
                start_idx = 0
            frame_indices = list(range(start_idx, total_frames))
            clip = vr.get_batch(frame_indices)
            # Convert to torch.Tensor if needed
            if not isinstance(clip, torch.Tensor):
                clip = torch.from_numpy(clip.asnumpy() if hasattr(clip, 'asnumpy') else np.array(clip))
            
            # Resize temporal dimension to target_clip_length
            resized_clip = resize_clip_temporal(clip, target_clip_length)
            yield clip_index, resized_clip
            
    except Exception as e:
        print(f"❌ Error processing video {video_path}: {str(e)}")
        return

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

class AsyncVideoProcessor:
    """
    🚀 异步视频处理器：实现读取→处理→保存的流水线并行
    """
    def __init__(self, 
                 read_queue_size: int = 5,
                 process_queue_size: int = 10,
                 num_process_workers: int = None,
                 num_save_workers: int = 2):
        """
        Args:
            read_queue_size: 读取队列大小（控制内存使用）
            process_queue_size: 处理队列大小
            num_process_workers: 处理工作线程数
            num_save_workers: 保存工作线程数
        """
        self.read_queue = queue.Queue(maxsize=read_queue_size)
        self.process_queue = queue.Queue(maxsize=process_queue_size)
        
        if num_process_workers is None:
            num_process_workers = min(multiprocessing.cpu_count(), 4)
        
        self.num_process_workers = num_process_workers
        self.num_save_workers = num_save_workers
        
        self.stop_event = Event()
        self.stats = {
            'clips_read': 0,
            'clips_processed': 0,
            'clips_saved': 0,
            'read_time': 0,
            'process_time': 0,
            'save_time': 0
        }
        
    def reader_worker(self, video_path: str, target_clip_length: int, overlap: int):
        """读取工作线程：负责从视频中读取clips"""
        try:
            start_time = time.time()
            for clip_index, clip in extract_video_clips_generator(video_path, target_clip_length, overlap):
                if self.stop_event.is_set():
                    break
                    
                self.read_queue.put((clip_index, clip))
                self.stats['clips_read'] += 1
                
            # 发送结束信号
            self.read_queue.put(None)
            self.stats['read_time'] = time.time() - start_time
            print(f"📖 Reader finished: {self.stats['clips_read']} clips read in {self.stats['read_time']:.2f}s")
            
        except Exception as e:
            print(f"❌ Reader error: {str(e)}")
            self.read_queue.put(None)
    
    def processor_worker(self, resolution: Tuple[int, int], device: str, use_torch: bool):
        """处理工作线程：负责resize clips"""
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.read_queue.get(timeout=1)
                    if item is None:  # 结束信号
                        self.process_queue.put(None)
                        break
                        
                    clip_index, clip = item
                    
                    start_time = time.time()
                    # 处理clip
                    processed_clip = resize_video_frames(clip, resolution, device, use_torch)
                    process_time = time.time() - start_time
                    
                    self.process_queue.put((clip_index, processed_clip))
                    self.stats['clips_processed'] += 1
                    self.stats['process_time'] += process_time
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            print(f"❌ Processor error: {str(e)}")
            self.process_queue.put(None)
    
    def saver_worker(self, output_dir: str, video_name: str, resolution_name: str, fps: int):
        """保存工作线程：负责保存processed clips"""
        try:
            resolution_dir = os.path.join(output_dir, resolution_name, video_name)
            
            while not self.stop_event.is_set():
                try:
                    item = self.process_queue.get(timeout=1)
                    if item is None:  # 结束信号
                        break
                        
                    clip_index, processed_clip = item
                    
                    start_time = time.time()
                    # 保存clip
                    output_path = os.path.join(resolution_dir, f"clip_{clip_index:04d}.mp4")
                    save_video_clip(processed_clip, output_path, fps)
                    save_time = time.time() - start_time
                    
                    self.stats['clips_saved'] += 1
                    self.stats['save_time'] += save_time
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            print(f"❌ Saver error: {str(e)}")
    
    def process_video_async(self, 
                          video_path: str,
                          output_dir: str,
                          resolution: Tuple[int, int],
                          video_name: str,
                          resolution_name: str,
                          clip_length: int = 17,
                          overlap: int = 0,
                          fps: int = 25,
                          device: str = "cpu",
                          use_torch: bool = False):
        """
        🚀 异步处理单个视频：读取、处理、保存并行进行
        """
        print(f"🚀 Starting async processing: {video_name}")
        print(f"   📊 Pipeline: Reader(1) -> Processors({self.num_process_workers}) -> Savers({self.num_save_workers})")
        
        # 重置统计
        self.stats = {key: 0 for key in self.stats}
        self.stop_event.clear()
        
        # 启动线程
        with ThreadPoolExecutor(max_workers=1 + self.num_process_workers + self.num_save_workers) as executor:
            # 启动读取线程
            reader_future = executor.submit(
                self.reader_worker, video_path, clip_length, overlap
            )
            
            # 启动处理线程
            processor_futures = []
            for _ in range(self.num_process_workers):
                future = executor.submit(
                    self.processor_worker, resolution, device, use_torch
                )
                processor_futures.append(future)
            
            # 启动保存线程
            saver_futures = []
            for _ in range(self.num_save_workers):
                future = executor.submit(
                    self.saver_worker, output_dir, video_name, resolution_name, fps
                )
                saver_futures.append(future)
            
            # 等待所有线程完成
            try:
                reader_future.result()
                for future in processor_futures:
                    future.result()
                for future in saver_futures:
                    future.result()
                    
            except Exception as e:
                print(f"❌ Error in async processing: {str(e)}")
                self.stop_event.set()
        
        # 打印统计信息
        total_time = max(self.stats['read_time'], self.stats['process_time'], self.stats['save_time'])
        print(f"📊 Processing Stats for {video_name}:")
        print(f"   📖 Read: {self.stats['clips_read']} clips in {self.stats['read_time']:.2f}s")
        print(f"   🔄 Process: {self.stats['clips_processed']} clips in {self.stats['process_time']:.2f}s")
        print(f"   💾 Save: {self.stats['clips_saved']} clips in {self.stats['save_time']:.2f}s")
        print(f"   ⏱️  Total pipeline time: {total_time:.2f}s")
        
        if self.stats['clips_processed'] > 0:
            avg_process_time = self.stats['process_time'] / self.stats['clips_processed']
            print(f"   📈 Avg processing time per clip: {avg_process_time:.3f}s")

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

def preprocess_video_file_async(
    video_path: str,
    output_dir: str,
    target_resolutions: List[Tuple[int, int]],
    clip_length: int = 17,
    overlap: int = 0,
    fps: int = 25,
    device: str = "auto",
    use_torch: bool = False,
    read_queue_size: int = 5,
    process_queue_size: int = 10,
    num_process_workers: int = None,
    num_save_workers: int = 2
):
    """
    🚀 异步预处理单个视频文件
    Args:
        video_path: path to input video
        output_dir: output directory
        target_resolutions: list of (height, width) tuples
        clip_length: target number of frames per clip after temporal resizing
        overlap: overlap between clips (in seconds)
        fps: output fps
        device: "cpu", "cuda", or "auto" for automatic detection
        use_torch: whether to use torch operations for resizing (GPU-friendly)
        read_queue_size: 读取队列大小（控制内存使用）
        process_queue_size: 处理队列大小
        num_process_workers: 处理工作线程数
        num_save_workers: 保存工作线程数
    """
    video_name = pathlib.Path(video_path).stem
    
    # Auto-detect device
    if device == "auto":
        device = get_device_info()
    elif device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    print(f"🎬 Processing video: {video_path}")
    print(f"⚙️  Device: {device.upper()}, Use torch: {use_torch}")
    print(f"🔧 Queue sizes: Read({read_queue_size}), Process({process_queue_size})")
    print(f"👥 Workers: Process({num_process_workers or 'auto'}), Save({num_save_workers})")
    
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
    
    # 创建异步处理器
    processor = AsyncVideoProcessor(
        read_queue_size=read_queue_size,
        process_queue_size=process_queue_size,
        num_process_workers=num_process_workers,
        num_save_workers=num_save_workers
    )
    
    # 异步处理视频
    processor.process_video_async(
        video_path=video_path,
        output_dir=output_dir,
        resolution=best_resolution,
        video_name=video_name,
        resolution_name=resolution_name,
        clip_length=clip_length,
        overlap=overlap,
        fps=fps,
        device=device,
        use_torch=use_torch
    )
    
    # Save metadata
    resolution_dir = os.path.join(output_dir, resolution_name, video_name)
    metadata = {
        "original_video": video_path,
        "original_resolution": original_size,
        "clip_length": clip_length,
        "overlap": overlap,
        "target_resolution": best_resolution,
        "num_clips": processor.stats['clips_saved'],
        "fps": fps,
        "device": device,
        "use_torch": use_torch,
        "async_config": {
            "read_queue_size": read_queue_size,
            "process_queue_size": process_queue_size,
            "num_process_workers": processor.num_process_workers,
            "num_save_workers": num_save_workers
        },
        "performance_stats": processor.stats
    }
    metadata_path = os.path.join(resolution_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Completed {resolution_name} - saved to {resolution_dir}")

def preprocess_videos_async(
    input_dir: str,
    output_dir: str,
    target_resolutions: List[Tuple[int, int]] = None,
    clip_length: int = 17,
    overlap: int = 0,
    fps: int = 25,
    video_extensions: List[str] = None,
    device: str = "auto",
    use_torch: bool = False,
    read_queue_size: int = 5,
    process_queue_size: int = 10,
    num_process_workers: int = None,
    num_save_workers: int = 2,
    max_concurrent_videos: int = 2
):
    """
    🚀 异步预处理目录中的所有视频文件 - 支持多视频并行处理
    Args:
        input_dir: input directory containing videos
        output_dir: output directory
        target_resolutions: list of (height, width) tuples
        clip_length: target number of frames per clip after temporal resizing
        overlap: overlap between clips (in seconds)
        fps: output fps
        video_extensions: supported video file extensions
        device: "cpu", "cuda", or "auto" for automatic detection
        use_torch: whether to use torch operations for resizing (GPU-friendly)
        read_queue_size: 读取队列大小（控制内存使用）
        process_queue_size: 处理队列大小
        num_process_workers: 处理工作线程数
        num_save_workers: 保存工作线程数
        max_concurrent_videos: 最大并行处理的视频数量
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
    print(f"🚀 Processing with max {max_concurrent_videos} concurrent videos")
    print(f"📊 Each video pipeline: Reader(1) -> Processors({num_process_workers or 'auto'}) -> Savers({num_save_workers})")
    
    # Auto-detect device once
    if device == "auto":
        device = get_device_info()
    elif device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    # Process videos concurrently
    def process_single_video_wrapper(video_file):
        """包装函数用于并行处理单个视频"""
        try:
            video_path = str(video_file)
            print(f"🎬 Starting video: {video_file.name}")
            
            preprocess_video_file_async(
                video_path,
                output_dir,
                target_resolutions,
                clip_length,
                overlap,
                fps,
                device,
                use_torch,
                read_queue_size,
                process_queue_size,
                num_process_workers,
                num_save_workers
            )
            
            print(f"✅ Completed video: {video_file.name}")
            return video_path, True, None
            
        except Exception as e:
            error_msg = f"Error processing {video_file}: {str(e)}"
            print(f"❌ {error_msg}")
            return str(video_file), False, error_msg
    
    # Track results
    successful = 0
    failed = 0
    start_time = time.time()
    
    if max_concurrent_videos == 1:
        # Sequential processing
        print("🔄 Sequential processing mode")
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path, success, error = process_single_video_wrapper(video_file)
            if success:
                successful += 1
            else:
                failed += 1
    else:
        # Concurrent processing
        print(f"🚀 Concurrent processing mode ({max_concurrent_videos} videos in parallel)")
        
        with ThreadPoolExecutor(max_workers=max_concurrent_videos) as executor:
            # Submit all video processing tasks
            future_to_video = {
                executor.submit(process_single_video_wrapper, video_file): video_file 
                for video_file in video_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(video_files), desc="Processing videos") as pbar:
                for future in as_completed(future_to_video):
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
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': successful, 
                        'Failed': failed,
                        'Active': len([f for f in future_to_video.keys() if not f.done()])
                    })
    
    total_time = time.time() - start_time
    
    print(f"\n📊 Batch Processing Complete:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏱️  Total time: {total_time:.2f}s")
    print(f"   📈 Average time per video: {total_time/len(video_files):.2f}s")
    
    if successful > 0:
        print(f"   🚀 Effective throughput: {successful/(total_time/60):.1f} videos/minute")

def main():
    parser = argparse.ArgumentParser(description="🚀 Async Video Preprocessing with Pipeline Parallelism")
    parser.add_argument("--input_dir", type=str, help="Input directory containing videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--clip_length", type=int, default=17, help="Target number of frames per clip after temporal resizing")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap between clips in seconds")
    parser.add_argument("--fps", type=int, default=17, help="Output FPS")
    parser.add_argument("--single_video", type=str, help="Process a single video file instead of directory")
    
    # Performance options
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                       help="Processing device: auto (detect), cpu, or cuda")
    parser.add_argument("--use_torch", action="store_true", 
                       help="Use torch operations for resizing (GPU-friendly, slower on CPU)")
    
    # 🚀 Async-specific options
    parser.add_argument("--read_queue_size", type=int, default=5, 
                       help="Size of read queue (controls memory usage)")
    parser.add_argument("--process_queue_size", type=int, default=10, 
                       help="Size of process queue")
    parser.add_argument("--num_process_workers", type=int, default=None, 
                       help="Number of processing worker threads")
    parser.add_argument("--num_save_workers", type=int, default=2, 
                       help="Number of saving worker threads")
    
    # 🚀 Async-specific options
    parser.add_argument("--max_concurrent_videos", type=int, default=2, 
                       help="Maximum number of concurrent videos to process")
    
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
    print("🚀 Async Video Preprocessing Configuration:")
    print(f"   Device: {args.device}")
    print(f"   Use Torch: {args.use_torch}")
    print(f"   Read Queue Size: {args.read_queue_size}")
    print(f"   Process Queue Size: {args.process_queue_size}")
    print(f"   Process Workers: {args.num_process_workers or 'auto'}")
    print(f"   Save Workers: {args.num_save_workers}")
    print(f"   Target clip length: {args.clip_length} frames (after temporal resizing)")
    print(f"   Clip extraction: 1-second segments from original video")
    print(f"   Overlap: {args.overlap} seconds")
    print(f"   Output FPS: {args.fps}")
    print(f"   Max Concurrent Videos: {args.max_concurrent_videos}")
    print()

    if args.single_video:
        # Process single video
        preprocess_video_file_async(
            args.single_video,
            args.output_dir,
            target_resolutions,
            args.clip_length,
            args.overlap,
            args.fps,
            args.device,
            args.use_torch,
            args.read_queue_size,
            args.process_queue_size,
            args.num_process_workers,
            args.num_save_workers
        )
    else:
        # Process directory with async multi-video support
        preprocess_videos_async(
            args.input_dir,
            args.output_dir,
            target_resolutions,
            args.clip_length,
            args.overlap,
            args.fps,
            video_extensions=None,
            device=args.device,
            use_torch=args.use_torch,
            read_queue_size=args.read_queue_size,
            process_queue_size=args.process_queue_size,
            num_process_workers=args.num_process_workers,
            num_save_workers=args.num_save_workers,
            max_concurrent_videos=args.max_concurrent_videos
        )
    
    print("🎉 Async preprocessing completed!")

if __name__ == "__main__":
    main()