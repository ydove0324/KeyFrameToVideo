#!/usr/bin/env python3
"""
视频解码瓶颈测试脚本
分别测量解码、处理、保存各阶段的性能
"""

import argparse
import time
import os
import pathlib
import numpy as np
import torch
import cv2
from typing import List, Tuple, Union
import decord
import psutil
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

decord.bridge.set_bridge("torch")

class DetailedPerformanceMonitor:
    """详细的性能监控器"""
    
    def __init__(self):
        self.stats = {
            'decode_times': [],
            'resize_times': [],
            'save_times': [],
            'total_frames': 0,
            'cpu_usage_during_decode': [],
            'memory_usage': []
        }
        self.monitoring = False
        
    def start_monitoring(self):
        """开始后台监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            self.stats['cpu_usage_during_decode'].append(cpu_percent)
            self.stats['memory_usage'].append(memory_percent)
            
            time.sleep(0.2)

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

def process_clips_multithread_resize(clips: List[torch.Tensor], 
                                   resolution: Tuple[int, int], 
                                   device: str = "cpu", 
                                   use_torch: bool = False, 
                                   num_workers: int = None) -> List[torch.Tensor]:
    """Process multiple clips with multithreading for resize testing"""
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

def test_decode_performance(video_path: str, num_test_clips: int = 5, test_multithreading: bool = True) -> dict:
    """测试视频解码性能，包括单线程和多线程resize对比"""
    
    print(f"🔍 测试视频: {video_path}")
    print(f"📊 测试clip数量: {num_test_clips}")
    
    # 性能监控器
    monitor = DetailedPerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        vr = decord.VideoReader(video_path, num_threads=1)  # 单线程解码
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        frames_per_clip = int(round(fps))
        
        print(f"📹 视频信息: {total_frames}帧, {fps:.2f}fps")
        print(f"🎯 每个clip: {frames_per_clip}帧")
        
        # 先收集所有clips用于后续测试
        all_clips = []
        decode_times = []
        
        for i in range(min(num_test_clips, total_frames // frames_per_clip)):
            start_idx = i * frames_per_clip
            end_idx = start_idx + frames_per_clip
            frame_indices = list(range(start_idx, end_idx))
            
            # 🔴 测试解码性能
            decode_start = time.time()
            frames = vr.get_batch(frame_indices)
            if not isinstance(frames, torch.Tensor):
                frames = torch.from_numpy(frames.asnumpy())
            decode_time = time.time() - decode_start
            decode_times.append(decode_time)
            all_clips.append(frames)
        
        # 🟡 测试单线程Resize性能
        print("\n🟡 测试单线程Resize性能...")
        single_thread_resize_times = []
        target_size = (480, 832)  # 固定目标分辨率
        
        for i, frames in enumerate(all_clips):
            resize_start = time.time()
            resized_frames = resize_video_frames(frames, target_size, "cpu", False)
            resize_time = time.time() - resize_start
            single_thread_resize_times.append(resize_time)
            print(f"  Clip {i}: 单线程Resize={resize_time:.3f}s")
        
        # 🟡 测试多线程Resize性能
        multithread_resize_times = []
        if test_multithreading and len(all_clips) > 1:
            print("\n🟡 测试多线程Resize性能...")
            num_workers = min(multiprocessing.cpu_count(), 4)
            print(f"  使用 {num_workers} 个工作线程")
            
            # 分批测试以获得更准确的时间
            batch_size = min(len(all_clips), 4)  # 每批处理4个clips
            for batch_start in range(0, len(all_clips), batch_size):
                batch_clips = all_clips[batch_start:batch_start + batch_size]
                
                multithread_start = time.time()
                multithread_resized = process_clips_multithread_resize(
                    batch_clips, target_size, "cpu", False, num_workers
                )
                multithread_time = time.time() - multithread_start
                
                # 平均到每个clip
                avg_time_per_clip = multithread_time / len(batch_clips)
                for _ in range(len(batch_clips)):
                    multithread_resize_times.append(avg_time_per_clip)
                
                print(f"  批次 {batch_start//batch_size}: 多线程Resize={multithread_time:.3f}s ({len(batch_clips)} clips)")
        
        # 使用最优的resize结果进行保存测试
        if multithread_resize_times:
            best_resize_time = min(np.mean(single_thread_resize_times), np.mean(multithread_resize_times))
            use_multithread = np.mean(multithread_resize_times) < np.mean(single_thread_resize_times)
        else:
            best_resize_time = np.mean(single_thread_resize_times)
            use_multithread = False
        
        # 🟢 测试保存性能
        print("\n🟢 测试保存性能...")
        save_times = []
        
        for i, frames in enumerate(all_clips):
            # 使用最优的resize方式
            if use_multithread and len(all_clips) > 1:
                resized_frames = process_clips_multithread_resize([frames], target_size, "cpu", False, 2)[0]
            else:
                resized_frames = resize_video_frames(frames, target_size, "cpu", False)
            
            save_start = time.time()
            # 模拟保存（创建VideoWriter但不实际写入）
            temp_path = f"/tmp/test_clip_{i}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 25, (target_size[1], target_size[0]))
            
            # 转换为numpy array进行保存
            if isinstance(resized_frames, torch.Tensor):
                resized_frames_np = resized_frames.numpy().astype(np.uint8)
            else:
                resized_frames_np = resized_frames
            
            for j in range(resized_frames_np.shape[0]):
                frame = resized_frames_np[j]
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            save_time = time.time() - save_start
            save_times.append(save_time)
            
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            print(f"  Clip {i}: 保存={save_time:.3f}s")
        
        monitor.stop_monitoring()
        
        # 计算统计数据
        avg_decode_time = np.mean(decode_times)
        avg_single_resize_time = np.mean(single_thread_resize_times)
        avg_multi_resize_time = np.mean(multithread_resize_times) if multithread_resize_times else avg_single_resize_time
        avg_save_time = np.mean(save_times)
        
        # 使用最优resize时间计算
        best_resize_time = min(avg_single_resize_time, avg_multi_resize_time)
        total_time = avg_decode_time + best_resize_time + avg_save_time
        
        decode_percentage = (avg_decode_time / total_time) * 100
        resize_percentage = (best_resize_time / total_time) * 100
        save_percentage = (avg_save_time / total_time) * 100
        
        T, H, W, C = all_clips[0].shape
        
        result = {
            'video_info': {
                'path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'resolution': f"{W}x{H}"
            },
            'performance': {
                'avg_decode_time': avg_decode_time,
                'avg_resize_time_single': avg_single_resize_time,
                'avg_resize_time_multi': avg_multi_resize_time,
                'avg_resize_time_best': best_resize_time,
                'avg_save_time': avg_save_time,
                'total_time_per_clip': total_time,
                'decode_percentage': decode_percentage,
                'resize_percentage': resize_percentage,
                'save_percentage': save_percentage,
                'multithread_speedup': avg_single_resize_time / avg_multi_resize_time if multithread_resize_times else 1.0,
                'recommended_multithread': use_multithread
            },
            'bottleneck_analysis': {
                'primary_bottleneck': 'decode' if decode_percentage > 50 else 'resize' if resize_percentage > 30 else 'save',
                'decode_fps': frames_per_clip / avg_decode_time,
                'resize_fps_single': frames_per_clip / avg_single_resize_time,
                'resize_fps_multi': frames_per_clip / avg_multi_resize_time if multithread_resize_times else 0,
                'save_fps': frames_per_clip / avg_save_time,
                'overall_fps': frames_per_clip / total_time
            },
            'system_usage': {
                'avg_cpu_during_test': np.mean(monitor.stats['cpu_usage_during_decode']) if monitor.stats['cpu_usage_during_decode'] else 0,
                'max_cpu_during_test': np.max(monitor.stats['cpu_usage_during_decode']) if monitor.stats['cpu_usage_during_decode'] else 0,
                'avg_memory_usage': np.mean(monitor.stats['memory_usage']) if monitor.stats['memory_usage'] else 0
            }
        }
        
        return result
        
    except Exception as e:
        monitor.stop_monitoring()
        return {'error': str(e)}

def test_decode_optimization(video_path: str) -> dict:
    """测试不同解码优化方案"""
    
    print(f"\n🚀 测试解码优化方案...")
    
    optimization_results = {}
    
    # 测试不同的decord线程数
    thread_counts = [1, 2, 4, 8]
    
    for num_threads in thread_counts:
        print(f"\n📊 测试 {num_threads} 解码线程...")
        
        try:
            vr = decord.VideoReader(video_path, num_threads=num_threads)
            fps = vr.get_avg_fps()
            frames_per_clip = int(round(fps))
            frame_indices = list(range(0, frames_per_clip))
            
            # 测试解码速度
            times = []
            for i in range(3):  # 测试3次取平均
                start_time = time.time()
                frames = vr.get_batch(frame_indices)
                decode_time = time.time() - start_time
                times.append(decode_time)
            
            avg_time = np.mean(times)
            optimization_results[f'{num_threads}_threads'] = {
                'avg_decode_time': avg_time,
                'decode_fps': frames_per_clip / avg_time,
                'improvement_vs_single': (times[0] / avg_time) if num_threads > 1 else 1.0
            }
            
            print(f"   解码时间: {avg_time:.3f}s, FPS: {frames_per_clip/avg_time:.1f}")
            
        except Exception as e:
            print(f"   ❌ {num_threads}线程测试失败: {e}")
            optimization_results[f'{num_threads}_threads'] = {'error': str(e)}
    
    return optimization_results

def print_analysis_results(result: dict):
    """打印分析结果"""
    
    if 'error' in result:
        print(f"❌ 测试失败: {result['error']}")
        return
    
    print("\n" + "="*60)
    print("📊 视频处理瓶颈分析结果")
    print("="*60)
    
    # 视频信息
    video_info = result['video_info']
    print(f"\n📹 视频信息:")
    print(f"   文件: {pathlib.Path(video_info['path']).name}")
    print(f"   分辨率: {video_info['resolution']}")
    print(f"   帧率: {video_info['fps']:.2f} fps")
    print(f"   总帧数: {video_info['total_frames']}")
    
    # 性能分析
    perf = result['performance']
    print(f"\n⏱️  各阶段平均耗时:")
    print(f"   🔴 解码: {perf['avg_decode_time']:.3f}s ({perf['decode_percentage']:.1f}%)")
    
    # Resize性能对比
    if 'avg_resize_time_multi' in perf and perf['avg_resize_time_multi'] != perf['avg_resize_time_single']:
        print(f"   🟡 Resize (单线程): {perf['avg_resize_time_single']:.3f}s")
        print(f"   🟡 Resize (多线程): {perf['avg_resize_time_multi']:.3f}s")
        print(f"   🟡 Resize (最优): {perf['avg_resize_time_best']:.3f}s ({perf['resize_percentage']:.1f}%)")
        print(f"   📈 多线程加速比: {perf['multithread_speedup']:.2f}x")
        print(f"   ✅ 推荐使用: {'多线程' if perf['recommended_multithread'] else '单线程'}")
    else:
        print(f"   🟡 Resize: {perf['avg_resize_time_single']:.3f}s ({perf['resize_percentage']:.1f}%)")
    
    print(f"   🟢 保存: {perf['avg_save_time']:.3f}s ({perf['save_percentage']:.1f}%)")
    print(f"   ⏱️  总计: {perf['total_time_per_clip']:.3f}s/clip")
    
    # 瓶颈分析
    bottleneck = result['bottleneck_analysis']
    print(f"\n🎯 瓶颈识别:")
    primary = bottleneck['primary_bottleneck']
    if primary == 'decode':
        print(f"   🔴 主要瓶颈: 视频解码 ({perf['decode_percentage']:.1f}%)")
        print(f"   💡 建议: 优化解码参数，使用多线程解码")
    elif primary == 'resize':
        print(f"   🟡 主要瓶颈: 图像处理 ({perf['resize_percentage']:.1f}%)")
        print(f"   💡 建议: 考虑GPU加速或优化resize算法")
    else:
        print(f"   🟢 主要瓶颈: 文件保存 ({perf['save_percentage']:.1f}%)")
        print(f"   💡 建议: 优化存储或编码参数")
    
    print(f"\n📈 处理速度:")
    print(f"   解码FPS: {bottleneck['decode_fps']:.1f}")
    if 'resize_fps_multi' in bottleneck and bottleneck['resize_fps_multi'] > 0:
        print(f"   Resize FPS (单线程): {bottleneck['resize_fps_single']:.1f}")
        print(f"   Resize FPS (多线程): {bottleneck['resize_fps_multi']:.1f}")
    else:
        print(f"   处理FPS: {bottleneck['resize_fps_single']:.1f}")
    print(f"   保存FPS: {bottleneck['save_fps']:.1f}")
    print(f"   综合FPS: {bottleneck['overall_fps']:.1f}")
    
    # 系统资源使用
    system = result['system_usage']
    print(f"\n💻 系统资源使用:")
    print(f"   平均CPU: {system['avg_cpu_during_test']:.1f}%")
    print(f"   峰值CPU: {system['max_cpu_during_test']:.1f}%")
    print(f"   内存使用: {system['avg_memory_usage']:.1f}%")

def print_optimization_suggestions(result: dict, optimization_result: dict):
    """打印优化建议"""
    
    print(f"\n💡 优化建议:")
    
    if 'performance' in result:
        decode_pct = result['performance']['decode_percentage']
        resize_pct = result['performance']['resize_percentage']
        
        # 多线程resize分析
        multithread_speedup = result['performance'].get('multithread_speedup', 1.0)
        recommended_multithread = result['performance'].get('recommended_multithread', False)
        
        if decode_pct > 60:
            print(f"   🔴 解码是主要瓶颈 ({decode_pct:.1f}%)")
            print(f"   建议:")
            print(f"     • 使用多线程解码: num_threads=4")
            print(f"     • 预处理视频格式以降低解码复杂度")
            print(f"     • 考虑硬件解码 (如果支持)")
            print(f"     • 减少并行视频数，增加单视频解码线程")
            
            # 分析多线程效果
            if '4_threads' in optimization_result:
                improvement = optimization_result['4_threads'].get('improvement_vs_single', 1.0)
                print(f"     • 4线程解码预期提升: {improvement:.1f}x")
        
        elif resize_pct > 40:
            print(f"   🟡 图像处理是主要瓶颈 ({resize_pct:.1f}%)")
            print(f"   建议:")
            if multithread_speedup > 1.2:
                print(f"     • ✅ 多线程resize有效 (加速比: {multithread_speedup:.2f}x)")
                print(f"     • 使用多线程处理: --num_process_workers 4-8")
            else:
                print(f"     • ⚠️  多线程resize效果有限 (加速比: {multithread_speedup:.2f}x)")
                print(f"     • 考虑GPU加速: --device cuda --use_torch")
            print(f"     • 增加处理线程: --num_process_workers 8+")
            print(f"     • 使用更快的resize算法")
        
        else:
            print(f"   ✅ 性能较为均衡，可以适当增加并行度")
            if multithread_speedup > 1.1:
                print(f"   📈 多线程resize有效 (加速比: {multithread_speedup:.2f}x)")
    
    # Resize优化专门建议
    if multithread_speedup > 1.0:
        print(f"\n🔧 Resize优化分析:")
        print(f"   多线程加速比: {multithread_speedup:.2f}x")
        if recommended_multithread:
            print(f"   ✅ 推荐使用多线程resize")
            print(f"   建议线程数: 4-8 (根据CPU核心数调整)")
        else:
            print(f"   ❌ 单线程resize更优")
            print(f"   原因: 可能是clips数量少或CPU核心数限制")
    
    print(f"\n🚀 推荐参数配置:")
    if decode_pct > 60:
        print(f"   # 解码瓶颈优化配置")
        print(f"   python tool/video_preprocess_async.py \\")
        print(f"       --max_concurrent_videos 2 \\")
        print(f"       --read_queue_size 8 \\")
        print(f"       --num_process_workers 6")
    elif resize_pct > 40 and recommended_multithread:
        print(f"   # Resize瓶颈 + 多线程优化配置")
        print(f"   python tool/video_preprocess_async.py \\")
        print(f"       --max_concurrent_videos 3 \\")
        print(f"       --read_queue_size 10 \\")
        print(f"       --num_process_workers 8")
    else:
        print(f"   # 均衡配置")
        print(f"   python tool/video_preprocess_async.py \\")
        print(f"       --max_concurrent_videos 4 \\")
        print(f"       --read_queue_size 10 \\")
        print(f"       --num_process_workers 8")

def main():
    parser = argparse.ArgumentParser(description="🔍 视频解码瓶颈分析工具")
    parser.add_argument("--video", type=str, required=True, help="测试视频文件路径")
    parser.add_argument("--num_clips", type=int, default=5, help="测试的clip数量")
    parser.add_argument("--test_optimization", action="store_true", help="测试解码优化方案")
    parser.add_argument("--no_multithread_test", action="store_true", help="跳过多线程resize测试")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ 视频文件不存在: {args.video}")
        return
    
    # 主要瓶颈测试
    print("🔍 开始视频处理瓶颈分析...")
    print(f"💻 CPU核心数: {multiprocessing.cpu_count()}")
    
    test_multithreading = not args.no_multithread_test
    if test_multithreading:
        print("📊 将测试单线程 vs 多线程 resize性能")
    else:
        print("📊 仅测试单线程resize性能")
    
    result = test_decode_performance(args.video, args.num_clips, test_multithreading)
    print_analysis_results(result)
    
    # 解码优化测试
    optimization_result = {}
    if args.test_optimization:
        optimization_result = test_decode_optimization(args.video)
    
    # 优化建议
    print_optimization_suggestions(result, optimization_result)

if __name__ == "__main__":
    main() 