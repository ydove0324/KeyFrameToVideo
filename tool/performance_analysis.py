#!/usr/bin/env python3
"""
视频预处理性能分析脚本
用于分析和对比原版与异步版本的性能差异，定位瓶颈
支持单视频和多视频批处理性能测试
"""

import time
import psutil
import threading
import subprocess
import argparse
import os
import pathlib
from typing import Dict, List
import json

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io_read': [],
            'disk_io_write': [],
            'timestamps': []
        }
        self.start_time = None
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.start_time = time.time()
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
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # 内存使用率
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # 磁盘I/O
                disk_io = psutil.disk_io_counters()
                
                # 记录数据
                current_time = time.time() - self.start_time
                self.stats['timestamps'].append(current_time)
                self.stats['cpu_usage'].append(cpu_percent)
                self.stats['memory_usage'].append(memory_percent)
                self.stats['disk_io_read'].append(disk_io.read_bytes if disk_io else 0)
                self.stats['disk_io_write'].append(disk_io.write_bytes if disk_io else 0)
                
                time.sleep(0.5)  # 每0.5秒采样一次
                
            except Exception as e:
                print(f"监控错误: {e}")
                
    def get_summary(self) -> Dict:
        """获取监控摘要"""
        if not self.stats['timestamps']:
            return {}
            
        # 计算平均值和峰值
        summary = {
            'duration': max(self.stats['timestamps']) if self.stats['timestamps'] else 0,
            'cpu_avg': sum(self.stats['cpu_usage']) / len(self.stats['cpu_usage']) if self.stats['cpu_usage'] else 0,
            'cpu_max': max(self.stats['cpu_usage']) if self.stats['cpu_usage'] else 0,
            'memory_avg': sum(self.stats['memory_usage']) / len(self.stats['memory_usage']) if self.stats['memory_usage'] else 0,
            'memory_max': max(self.stats['memory_usage']) if self.stats['memory_usage'] else 0,
        }
        
        # 磁盘I/O增量
        if len(self.stats['disk_io_read']) > 1:
            read_diff = self.stats['disk_io_read'][-1] - self.stats['disk_io_read'][0]
            write_diff = self.stats['disk_io_write'][-1] - self.stats['disk_io_write'][0]
            
            summary['disk_read_mb'] = read_diff / (1024 * 1024)
            summary['disk_write_mb'] = write_diff / (1024 * 1024)
        else:
            summary['disk_read_mb'] = 0
            summary['disk_write_mb'] = 0
            
        return summary

def find_test_videos(input_path: str, max_videos: int = 5) -> List[str]:
    """查找测试视频文件"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    if os.path.isfile(input_path):
        # 单个视频文件
        return [input_path]
    
    if os.path.isdir(input_path):
        # 目录，查找视频文件
        input_dir = pathlib.Path(input_path)
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(input_dir.rglob(f"*{ext}"))
            video_files.extend(input_dir.rglob(f"*{ext.upper()}"))
        
        # 限制测试视频数量
        video_files = video_files[:max_videos]
        return [str(f) for f in video_files]
    
    return []

def run_original_version(video_input: str, output_dir: str, args_dict: Dict, is_batch: bool = False) -> Dict:
    """运行原版脚本"""
    print("🔄 Running ORIGINAL version...")
    
    # 构建命令
    cmd = [
        "python", "tool/video_preprocess.py",
        "--output_dir", output_dir + "_original",
        "--device", args_dict.get('device', 'auto'),
        "--clip_length", str(args_dict.get('clip_length', 17)),
        "--overlap", str(args_dict.get('overlap', 0)),
        "--fps", str(args_dict.get('fps', 17)),
        "--num_workers", str(args_dict.get('num_workers', 4))
    ]
    
    if is_batch:
        cmd.extend(["--input_dir", video_input])
        cmd.extend(["--max_concurrent_videos", str(args_dict.get('max_concurrent_videos', 1))])
    else:
        cmd.extend(["--single_video", video_input])
    
    if args_dict.get('use_torch', False):
        cmd.append("--use_torch")
    
    # 开始监控
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # 运行脚本
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr
    except subprocess.TimeoutExpired:
        success = False
        output = "Process timed out"
        error = "Timeout after 2 hours"
    except Exception as e:
        success = False
        output = ""
        error = str(e)
    
    end_time = time.time()
    
    # 停止监控
    monitor.stop_monitoring()
    
    # 收集结果
    perf_summary = monitor.get_summary()
    
    return {
        'version': 'original',
        'success': success,
        'total_time': end_time - start_time,
        'output': output,
        'error': error,
        'performance': perf_summary,
        'is_batch': is_batch
    }

def run_async_version(video_input: str, output_dir: str, args_dict: Dict, is_batch: bool = False) -> Dict:
    """运行异步版脚本"""
    print("🚀 Running ASYNC version...")
    
    # 构建命令
    cmd = [
        "python", "tool/video_preprocess_async.py",
        "--output_dir", output_dir + "_async",
        "--device", args_dict.get('device', 'auto'),
        "--clip_length", str(args_dict.get('clip_length', 17)),
        "--overlap", str(args_dict.get('overlap', 0)),
        "--fps", str(args_dict.get('fps', 17)),
        "--read_queue_size", str(args_dict.get('read_queue_size', 5)),
        "--process_queue_size", str(args_dict.get('process_queue_size', 10)),
        "--num_process_workers", str(args_dict.get('num_process_workers', 4)),
        "--num_save_workers", str(args_dict.get('num_save_workers', 2))
    ]
    
    if is_batch:
        cmd.extend(["--input_dir", video_input])
        cmd.extend(["--max_concurrent_videos", str(args_dict.get('max_concurrent_videos', 2))])
    else:
        cmd.extend(["--single_video", video_input])
    
    if args_dict.get('use_torch', False):
        cmd.append("--use_torch")
    
    # 开始监控
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # 运行脚本
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        success = result.returncode == 0
        output = result.stdout
        error = result.stderr
    except subprocess.TimeoutExpired:
        success = False
        output = "Process timed out"
        error = "Timeout after 2 hours"
    except Exception as e:
        success = False
        output = ""
        error = str(e)
    
    end_time = time.time()
    
    # 停止监控
    monitor.stop_monitoring()
    
    # 收集结果
    perf_summary = monitor.get_summary()
    
    return {
        'version': 'async',
        'success': success,
        'total_time': end_time - start_time,
        'output': output,
        'error': error,
        'performance': perf_summary,
        'is_batch': is_batch
    }

def analyze_results(original_result: Dict, async_result: Dict, num_videos: int = 1) -> Dict:
    """分析对比结果"""
    print("\n" + "="*60)
    print("📊 PERFORMANCE ANALYSIS RESULTS")
    print("="*60)
    
    is_batch = original_result.get('is_batch', False) or async_result.get('is_batch', False)
    
    # 基本信息对比
    print(f"\n🕐 TIMING COMPARISON ({num_videos} video{'s' if num_videos > 1 else ''}):")
    orig_time = original_result['total_time']
    async_time = async_result['total_time']
    
    print(f"   Original version: {orig_time:.2f}s")
    print(f"   Async version: {async_time:.2f}s")
    
    speedup = 1.0
    improvement = 0.0
    if orig_time > 0 and async_time > 0:
        speedup = orig_time / async_time
        improvement = (orig_time - async_time) / orig_time * 100
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Improvement: {improvement:.1f}%")
        
        if num_videos > 1:
            orig_per_video = orig_time / num_videos
            async_per_video = async_time / num_videos
            print(f"   Original per video: {orig_per_video:.2f}s")
            print(f"   Async per video: {async_per_video:.2f}s")
            print(f"   Throughput improvement: {(orig_per_video/async_per_video):.2f}x")
    
    # 资源使用对比
    print("\n💻 RESOURCE USAGE COMPARISON:")
    
    orig_perf = original_result.get('performance', {})
    async_perf = async_result.get('performance', {})
    
    print(f"   CPU Usage (avg):")
    print(f"     Original: {orig_perf.get('cpu_avg', 0):.1f}%")
    print(f"     Async: {async_perf.get('cpu_avg', 0):.1f}%")
    
    print(f"   CPU Usage (max):")
    print(f"     Original: {orig_perf.get('cpu_max', 0):.1f}%")
    print(f"     Async: {async_perf.get('cpu_max', 0):.1f}%")
    
    print(f"   Memory Usage (avg):")
    print(f"     Original: {orig_perf.get('memory_avg', 0):.1f}%")
    print(f"     Async: {async_perf.get('memory_avg', 0):.1f}%")
    
    print(f"   Memory Usage (max):")
    print(f"     Original: {orig_perf.get('memory_max', 0):.1f}%")
    print(f"     Async: {async_perf.get('memory_max', 0):.1f}%")
    
    print(f"   Disk I/O:")
    print(f"     Original Read: {orig_perf.get('disk_read_mb', 0):.1f}MB")
    print(f"     Async Read: {async_perf.get('disk_read_mb', 0):.1f}MB")
    print(f"     Original Write: {orig_perf.get('disk_write_mb', 0):.1f}MB")
    print(f"     Async Write: {async_perf.get('disk_write_mb', 0):.1f}MB")
    
    # 瓶颈分析
    print("\n🔍 BOTTLENECK ANALYSIS:")
    
    # CPU利用率分析
    orig_cpu_avg = orig_perf.get('cpu_avg', 0)
    async_cpu_avg = async_perf.get('cpu_avg', 0)
    
    if orig_cpu_avg < 50:
        print("   ⚠️  Original version: LOW CPU utilization - likely I/O bound")
    elif orig_cpu_avg > 80:
        print("   🔥 Original version: HIGH CPU utilization - likely CPU bound")
    else:
        print("   ✅ Original version: MODERATE CPU utilization")
    
    if async_cpu_avg < 50:
        print("   ⚠️  Async version: LOW CPU utilization - likely I/O bound")
    elif async_cpu_avg > 80:
        print("   🔥 Async version: HIGH CPU utilization - likely CPU bound")
    else:
        print("   ✅ Async version: MODERATE CPU utilization")
    
    # 内存使用分析
    orig_mem_max = orig_perf.get('memory_max', 0)
    async_mem_max = async_perf.get('memory_max', 0)
    
    if is_batch and num_videos > 1:
        print(f"\n📊 BATCH PROCESSING ANALYSIS:")
        if async_mem_max < orig_mem_max * 0.8:
            print("   🚀 Async version shows significant memory efficiency improvement")
        elif async_mem_max < orig_mem_max:
            print("   👍 Async version shows moderate memory efficiency improvement")
        else:
            print("   ⚠️  Async version memory usage similar to original")
    
    # 性能提升建议
    print("\n💡 OPTIMIZATION SUGGESTIONS:")
    
    if async_cpu_avg > orig_cpu_avg * 1.2:
        print("   🚀 Async version shows better CPU utilization")
    
    if speedup > 2.0:
        print("   ✅ Excellent performance improvement with async processing")
    elif speedup > 1.5:
        print("   ✅ Significant performance improvement with async processing")
    elif speedup > 1.1:
        print("   👍 Moderate performance improvement with async processing")
    else:
        print("   ⚠️  Limited performance improvement - consider tuning parameters")
    
    if is_batch and num_videos > 1:
        print(f"   📈 For {num_videos} videos, consider:")
        if speedup < 1.5:
            print("     • Increase max_concurrent_videos")
            print("     • Optimize queue sizes")
        if async_cpu_avg < 70:
            print("     • Increase num_process_workers")
        if async_mem_max > 80:
            print("     • Decrease read_queue_size or max_concurrent_videos")
    
    # 返回分析结果
    analysis = {
        'timing': {
            'original_time': orig_time,
            'async_time': async_time,
            'speedup': speedup,
            'improvement_percent': improvement,
            'num_videos': num_videos,
            'is_batch': is_batch
        },
        'resource_usage': {
            'original': orig_perf,
            'async': async_perf
        },
        'bottleneck_analysis': {
            'original_cpu_bound': orig_cpu_avg > 80,
            'async_cpu_bound': async_cpu_avg > 80,
            'original_io_bound': orig_cpu_avg < 50,
            'async_io_bound': async_cpu_avg < 50,
            'memory_efficient': async_mem_max < orig_mem_max * 0.9 if is_batch else False
        }
    }
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="🔍 Video Preprocessing Performance Analysis (Single & Multi-Video)")
    
    # Input options
    parser.add_argument("--video", type=str, help="Single test video file path")
    parser.add_argument("--input_dir", type=str, help="Directory containing test videos")
    parser.add_argument("--max_test_videos", type=int, default=5, help="Maximum number of videos to test in batch mode")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    
    # Processing parameters
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--use_torch", action="store_true")
    parser.add_argument("--clip_length", type=int, default=17)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--fps", type=int, default=17)
    
    # Original version parameters
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Async version parameters
    parser.add_argument("--read_queue_size", type=int, default=5)
    parser.add_argument("--process_queue_size", type=int, default=10)
    parser.add_argument("--num_process_workers", type=int, default=4)
    parser.add_argument("--num_save_workers", type=int, default=2)
    parser.add_argument("--max_concurrent_videos", type=int, default=2)
    
    # Test options
    parser.add_argument("--skip_original", action="store_true", help="Skip original version test")
    parser.add_argument("--skip_async", action="store_true", help="Skip async version test")
    parser.add_argument("--batch_test", action="store_true", help="Force batch test mode")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.video and not args.input_dir:
        parser.error("Either --video or --input_dir must be specified")
    
    # 确定测试输入
    if args.video:
        video_input = args.video
        test_videos = [args.video]
        is_batch_test = False
    else:
        video_input = args.input_dir
        test_videos = find_test_videos(args.input_dir, args.max_test_videos)
        is_batch_test = True or args.batch_test
    
    if not test_videos:
        print("❌ No test videos found!")
        return
    
    num_videos = len(test_videos)
    
    # 转换参数为字典
    args_dict = {
        'device': args.device,
        'use_torch': args.use_torch,
        'clip_length': args.clip_length,
        'overlap': args.overlap,
        'fps': args.fps,
        'num_workers': args.num_workers,
        'read_queue_size': args.read_queue_size,
        'process_queue_size': args.process_queue_size,
        'num_process_workers': args.num_process_workers,
        'num_save_workers': args.num_save_workers,
        'max_concurrent_videos': args.max_concurrent_videos
    }
    
    print("🔍 Starting performance analysis...")
    print(f"   Input: {video_input}")
    print(f"   Videos to test: {num_videos}")
    print(f"   Mode: {'Batch' if is_batch_test else 'Single'}")
    print(f"   Output: {args.output_dir}")
    print(f"   Device: {args.device}")
    if is_batch_test:
        print(f"   Max concurrent videos: {args.max_concurrent_videos}")
    print()
    
    results = {}
    
    # 运行原版
    if not args.skip_original:
        original_result = run_original_version(video_input, args.output_dir, args_dict, is_batch_test)
        results['original'] = original_result
        
        if not original_result['success']:
            print(f"❌ Original version failed: {original_result['error']}")
    
    # 运行异步版
    if not args.skip_async:
        async_result = run_async_version(video_input, args.output_dir, args_dict, is_batch_test)
        results['async'] = async_result
        
        if not async_result['success']:
            print(f"❌ Async version failed: {async_result['error']}")
    
    # 分析结果
    if 'original' in results and 'async' in results:
        if results['original']['success'] and results['async']['success']:
            analysis = analyze_results(results['original'], results['async'], num_videos)
            results['analysis'] = analysis
        else:
            print("⚠️  Cannot perform comparison - one or both versions failed")
    
    # 保存结果
    results_file = os.path.join(args.output_dir, "performance_analysis.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 添加测试配置到结果中
    results['test_config'] = {
        'video_input': video_input,
        'num_videos': num_videos,
        'is_batch_test': is_batch_test,
        'test_videos': test_videos,
        'parameters': args_dict
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🎉 Performance analysis completed!")

if __name__ == "__main__":
    main() 