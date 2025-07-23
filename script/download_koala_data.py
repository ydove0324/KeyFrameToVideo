'''
Koala数据下载和处理脚本

功能：
1. 从多个CSV文件读取videoID和时间戳信息
2. 下载视频文件
3. 根据时间戳裁剪视频片段
4. 对top_30_percent中的片段进行1秒裁剪
5. 多线程处理下载和裁剪任务
'''

import csv
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Set, Dict, List, Tuple
import logging
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 为decord和torchvision添加导入
try:
    import decord
    import torch
    from torchvision.io import write_video
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_koala.log'),
        logging.StreamHandler()
    ]
)

def calculate_duration(start_time, end_time):
    # 解析时间字符串（支持 HH:MM:SS.sss）
    fmt = "%H:%M:%S.%f" if "." in start_time else "%H:%M:%S"
    t1 = datetime.strptime(start_time, fmt)
    t2 = datetime.strptime(end_time, fmt)
    
    # 计算时间差（返回 timedelta 对象）
    delta = t2 - t1
    
    # 获取总秒数（float）
    return delta.total_seconds()

class KoalaDataProcessor:
    def __init__(self, world_size=1, node_rank=0):
        # 分布式参数
        self.world_size = world_size
        self.node_rank = node_rank
        
        # 文件路径
        self.csv_paths = [
            # "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_1.csv",
            # "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_2.csv",
            # "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_3.csv",
            # "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_4.csv",
            # "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_5.csv",
            # "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_6.csv",
            # "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_7.csv",
            # "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_8.csv",
            # "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_9.csv",
            # "/share/project/huangxu/workspace/tmp/koala-36m_metadata/Koala-36M-v1_20250529/Koala_36M_10.csv",
            "/share/project/huangxu/workspace/KeyFrameToVideo/top_30_percent_koala.csv"
        ]
        self.top_30_percent_csv = "/share/project/huangxu/workspace/KeyFrameToVideo/top_30_percent_koala.csv"
        self.all_videos_file = "/share/project/huangxu/workspace/tmp/all_videos.txt"
        self.all_videos_part2_file = "/share/project/huangxu/workspace/tmp/all_videos_part2.txt"
        
        # 目录路径
        self.download_dir = "/share/project/huangxu/video-data/koala-36m/"
        self.clip_dir = "/share/project/huangxu/video-data/koala-36m-clip/"
        self.clip_1s_dir = "/share/project/huangxu/video-data/koala-36m-clip-1s/"
        
        # 数据存储
        self.video_timestamps: Dict[str, List[Tuple[str, List[str]]]] = {}  # videoID -> [(segment_id, [start_time, end_time])]
        self.target_video_ids: Set[str] = set()
        self.downloaded_video_ids: Set[str] = set()
        self.top_30_percent_segments: Set[str] = set()
        
        # 线程控制
        self.max_concurrent_downloads = 10
        self.max_concurrent_clips = 5
        self.lock = threading.Lock()
        
        # 确保目录存在
        for dir_path in [self.download_dir, self.clip_dir, self.clip_1s_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def parse_timestamp_string(self, timestamp_str: str) -> List[str]:
        """解析时间戳字符串，返回[start_time, end_time]"""
        # 移除引号和方括号
        timestamp_str = timestamp_str.strip('[]"\'')
        # 分割时间戳，并移除多余的引号
        timestamps = [t.strip().strip('"\'').replace("'", "") for t in timestamp_str.split("', '")]
        return timestamps
    
    def load_video_timestamps_from_csvs(self):
        """从多个CSV文件加载videoID和时间戳信息"""
        logging.info("开始从CSV文件加载videoID和时间戳信息...")
        
        for csv_path in self.csv_paths:
            if not os.path.exists(csv_path):
                logging.warning(f"CSV文件不存在: {csv_path}")
                continue
                
            logging.info(f"处理CSV文件: {csv_path}")
            
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in tqdm(reader, desc=f"处理 {os.path.basename(csv_path)}"):
                        video_id_full = row['videoID']
                        
                        # 提取基础videoID和片段号
                        match = re.match(r'(.+?)_(\d+)$', video_id_full)
                        if not match:
                            continue
                            
                        base_video_id = match.group(1)
                        segment_id = match.group(2)
                        
                        # 解析时间戳
                        timestamp_str = row['timestamp']
                        try:
                            timestamps = self.parse_timestamp_string(timestamp_str)
                            if len(timestamps) == 2:
                                if base_video_id not in self.video_timestamps:
                                    self.video_timestamps[base_video_id] = []
                                self.video_timestamps[base_video_id].append((segment_id, timestamps))
                        except Exception as e:
                            logging.warning(f"解析时间戳失败 {video_id_full}: {e}")
                            
            except Exception as e:
                logging.error(f"读取CSV文件失败 {csv_path}: {e}")
        
        logging.info(f"从CSV文件加载了 {len(self.video_timestamps)} 个videoID的时间戳信息")
    
    def get_target_video_ids_from_top30(self):
        """从top30文件中提取需要下载的videoID"""
        logging.info("从top30文件中提取目标videoID...")
        
        if not os.path.exists(self.top_30_percent_csv):
            logging.warning(f"top_30_percent文件不存在: {self.top_30_percent_csv}")
            return
        
        try:
            with open(self.top_30_percent_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_id_full = row['videoID']
                    
                    # 提取基础videoID
                    match = re.match(r'(.+?)_\d+$', video_id_full)
                    if match:
                        base_video_id = match.group(1)
                        self.target_video_ids.add(base_video_id)
                        self.top_30_percent_segments.add(video_id_full)
            
            logging.info(f"从top30文件中提取了 {len(self.target_video_ids)} 个目标videoID")
            logging.info(f"包含 {len(self.top_30_percent_segments)} 个top30片段")
        except Exception as e:
            logging.error(f"读取top_30_percent文件失败: {e}")
    
    def load_top_30_percent_segments(self):
        """加载top_30_percent_koala.csv中的片段信息（已合并到get_target_video_ids_from_top30中）"""
        pass
    
    def init_downloaded_video_ids(self):
        """初始化已下载的videoID列表"""
        logging.info("初始化已下载的videoID列表...")
        
        # 检查下载目录
        if os.path.exists(self.download_dir):
            for file_path in Path(self.download_dir).rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.mkv', '.webm']:
                    filename = file_path.stem
                    video_id = filename.replace('_video_audio', '')
                    self.downloaded_video_ids.add(video_id)
        
        # 检查clip目录
        if os.path.exists(self.clip_dir):
            for file_path in Path(self.clip_dir).rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.mkv', '.webm']:
                    filename = file_path.stem
                    # 提取基础videoID（去掉片段号）
                    match = re.match(r'(.+?)_\d+$', filename)
                    if match:
                        base_video_id = match.group(1)
                        self.downloaded_video_ids.add(base_video_id)
        
        logging.info(f"找到 {len(self.downloaded_video_ids)} 个已下载的视频")
    
    def extract_video_id_from_object_name(self, object_name: str) -> str:
        """从ObjectName中提取videoID"""
        filename = os.path.basename(object_name)
        video_id = filename.replace('_video_audio.mp4', '').replace('_video_audio.mkv', '').replace('_video_audio.webm', '')
        return video_id
    
    def get_ks3_command(self, object_name: str, local_path: str) -> str:
        """根据bucket类型生成相应的ks3下载命令"""
        if "lingjiang-datasets" in object_name:
            return f'/share/project/huangxu/ks3util-linux-amd64 --config-file /share/project/zhangfan/misc/ks3lingjiangconfig cp "{object_name}" "{local_path}"'
        elif "dorc-beisai-2025" in object_name:
            return f'/share/project/huangxu/ks3util-linux-amd64 cp "{object_name}" "{local_path}"'
        else:
            raise ValueError(f"未知的bucket类型: {object_name}")
    
    def download_video(self, object_name: str, video_id: str) -> bool:
        """下载单个视频"""
        try:
            filename = os.path.basename(object_name)
            local_path = os.path.join(self.download_dir, filename)
            
            command = self.get_ks3_command(object_name, local_path)
            
            logging.info(f"开始下载 {video_id}")
            
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"成功下载 {video_id}")
                return True
            else:
                logging.error(f"下载失败 {video_id}: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"下载 {video_id} 时发生错误: {e}")
            return False
    
    def clip_video_segment(self, video_id: str, segment_id: str, start_time: str, end_time: str) -> bool:
        """裁剪视频片段"""
        try:
            # 查找视频文件
            video_file = None
            for ext in ['.mp4', '.mkv', '.webm']:
                potential_file = os.path.join(self.download_dir, f"{video_id}_video_audio{ext}")
                if os.path.exists(potential_file):
                    video_file = potential_file
                    break
            
            if not video_file:
                logging.warning(f"找不到视频文件: {video_id}")
                return False
            
            # 输出文件路径
            output_file = os.path.join(self.clip_dir, f"{video_id}_{segment_id}.mp4")
            
            duration = str(int(calculate_duration(start_time, end_time)))
            command = [
                'ffmpeg', '-y',  # -y表示覆盖输出文件
                '-ss', start_time,
                '-i', video_file,
                '-t', duration,  # 使用-t指定持续时长
                '-c', 'copy',  # 使用流复制，不重新编码
                output_file
            ]
            
            logging.info(f"开始裁剪 {video_id}_{segment_id}: {start_time} -> {end_time}")
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"成功裁剪 {video_id}_{segment_id},成功移除{video_file}")
                # 删除原视频文件
                os.remove(video_file)
                return True
            else:
                logging.error(f"裁剪失败 {video_id}_{segment_id}: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"裁剪 {video_id}_{segment_id} 时发生错误: {e}")
            return False
    
    def clip_video_to_1s_segments(self, clip_file: str) -> bool:
        """将视频片段裁剪成1秒的片段，使用decord"""
        if not DECORD_AVAILABLE:
            logging.error("decord或torchvision未安装，请安装: pip install decord torchvision")
            return False
            
        try:
            # 使用decord读取视频
            vr = decord.VideoReader(clip_file)
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            duration = total_frames / fps
            total_seconds = int(duration)
            
            logging.info(f"视频时长: {duration:.2f}秒, FPS: {fps:.2f}, 总帧数: {total_frames}, 将创建 {total_seconds} 个1秒片段")
            
            # 为每一秒创建片段
            for i in range(total_seconds):
                start_frame = int(i * fps)
                end_frame = int((i + 1) * fps)
                
                # 确保不超出视频范围
                if end_frame > total_frames:
                    end_frame = total_frames
                
                # 读取这一秒的帧
                frames = vr.get_batch(range(start_frame, end_frame))
                
                # 转换为torch tensor
                if not isinstance(frames, torch.Tensor):
                    frames = torch.from_numpy(frames.asnumpy() if hasattr(frames, "asnumpy") else frames)
                
                # 输出文件名
                base_name = os.path.splitext(os.path.basename(clip_file))[0]
                output_file = os.path.join(self.clip_1s_dir, f"{base_name}_{i+1:03d}.mp4")
                
                logging.info(f"创建1秒片段: {output_file} (帧 {start_frame}-{end_frame})")
                
                # 使用torchvision写入视频,使用无损编码
                try:
                    write_video(output_file, frames, fps=int(fps), video_codec="libx264", options={
                        "crf": "20",  # 无损质量
                        "preset": "slow"  # 最高压缩效率
                    })
                except Exception as e:
                    logging.error(f"创建1秒片段失败: {output_file}, 错误: {e}")
                    return False
            
            logging.info(f"成功创建 {total_seconds} 个1秒片段: {clip_file}")
            return True
            
        except Exception as e:
            logging.error(f"创建1秒片段时发生错误: {e}")
            return False
    
    def process_downloaded_video(self, video_id: str):
        """处理已下载的视频：裁剪片段并检查是否需要1秒裁剪"""
        if video_id not in self.video_timestamps:
            return
        
        for segment_id, timestamps in self.video_timestamps[video_id]:
            start_time, end_time = timestamps
            
            # 裁剪片段
            if self.clip_video_segment(video_id, segment_id, start_time, end_time):
                clip_filename = f"{video_id}_{segment_id}.mp4"
                clip_file = os.path.join(self.clip_dir, clip_filename)
                
                # 检查是否在top_30_percent中
                full_segment_id = f"{video_id}_{segment_id}"
                if full_segment_id in self.top_30_percent_segments:
                    logging.info(f"发现top_30_percent片段: {full_segment_id}")
                    self.clip_video_to_1s_segments(clip_file)
    
    def process_video_list_file(self, file_path: str):
        """处理视频列表文件"""
        logging.info(f"开始处理文件: {file_path}")
        
        if not os.path.exists(file_path):
            logging.warning(f"文件不存在: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                next(f)  # 跳过标题行
                lines = list(f)
                download_candidates = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    object_name = parts[-1]
                    
                    if not any(ext in object_name.lower() for ext in ['.mp4', '.mkv', '.webm']):
                        continue
                    
                    video_id = self.extract_video_id_from_object_name(object_name)
                    
                    if video_id in self.target_video_ids and video_id not in self.downloaded_video_ids:
                        download_candidates.append((object_name, video_id))
                
                if download_candidates:
                    with tqdm(total=len(download_candidates), desc=f"下载 {os.path.basename(file_path)}") as pbar:
                        with ThreadPoolExecutor(max_workers=self.max_concurrent_downloads) as executor:
                            future_to_video = {
                                executor.submit(self.download_and_process_video, object_name, video_id): (object_name, video_id)
                                for object_name, video_id in download_candidates
                            }
                            
                            for future in as_completed(future_to_video):
                                object_name, video_id = future_to_video[future]
                                try:
                                    success = future.result()
                                    if success:
                                        pbar.update(1)
                                        pbar.set_postfix_str(f"成功: {video_id}")
                                    else:
                                        pbar.set_postfix_str(f"失败: {video_id}")
                                except Exception as e:
                                    pbar.set_postfix_str(f"错误: {video_id}")
                                    logging.error(f"处理 {video_id} 时发生错误: {e}")
                else:
                    logging.info(f"文件 {os.path.basename(file_path)} 中没有需要下载的视频")
                    
        except Exception as e:
            logging.error(f"处理文件 {file_path} 时发生错误: {e}")
    
    def download_and_process_video(self, object_name: str, video_id: str) -> bool:
        """下载视频并立即处理"""
        # 下载视频
        if self.download_video(object_name, video_id):
            with self.lock:
                self.downloaded_video_ids.add(video_id)
            
            # 立即处理视频
            self.process_downloaded_video(video_id)
            return True
        return False
    
    def partition_target_video_ids(self):
        """根据world_size和node_rank划分目标videoID"""
        if self.world_size <= 1:
            logging.info("单机模式，不进行数据划分")
            return
        
        # 将target_video_ids转换为有序列表
        target_video_list = sorted(list(self.target_video_ids))
        total_videos = len(target_video_list)
        
        # 计算每个节点应该处理的视频数量
        videos_per_node = total_videos // self.world_size
        remainder = total_videos % self.world_size
        
        # 计算当前节点的起始和结束索引
        start_idx = self.node_rank * videos_per_node + min(self.node_rank, remainder)
        end_idx = start_idx + videos_per_node + (1 if self.node_rank < remainder else 0)
        
        # 获取当前节点应该处理的视频ID
        node_video_ids = set(target_video_list[start_idx:end_idx])
        
        # 更新target_video_ids为当前节点负责的部分
        self.target_video_ids = node_video_ids
        
        logging.info(f"节点 {self.node_rank}/{self.world_size} 负责处理 {len(node_video_ids)} 个视频")
        logging.info(f"视频ID范围: {start_idx}-{end_idx-1} (总共 {total_videos} 个视频)")
        
        # 同时更新top_30_percent_segments，只保留当前节点负责的片段
        node_segments = set()
        for segment_id in self.top_30_percent_segments:
            # 提取基础videoID
            match = re.match(r'(.+?)_\d+$', segment_id)
            if match:
                base_video_id = match.group(1)
                if base_video_id in node_video_ids:
                    node_segments.add(segment_id)
        
        self.top_30_percent_segments = node_segments
        logging.info(f"节点 {self.node_rank} 负责的top_30_percent片段数量: {len(node_segments)}")

    def process_existing_videos(self):
        """处理已存在的视频文件，进行clip裁剪"""
        logging.info("处理已存在的视频文件...")
        
        if not os.path.exists(self.download_dir):
            return
        
        video_files = []
        for file_path in Path(self.download_dir).rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.mkv', '.webm']:
                # 提取videoID
                filename = file_path.stem
                video_id = filename.replace('_video_audio', '')
                
                # 只处理目标videoID
                if video_id in self.target_video_ids:
                    # 检查是否已经裁剪过了
                    all_segments_exist = True
                    if video_id in self.video_timestamps:
                        for segment_id, _ in self.video_timestamps[video_id]:
                            clip_filename = f"{video_id}_{segment_id}.mp4"
                            clip_file = os.path.join(self.clip_dir, clip_filename)
                            if not os.path.exists(clip_file):
                                all_segments_exist = False
                                break
                    
                    if not all_segments_exist:
                        video_files.append((str(file_path), video_id))
                    else:
                        logging.info(f"视频 {video_id} 的所有片段已存在，跳过处理")
        
        with tqdm(total=len(video_files), desc="处理已存在视频") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_concurrent_clips) as executor:
                future_to_file = {
                    executor.submit(self.process_existing_video, file_path, video_id): (file_path, video_id)
                    for file_path, video_id in video_files
                }
                
                for future in as_completed(future_to_file):
                    file_path, video_id = future_to_file[future]
                    try:
                        future.result()
                        pbar.update(1)
                    except Exception as e:
                        logging.error(f"处理视频文件时发生错误: {e}")
    
    def process_existing_video(self, video_file: str, video_id: str):
        """处理单个已存在的视频文件"""
        if video_id not in self.video_timestamps:
            return
        
        for segment_id, timestamps in self.video_timestamps[video_id]:
            start_time, end_time = timestamps
            
            # 裁剪片段
            if self.clip_video_segment_from_file(video_file, video_id, segment_id, start_time, end_time):
                clip_filename = f"{video_id}_{segment_id}.mp4"
                clip_file = os.path.join(self.clip_dir, clip_filename)
                
                # 检查是否在top_30_percent中
                full_segment_id = f"{video_id}_{segment_id}"
                if full_segment_id in self.top_30_percent_segments:
                    logging.info(f"发现top_30_percent片段: {full_segment_id}")
                    self.clip_video_to_1s_segments(clip_file)
    
    def clip_video_segment_from_file(self, video_file: str, video_id: str, segment_id: str, start_time: str, end_time: str) -> bool:
        """从指定文件裁剪视频片段"""
        try:
            # 输出文件路径
            output_file = os.path.join(self.clip_dir, f"{video_id}_{segment_id}.mp4")
            
            # 计算时长（秒）
            duration = str(int(calculate_duration(start_time, end_time)))
            command = [
                'ffmpeg', '-y',  # -y表示覆盖输出文件
                '-ss', start_time,
                '-i', video_file,
                '-t', duration,  # 使用-t指定持续时长
                '-c', 'copy',  # 使用流复制，不重新编码
                output_file
            ]
            
            logging.info(f"开始裁剪 {video_id}_{segment_id}: {start_time} -> {end_time}")
            logging.info(f"command: {command}")
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"成功裁剪 {video_id}_{segment_id}")
                return True
            else:
                logging.error(f"裁剪失败 {video_id}_{segment_id}: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"裁剪 {video_id}_{segment_id} 时发生错误: {e}")
            return False
    
    def run(self):
        """主运行函数"""
        logging.info("开始Koala数据处理任务")
        
        # 1. 加载CSV文件中的videoID和时间戳信息
        self.load_video_timestamps_from_csvs()
        
        # 2. 从top30文件中提取目标videoID
        self.get_target_video_ids_from_top30()

        # 3. 根据world_size和node_rank划分目标videoID
        self.partition_target_video_ids()
        
        # 4. 初始化已下载的videoID
        self.init_downloaded_video_ids()
        
        # 5. 处理已存在的视频文件
        self.process_existing_videos()
        
        # 6. 统计信息
        total_target = len(self.target_video_ids)
        total_downloaded = len(self.downloaded_video_ids)
        total_remaining = total_target - total_downloaded
        
        logging.info(f"总体进度: {total_downloaded}/{total_target} ({total_downloaded/total_target*100:.1f}%)")
        logging.info(f"待下载数量: {total_remaining}")
        
        if total_remaining > 0:
            # 7. 处理视频列表文件
            self.process_video_list_file(self.all_videos_file)
            self.process_video_list_file(self.all_videos_part2_file)
        
        # 最终统计
        final_downloaded = len(self.downloaded_video_ids)
        final_remaining = total_target - final_downloaded
        
        logging.info("数据处理任务完成")
        logging.info(f"目标videoID数量: {total_target}")
        logging.info(f"已下载videoID数量: {final_downloaded}")
        logging.info(f"待下载videoID数量: {final_remaining}")
        if total_target > 0:
            logging.info(f"最终进度: {final_downloaded}/{total_target} ({final_downloaded/total_target*100:.1f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Koala数据处理脚本')
    parser.add_argument('--world_size', type=int, default=1, help='总节点数量 (默认: 1)')
    parser.add_argument('--node_rank', type=int, default=0, help='当前节点排名 (默认: 0)')
    
    args = parser.parse_args()
    
    logging.info(f"启动Koala数据处理 - 节点 {args.node_rank}/{args.world_size}")
    
    processor = KoalaDataProcessor(world_size=args.world_size, node_rank=args.node_rank)
    processor.run()