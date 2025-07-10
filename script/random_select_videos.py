import os
import random
import shutil
import json
from pathlib import Path
import cv2
import numpy as np

def calculate_flow_score(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return 0
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return 0
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_flow = 0
    frame_count = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Initialize flow with zeros of the correct shape
        flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        total_flow += np.mean(magnitude)
        
        prev_gray = gray
        frame_count += 1
    
    cap.release()
    return total_flow / frame_count if frame_count > 1 else 0

# 源目录和目标目录
source_dir = "/share/project/huangxu/video-data/pexel-clips-part2_1_filtered"
target_dir = "demo_videos"

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)
# 创建光流分数>3的视频子目录
gt3_subdir = os.path.join(target_dir, "flow_gt3")
os.makedirs(gt3_subdir, exist_ok=True)

# 获取源目录中的所有视频文件
video_files = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(os.path.join(root, file))

# 如果视频文件数量不足300个，打印警告
if len(video_files) < 300:
    print(f"警告：源目录中只有 {len(video_files)} 个视频文件，少于请求的300个")
    num_to_select = len(video_files)
else:
    num_to_select = 300

# 随机选择100个视频
selected_videos = random.sample(video_files, num_to_select)

# 准备validation数据
validation_data = []

# 复制选中的视频到目标目录并计算光流分数
for video in selected_videos:
    # 先计算光流分数
    flow_score = calculate_flow_score(video)

    # 仅保留光流分数 > 1 的视频
    if flow_score <= 1:
        continue

    # 获取相对于source_dir的路径
    rel_path = os.path.relpath(video, source_dir)
    # 获取父目录名和文件名
    dir_name = os.path.dirname(rel_path).replace(os.sep, '_')
    file_name = os.path.basename(video)
    # 如果有目录名，将其加入到文件名中
    if dir_name:
        new_filename = f"{dir_name}_{file_name}"
    else:
        new_filename = file_name

    # 根据光流分数选择目标目录
    if flow_score > 3:
        dest_dir = gt3_subdir
        video_path_rel = os.path.join("validate_videos", "flow_gt3", new_filename)
    else:
        dest_dir = target_dir
        video_path_rel = os.path.join("validate_videos", new_filename)

    os.makedirs(dest_dir, exist_ok=True)
    destination = os.path.join(dest_dir, new_filename)
    print(f"复制并处理: {new_filename} - flow_score: {flow_score:.2f}")
    shutil.copy2(video, destination)

    # 创建基础数据项
    data_item = {
        "caption": "",
        "image_path": None,
        "video_path": video_path_rel,
        "num_inference_steps": 50,
        "height": 480,
        "width": 832,
        "num_frames": 17,
        "flow_score": float(flow_score)
    }
    validation_data.append(data_item)

# 创建三个不同的validation文件
key_frames_patterns = {
    1: [[0, 16]],
    2: [[0, 8, 16]],
    3: [[0, 4, 8, 12, 16]]
}

for pattern_num, key_frames in key_frames_patterns.items():
    validation_json = {"data": []}
    for item in validation_data:
        item_copy = item.copy()
        item_copy["key_frames_indices"] = key_frames
        validation_json["data"].append(item_copy)
    
    with open(f"validation_{pattern_num}.json", "w") as f:
        json.dump(validation_json, f, indent=2)

print(f"\n完成！已成功处理 {num_to_select} 个视频文件并生成validation文件") 