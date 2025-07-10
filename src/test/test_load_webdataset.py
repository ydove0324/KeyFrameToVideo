import webdataset as wds
import torch
from PIL import Image
import io
import glob
import os
import datasets
import torchvision
import decord
# 使用webdataset读取tar文件
dataset_name = "/share/project/huangxu/video-data/pexel-clips-part2_1_filtered_webdataset"

# 创建dataset并添加错误处理
tar_files = sorted(glob.glob(os.path.join(dataset_name, "*.tar")))
data = datasets.load_dataset("webdataset",data_files={"train": tar_files},split="train",streaming=True)
data = data.rename_column("mp4","video")
data = data.cast_column("video", datasets.Video())

iterator = iter(data)
for i in range(10):
    sample = next(iterator)
    print(sample)
    print(isinstance(sample["video"],(decord.VideoReader, torchvision.io.video_reader.VideoReader)))
    break