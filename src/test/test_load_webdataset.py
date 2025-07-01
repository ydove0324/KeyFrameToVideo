import webdataset as wds
import torch
from PIL import Image
import io
import glob
import os
import datasets

# 使用webdataset读取tar文件
dataset_name = "/share/project/datasets/pixabay_img"

# 创建dataset并添加错误处理
tar_files = sorted(glob.glob(os.path.join(dataset_name, "*.tar")))
data = datasets.load_dataset("webdataset",data_files={"train": tar_files},split="train",streaming=True)