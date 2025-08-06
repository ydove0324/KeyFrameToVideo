#torchrun --nproc_per_node=1 src/test/test_load_webdataset.py
import webdataset as wds
import torch
from PIL import Image
import time
import torch.distributed as dist
import os, glob, random, tempfile
import webdataset as wds
from torchvision.io import VideoReader
import decord
import numpy as np
import av   
import io

def make_videoreader_from_bytes(video_bytes):
    # VideoReader 只能读文件，所以写入临时文件
    tmp = tempfile.NamedTemporaryFile(delete=False,dir="/dev/shm", suffix=".mp4")
    tmp.write(video_bytes)
    tmp.flush()
    tmp.seek(0)
    return VideoReader(tmp.name, "video")  # 返回 VideoReader 对象
def pyav_video_from_bytes(video_bytes, decode_all=False):
    """把 bytes 转换成 PyAV 的容器，可一次性解码或按需读取。"""
    container = av.open(io.BytesIO(video_bytes))

    if decode_all:
        frames = []
        for frame in container.decode(video=0):
            arr = frame.to_rgb().to_ndarray()       # (H, W, 3), uint8
            frames.append(torch.from_numpy(arr))
        return torch.stack(frames)                 # (T, H, W, 3)
    else:
        # 返回容器，由用户自己在需要时 decode
        return container
def decord_vr_from_bytes(video_bytes):
    # 把字节流写到内存盘 (Linux) 避免慢磁盘
    tmp = tempfile.NamedTemporaryFile(delete=False, dir="/dev/shm", suffix=".mp4")
    tmp.write(video_bytes)
    tmp.flush()
    tmp.seek(0)
    return decord.VideoReader(tmp.name, ctx=decord.cpu(0))

def cast_video_column(sample):
    if "mp4" in sample:  # key 就是 mp4
        sample["video"] = pyav_video_from_bytes(sample["mp4"],decode_all=True)
    return sample


# 使用webdataset读取tar文件
# dataset_name = "/share/project/huangxu/video-data/koala-36m-clip-1s-webdataset"
dataset_name = "/share/project/zhuoyanluo/data/dataset/coyo700m_facial_data_maniqa_0p4"
# dataset_name = "/share/project/datasets/cc-12m/CC12M_filter"
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
# dataset_name = "/share/project/zhangfan/codes/UltraCapsFusion/result/dedup_tars"
tar_files = sorted(glob.glob(os.path.join(dataset_name, "*.tar")))
# random.shuffle(tar_files)
# 创建 WebDataset，并处理 jpg + json，自动跳过坏样本

import webdataset as wds
import io
from torchvision.io import read_video

SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]


data = (
    wds.WebDataset(tar_files, handler=wds.handlers.ignore_and_continue,nodesplitter=wds.shardlists.split_by_node,workersplitter=wds.shardlists.split_by_worker)
    .decode("pil")   
    # .map(cast_video_column)                  # 把 jpg 解码成 PIL Image
    # .to_tuple("jpg", "json")            # 输出 (image, metadata)
)
# data = data.map(cast_video_column)

data = data.rename(**{"image": "0.png"}) 
# data = data.rename_column("0.png", "image")
# # data = datasets.load_dataset("webdataset",data_files={"train": tar_files},split="train",streaming=True)
# print(isinstance(data,wds.WebDataset))
# print(data)

# data = data.rename_column("mp4", "video")
# data = data.cast_column("video", datasets.Video())
# data = data.rename_column("0.png", "image")
# data = data.cast_column("image", datasets.Image(mode="RGB"))

def safe_iterator(it):
    it = iter(it)
    while True:
        try:
            yield next(it)
        except StopIteration:
            break
        except Exception as e:
            # 打印一下，直接跳过这个样本
            print("Bad sample, skipping...", e)
            continue
# iterator = iter(data)
# data = wds.split_by_node(data)
# data = wds.split_by_worker(data)
iterator = iter(data)
item=0
# item=62256
# t = time.time()
_now = time.time()
for sample in iterator:
    item += 1
    # print(sample["video"])
    # if rank == 0:
        # print(sample)
    # if item <= 100 and rank == 0:
    #     print(item,rank,sample["image"])
    #     os.makedirs("data/coyo-sample-image", exist_ok=True)
    #     image_path = os.path.join("data/coyo-sample-image", f"sample_{item}.png")
        
    #     sample["image"].save(image_path)
    #     print(f"Saved image to {image_path}")

    if item % 1000 == 0:
        print(item, time.time() - _now)
        _now = time.time()
# while True:
#     try:
#         sample = next(iterator)
#         item += 1
#         if item % 1000 == 0:
#             print(item,time.time()-_now)
#             _now = time.time()
#     except StopIteration:
#         break
#     except Exception as e:
#         print("error",item)
#         now = time.time()
#         item += 1
#         iterator = iter(data.skip(item))
#         print(f"time: {now-t}")
#         t = now
#         # print(item)
#         continue
