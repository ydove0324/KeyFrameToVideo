# calc_raft_flow_dataset.py  (只展示核心部分)

'''
torchrun --nproc_per_node 4 --master_port 29501 script/calc_raft_flow_webdataset.py \
        --dataset_dir /share/project/huangxu/video-data/pexel-clips-part2_3_filtered_webdataset \
        --out_json /share/project/huangxu/video-data/pexel-clips-part2_3_filtered_webdataset/flow_scores.json \
        --batch_size 4
'''
import os, glob, json, torch, decord, datasets
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from typing import Dict, List
import torch.nn.functional as F
import argparse
from torchvision.models.optical_flow import raft_small
from tqdm import tqdm

# ----------------------------- 分布式工具 ---------------------------------- #
def init_distributed():
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank       = int(os.getenv("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    return rank, world_size, torch.device(f"cuda:{rank}")

# ----------------------------- 数据集构建 ----------------------------------- #
def build_streaming_dataset(dataset_dir: str,
                            rank: int, world_size: int):
    """每个进程只拿部分 tar，天然 shard-aware"""
    shards_all   = sorted(glob.glob(os.path.join(dataset_dir, "*.tar")))
    shards_local = shards_all[rank::world_size]            # round-robin 切片
    if not shards_local:
        raise RuntimeError(f"Rank{rank}: 没分到 tar 文件？")
    ds = datasets.load_dataset(
        "webdataset",
        data_files={"train": shards_local},
        split="train",
        streaming=True,
    )
    ds = ds.rename_column("mp4", "video")          # hf webdataset 里的键
    ds = ds.cast_column("video", datasets.Video()) # 返回 decord.VideoReader
    return ds

# ---------------------- decord → Tensor 预处理函数 -------------------------- #
def preprocess_video(vreader: decord.VideoReader) -> torch.Tensor:
    """
    输出: float32, shape [T, 3, H, W], 值域 [-1, 1]
    1) 一次性解码所有帧
    2) 通道/维度转换
    3) 归一化到 [-1, 1]
    4) (可选) resize / padding 等自己加
    """
    frames = vreader.get_batch(range(len(vreader)))        # (T,H,W,3) uint8
    frames = torch.from_numpy(frames.asnumpy())            # → Tensor uint8
    frames = frames.permute(0, 3, 1, 2).float() / 127.5 - 1.0
    # Resize each frame to half of its original spatial resolution
    _, _, H, W = frames.shape
    frames = F.interpolate(frames, size=(512, 512), mode="bilinear", align_corners=False)
    return frames                                          # 现在在 CPU，后面再 .to(device)

# ------------------------ HF datasets .map() 包装 --------------------------- #
def add_tensor_column(ds):
    def _map_fn(sample: Dict):
        sample["video_tensor"] = preprocess_video(sample["video"])
        return sample
    # keep_in_memory=False -> 流式； batched=False 因为 decord 对象无法批处理
    return ds.map(_map_fn, batched=False)

# ------------------------- DataLoader + collate ---------------------------- #
def collate_fn(samples: List[Dict]):
    # video_tensor 尺寸(T,H,W)不同，直接打包成 list，后面自己循环
    vids  = [s["video_tensor"] for s in samples]
    keys  = [s["__key__"]       for s in samples]
    return vids, keys

def build_loader(ds, batch_size, num_workers=4):
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

# ------------------------------ 主逻辑 ------------------------------------- #
def main(args):
    rank, world_size, device = init_distributed()

    ds_stream = build_streaming_dataset(args.dataset_dir, rank, world_size)
    ds_stream = add_tensor_column(ds_stream)
    loader    = build_loader(ds_stream, batch_size=args.batch_size)

    # === 您可以在这里加载 RAFT（示例省略） ===
    model = raft_small(pretrained=True).to(device).eval()  # 仅示意

    local_scores = {}

    for vids, keys in tqdm(loader, desc="Batches", disable=rank!=0):  # batch_size 条样本
        # vids 是 list[Tensor]; 每条视频长度不同，逐条送入 RAFT
        # 收集当前batch中所有视频的帧对
        all_img1 = []
        all_img2 = []
        vid_lengths = []  # 记录每个视频的帧对数量
        
        for vid_tensor, k in tqdm(list(zip(vids, keys)), total=len(vids), desc="Videos", leave=False, disable=rank!=0):
            vid_tensor = vid_tensor.to(device)   # [T,3,H,W] 归一化到 [-1,1]
            step = 4  # 每隔 step 帧计算一次光流

            # 需要至少 step+1 帧才能形成一对
            if vid_tensor.shape[0] > step:
                # 取下标 \[0,4,8,...] 与 \[4,8,12,...] 组成成对帧
                idx1 = list(range(0, vid_tensor.shape[0] - step, step))
                idx2 = [i + step for i in idx1]

                img1 = vid_tensor[idx1]  # [N, 3, H, W]
                img2 = vid_tensor[idx2]  # [N, 3, H, W]
                
                all_img1.append(img1)
                all_img2.append(img2)
                vid_lengths.append(len(idx1))
            else:
                vid_lengths.append(0)

        # 将所有帧对拼接成一个大batch
        if sum(vid_lengths) > 0:
            img1_batch = torch.cat(all_img1, dim=0)  # [sum(N), 3, H, W]
            img2_batch = torch.cat(all_img2, dim=0)  # [sum(N), 3, H, W]

            with torch.no_grad():
                list_of_flows = model(img1_batch, img2_batch)
                predicted_flows = list_of_flows[-1]  # [sum(N), 2, h, w]

            # 根据每个视频的长度分割结果并计算各自的平均值
            start = 0
            for k, length in zip(keys, vid_lengths):
                if length > 0:
                    end = start + length
                    video_flows = predicted_flows[start:end]  # [N, 2, h, w]
                    flow_score = video_flows.norm(dim=1).mean().item()
                    local_scores[k] = flow_score
                    print(flow_score, k)
                    start = end
                else:
                    local_scores[k] = 0.0
        else:
            # 如果batch中所有视频都太短，全部记为0分
            for k in keys:
                local_scores[k] = 0.0

            local_scores[k] = flow_score

    # ---- 分布式 all_gather 到 rank0，写 json -----
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_scores)
        if rank == 0:
            merged = {}
            for d in gathered:
                if d: merged.update(d)
            with open(args.out_json, "w") as f:
                json.dump(merged, f, indent=2)
        dist.barrier()  # 可选
    elif rank == 0:
        with open(args.out_json, "w") as f:
            json.dump(local_scores, f, indent=2)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    main(args)