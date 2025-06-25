import os
import random
import glob
import torch
import numpy as np
from PIL import Image
import decord
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict
import cv2

# 导入必要的模块
from WanKeyFrame2VideoPipeline import WanKeyFrame2VideoPipeline
from finetrainers.models.wan.transformer_wan import WanTransformer3DModel,T2VModel2I2VModelConverter
from diffusers import AutoencoderKLWan
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video

# 导入视频评估指标
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: FID not available. Install with: pip install pytorch-fid")

class VideoMetrics:
    """视频质量评估指标计算类"""
    
    def __init__(self, device='cuda'):
        self.device = device
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
    def extract_frames_from_video_cv2(self, video_path: str) -> List[np.ndarray]:
        """使用OpenCV从视频中提取所有帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV读取的是BGR，转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """将帧列表转换为tensor，归一化到[-1,1]"""
        # frames: List[np.ndarray] with shape (H, W, C) and values in [0, 255]
        frames_tensor = torch.stack([
            torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            for frame in frames
        ])  # Shape: (T, C, H, W)
        
        # 归一化到[-1, 1]
        frames_tensor = frames_tensor * 2.0 - 1.0
        return frames_tensor.to(self.device)
    
    def resize_frames(self, frames_tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """调整帧尺寸"""
        import torch.nn.functional as F
        # frames_tensor: (T, C, H, W)
        resized = F.interpolate(frames_tensor, size=target_size, mode='bilinear', align_corners=False)
        return resized
    
    def calculate_frame_lpips(self, real_frame: np.ndarray, gen_frame: np.ndarray) -> float:
        """计算两个单帧之间的LPIPS分数"""
        if not LPIPS_AVAILABLE:
            return -1.0
        
        # 转换为tensor
        real_tensor = self.frames_to_tensor([real_frame])  # (1, C, H, W)
        gen_tensor = self.frames_to_tensor([gen_frame])    # (1, C, H, W)
        
        # 调整尺寸到相同大小
        if real_tensor.shape != gen_tensor.shape:
            target_size = (min(real_tensor.shape[2], gen_tensor.shape[2]), 
                          min(real_tensor.shape[3], gen_tensor.shape[3]))
            real_tensor = self.resize_frames(real_tensor, target_size)
            gen_tensor = self.resize_frames(gen_tensor, target_size)
        
        # 计算LPIPS
        with torch.no_grad():
            score = self.lpips_fn(real_tensor, gen_tensor)
            return score.item()
    
    def calculate_frame_psnr(self, real_frame: np.ndarray, gen_frame: np.ndarray) -> float:
        """计算两个单帧之间的PSNR"""
        real_frame = real_frame.astype(np.float32)
        gen_frame = gen_frame.astype(np.float32)
        
        # 调整尺寸
        if real_frame.shape != gen_frame.shape:
            target_shape = (min(real_frame.shape[0], gen_frame.shape[0]),
                           min(real_frame.shape[1], gen_frame.shape[1]))
            real_frame = cv2.resize(real_frame, (target_shape[1], target_shape[0]))
            gen_frame = cv2.resize(gen_frame, (target_shape[1], target_shape[0]))
        
        mse = np.mean((real_frame - gen_frame) ** 2)
        if mse == 0:
            return float('inf')
        else:
            return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def calculate_frame_ssim(self, real_frame: np.ndarray, gen_frame: np.ndarray) -> float:
        """计算两个单帧之间的SSIM"""
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            print("Warning: scikit-image not available. Install with: pip install scikit-image")
            return -1.0
        
        # 调整尺寸
        if real_frame.shape != gen_frame.shape:
            target_shape = (min(real_frame.shape[0], gen_frame.shape[0]),
                           min(real_frame.shape[1], gen_frame.shape[1]))
            real_frame = cv2.resize(real_frame, (target_shape[1], target_shape[0]))
            gen_frame = cv2.resize(gen_frame, (target_shape[1], target_shape[0]))
        
        # 转换为灰度图像计算SSIM
        real_gray = cv2.cvtColor(real_frame, cv2.COLOR_RGB2GRAY)
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_RGB2GRAY)
        
        score = ssim(real_gray, gen_gray, data_range=255)
        return score
    
    def calculate_lpips(self, real_video_path: str, generated_video_path: str) -> float:
        """计算两个视频之间的LPIPS分数"""
        if not LPIPS_AVAILABLE:
            return -1.0
        
        # 提取帧
        real_frames = self.extract_frames_from_video_cv2(real_video_path)
        gen_frames = self.extract_frames_from_video_cv2(generated_video_path)
        
        # 确保帧数一致
        min_frames = min(len(real_frames), len(gen_frames))
        real_frames = real_frames[:min_frames]
        gen_frames = gen_frames[:min_frames]
        
        if min_frames == 0:
            return -1.0
        
        # 转换为tensor
        real_tensor = self.frames_to_tensor(real_frames)
        gen_tensor = self.frames_to_tensor(gen_frames)
        
        # 调整尺寸到相同大小
        if real_tensor.shape != gen_tensor.shape:
            target_size = (min(real_tensor.shape[2], gen_tensor.shape[2]), 
                          min(real_tensor.shape[3], gen_tensor.shape[3]))
            real_tensor = self.resize_frames(real_tensor, target_size)
            gen_tensor = self.resize_frames(gen_tensor, target_size)
        
        # 计算LPIPS
        lpips_scores = []
        with torch.no_grad():
            for i in range(min_frames):
                real_frame = real_tensor[i:i+1]  # (1, C, H, W)
                gen_frame = gen_tensor[i:i+1]    # (1, C, H, W)
                
                score = self.lpips_fn(real_frame, gen_frame)
                lpips_scores.append(score.item())
        
        return np.mean(lpips_scores)
    
    def calculate_psnr(self, real_video_path: str, generated_video_path: str) -> float:
        """计算PSNR"""
        real_frames = self.extract_frames_from_video_cv2(real_video_path)
        gen_frames = self.extract_frames_from_video_cv2(generated_video_path)
        
        min_frames = min(len(real_frames), len(gen_frames))
        if min_frames == 0:
            return -1.0
        
        psnr_scores = []
        for i in range(min_frames):
            real_frame = real_frames[i].astype(np.float32)
            gen_frame = gen_frames[i].astype(np.float32)
            
            # 调整尺寸
            if real_frame.shape != gen_frame.shape:
                target_shape = (min(real_frame.shape[0], gen_frame.shape[0]),
                               min(real_frame.shape[1], gen_frame.shape[1]))
                real_frame = cv2.resize(real_frame, (target_shape[1], target_shape[0]))
                gen_frame = cv2.resize(gen_frame, (target_shape[1], target_shape[0]))
            
            mse = np.mean((real_frame - gen_frame) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            psnr_scores.append(psnr)
        
        return np.mean(psnr_scores)
    
    def calculate_ssim(self, real_video_path: str, generated_video_path: str) -> float:
        """计算SSIM"""
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            print("Warning: scikit-image not available. Install with: pip install scikit-image")
            return -1.0
        
        real_frames = self.extract_frames_from_video_cv2(real_video_path)
        gen_frames = self.extract_frames_from_video_cv2(generated_video_path)
        
        min_frames = min(len(real_frames), len(gen_frames))
        if min_frames == 0:
            return -1.0
        
        ssim_scores = []
        for i in range(min_frames):
            real_frame = real_frames[i]
            gen_frame = gen_frames[i]
            
            # 调整尺寸
            if real_frame.shape != gen_frame.shape:
                target_shape = (min(real_frame.shape[0], gen_frame.shape[0]),
                               min(real_frame.shape[1], gen_frame.shape[1]))
                real_frame = cv2.resize(real_frame, (target_shape[1], target_shape[0]))
                gen_frame = cv2.resize(gen_frame, (target_shape[1], target_shape[0]))
            
            # 转换为灰度图像计算SSIM
            real_gray = cv2.cvtColor(real_frame, cv2.COLOR_RGB2GRAY)
            gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_RGB2GRAY)
            
            score = ssim(real_gray, gen_gray, data_range=255)
            ssim_scores.append(score)
        
        return np.mean(ssim_scores)

def extract_frames_from_video(video_path, first_frame_idx=0, last_frame_idx=16):
    """从视频中提取指定帧并转换为PIL Image"""
    # 使用decord读取视频
    vr = decord.VideoReader(video_path)
    
    # 检查帧索引是否有效
    total_frames = len(vr)
    if first_frame_idx >= total_frames:
        raise ValueError(f"第一帧索引{first_frame_idx}超出视频总帧数{total_frames}")
    if last_frame_idx >= total_frames:
        raise ValueError(f"最后一帧索引{last_frame_idx}超出视频总帧数{total_frames}")
    
    # 读取指定帧
    frame_indices = [first_frame_idx, last_frame_idx]
    frames = vr.get_batch(frame_indices)  # torch.Tensor, shape: (2, H, W, C)
    
    # 将torch.Tensor转换为numpy数组
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    
    # 确保数据类型正确（uint8，范围0-255）
    if frames.dtype != np.uint8:
        # 如果是浮点数类型且在[0,1]范围内，则乘以255
        if frames.dtype in [np.float32, np.float64] and frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    
    # 转换为PIL Image
    first_pil = Image.fromarray(frames[0])
    last_pil = Image.fromarray(frames[1])
    
    return first_pil, last_pil

def load_pipeline(model_id: str, transformer_path: str, device: str = "cuda"):
    """加载模型管道"""
    print(f"Loading model from {transformer_path}...")
    
    # 加载transformer
    transformer = WanTransformer3DModel.from_pretrained(
        transformer_path, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    )
    
    # 转换模型
    converter = T2VModel2I2VModelConverter(transformer)
    converter.convert()
    
    # 加载其他组件
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.bfloat16)
    image_processor = CLIPImageProcessor.from_pretrained(model_id, subfolder="image_processor")
    scheduler = FlowMatchEulerDiscreteScheduler()
    
    # 创建管道
    pipe = WanKeyFrame2VideoPipeline(
        transformer=transformer,
        vae=vae,
        image_encoder=image_encoder,
        image_processor=image_processor,
        scheduler=scheduler,
    ).to(device)
    
    return pipe

def get_random_videos(dataset_path: str, num_videos: int = 20) -> List[str]:
    """从数据集中随机选择视频"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    all_videos = []
    
    for ext in video_extensions:
        videos = glob.glob(os.path.join(dataset_path, ext))
        all_videos.extend(videos)
    
    if len(all_videos) < num_videos:
        print(f"Warning: Only found {len(all_videos)} videos in {dataset_path}, requested {num_videos}")
        return all_videos
    
    return random.sample(all_videos, num_videos)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Validate video generation models")
    parser.add_argument("--load_from", type=str, default=None, 
                       help="Path to test_videos.json file to load existing test video list")
    args = parser.parse_args()
    
    # 配置参数
    model_id = "/share/project/huangxu/Wan2.1-T2V-1.3B-diffusers"
    base_transformer_path = "/share/project/huangxu/wan-t2v-pexel-part2_0/model_weights"
    
    # 模型步数列表
    model_steps = [1500, 3000, 4500, 6000, 7500, 9000, 10500, 12000]
    
    # 数据集路径
    dataset_paths = [
        "pexel_part2_0",
        "pexel_part2_1"
    ]
    
    # 创建输出目录
    output_dir = Path("validate_results")
    output_dir.mkdir(exist_ok=True)
    
    # 加载encoder_hidden_states
    encoder_hidden_states = torch.load("debug_tensors/encoder_hidden_states_t1000.pt").to("cuda")
    
    # 初始化视频评估指标
    metrics = VideoMetrics(device="cuda")
    
    # 收集所有测试视频
    if args.load_from:
        # 从指定的JSON文件加载测试视频列表
        print(f"Loading test videos from: {args.load_from}")
        try:
            with open(args.load_from, "r") as f:
                test_videos_info = json.load(f)
            
            all_test_videos = [(video_info["path"], video_info["dataset"]) 
                             for video_info in test_videos_info["videos"]]
            
            # 验证视频文件是否存在
            valid_videos = []
            for video_path, dataset in all_test_videos:
                if os.path.exists(video_path):
                    valid_videos.append((video_path, dataset))
                else:
                    print(f"Warning: Video file not found: {video_path}")
            
            all_test_videos = valid_videos
            print(f"Loaded {len(all_test_videos)} valid test videos from JSON file")
            
        except Exception as e:
            print(f"Error loading test videos from {args.load_from}: {e}")
            print("Falling back to random selection...")
            all_test_videos = []
            for dataset_path in dataset_paths:
                videos = get_random_videos(dataset_path, num_videos=20)
                all_test_videos.extend([(video, dataset_path) for video in videos])
    else:
        # 随机选择测试视频
        print("Randomly selecting test videos...")
        all_test_videos = []
        for dataset_path in dataset_paths:
            videos = get_random_videos(dataset_path, num_videos=20)
            all_test_videos.extend([(video, dataset_path) for video in videos])
    
    print(f"Total test videos: {len(all_test_videos)}")
    
    # 保存测试视频列表（如果不是从文件加载的话）
    if not args.load_from:
        test_videos_info = {
            "videos": [{"path": video, "dataset": dataset} for video, dataset in all_test_videos],
            "model_steps": model_steps
        }
        
        with open(output_dir / "test_videos.json", "w") as f:
            json.dump(test_videos_info, f, indent=2)
        print(f"Test video list saved to: {output_dir / 'test_videos.json'}")
    
    # 存储所有结果
    all_results = {}
    
    # 对每个模型步数进行测试
    for step in model_steps:
        step_str = f"{step:06d}"
        transformer_path = os.path.join(base_transformer_path, step_str)
        
        if not os.path.exists(transformer_path):
            print(f"Warning: Model path {transformer_path} does not exist, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing model step: {step} ({step_str})")
        print(f"{'='*60}")
        
        # 创建该步数的输出目录
        step_output_dir = output_dir / f"step_{step_str}"
        step_output_dir.mkdir(exist_ok=True)
        
        # 加载模型
        try:
            pipe = load_pipeline(model_id, transformer_path)
        except Exception as e:
            print(f"Failed to load model for step {step}: {e}")
            continue
        
        step_results = {
            "step": step,
            "videos": [],
            "summary": {}
        }
        
        # 对每个测试视频进行生成和评估
        for video_idx, (video_path, dataset_name) in enumerate(tqdm(all_test_videos, desc=f"Processing videos for step {step}")):
            try:
                # 提取第一帧和最后一帧
                first_pil, last_pil = extract_frames_from_video(video_path, first_frame_idx=0, last_frame_idx=16)
                
                # 生成视频
                frames = pipe(
                    encoder_hidden_states=encoder_hidden_states,
                    first_image=first_pil,
                    last_image=last_pil,
                    height=480,
                    width=832,
                    num_frames=17,
                    num_inference_steps=50,
                    generator=torch.Generator(device="cuda").manual_seed(42),
                )
                
                # 保存生成的视频
                video_name = os.path.basename(video_path).replace('.mp4', '')
                output_video_path = step_output_dir / f"{dataset_name}_{video_name}_generated.mp4"
                export_to_video(frames, str(output_video_path), fps=16)
                
                # 计算整体视频评估指标
                lpips_score = metrics.calculate_lpips(video_path, str(output_video_path))
                psnr_score = metrics.calculate_psnr(video_path, str(output_video_path))
                ssim_score = metrics.calculate_ssim(video_path, str(output_video_path))
                
                # 计算首帧和尾帧的指标
                real_frames = metrics.extract_frames_from_video_cv2(video_path)
                gen_frames = metrics.extract_frames_from_video_cv2(str(output_video_path))
                
                first_frame_metrics = {"lpips": -1.0, "psnr": -1.0, "ssim": -1.0}
                last_frame_metrics = {"lpips": -1.0, "psnr": -1.0, "ssim": -1.0}
                
                if len(real_frames) > 0 and len(gen_frames) > 0:
                    # 首帧指标
                    first_frame_metrics["lpips"] = metrics.calculate_frame_lpips(real_frames[0], gen_frames[0])
                    first_frame_metrics["psnr"] = metrics.calculate_frame_psnr(real_frames[0], gen_frames[0])
                    first_frame_metrics["ssim"] = metrics.calculate_frame_ssim(real_frames[0], gen_frames[0])
                    
                    # 尾帧指标（取最后一帧）
                    if len(real_frames) > 1 and len(gen_frames) > 1:
                        last_real_idx = min(len(real_frames) - 1, 16)  # 对应extract_frames_from_video中的last_frame_idx=16
                        last_gen_idx = min(len(gen_frames) - 1, 16)
                        last_frame_metrics["lpips"] = metrics.calculate_frame_lpips(real_frames[last_real_idx], gen_frames[last_gen_idx])
                        last_frame_metrics["psnr"] = metrics.calculate_frame_psnr(real_frames[last_real_idx], gen_frames[last_gen_idx])
                        last_frame_metrics["ssim"] = metrics.calculate_frame_ssim(real_frames[last_real_idx], gen_frames[last_gen_idx])
                
                video_result = {
                    "video_idx": video_idx,
                    "original_video": video_path,
                    "generated_video": str(output_video_path),
                    "dataset": dataset_name,
                    "lpips": lpips_score,
                    "psnr": psnr_score,
                    "ssim": ssim_score,
                    "first_frame_metrics": first_frame_metrics,
                    "last_frame_metrics": last_frame_metrics
                }
                
                step_results["videos"].append(video_result)
                
                print(f"Video {video_idx+1}/{len(all_test_videos)}: "
                      f"LPIPS={lpips_score:.4f}, PSNR={psnr_score:.2f}, SSIM={ssim_score:.4f}")
                print(f"  首帧: LPIPS={first_frame_metrics['lpips']:.4f}, PSNR={first_frame_metrics['psnr']:.2f}, SSIM={first_frame_metrics['ssim']:.4f}")
                print(f"  尾帧: LPIPS={last_frame_metrics['lpips']:.4f}, PSNR={last_frame_metrics['psnr']:.2f}, SSIM={last_frame_metrics['ssim']:.4f}")
                
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                continue
        
        # 计算该步数的平均指标
        if step_results["videos"]:
            lpips_scores = [v["lpips"] for v in step_results["videos"] if v["lpips"] > 0]
            psnr_scores = [v["psnr"] for v in step_results["videos"] if v["psnr"] > 0]
            ssim_scores = [v["ssim"] for v in step_results["videos"] if v["ssim"] > 0]
            
            # 首帧指标
            first_lpips_scores = [v["first_frame_metrics"]["lpips"] for v in step_results["videos"] if v["first_frame_metrics"]["lpips"] > 0]
            first_psnr_scores = [v["first_frame_metrics"]["psnr"] for v in step_results["videos"] if v["first_frame_metrics"]["psnr"] > 0]
            first_ssim_scores = [v["first_frame_metrics"]["ssim"] for v in step_results["videos"] if v["first_frame_metrics"]["ssim"] > 0]
            
            # 尾帧指标
            last_lpips_scores = [v["last_frame_metrics"]["lpips"] for v in step_results["videos"] if v["last_frame_metrics"]["lpips"] > 0]
            last_psnr_scores = [v["last_frame_metrics"]["psnr"] for v in step_results["videos"] if v["last_frame_metrics"]["psnr"] > 0]
            last_ssim_scores = [v["last_frame_metrics"]["ssim"] for v in step_results["videos"] if v["last_frame_metrics"]["ssim"] > 0]
            
            step_results["summary"] = {
                "avg_lpips": np.mean(lpips_scores) if lpips_scores else -1,
                "std_lpips": np.std(lpips_scores) if lpips_scores else -1,
                "avg_psnr": np.mean(psnr_scores) if psnr_scores else -1,
                "std_psnr": np.std(psnr_scores) if psnr_scores else -1,
                "avg_ssim": np.mean(ssim_scores) if ssim_scores else -1,
                "std_ssim": np.std(ssim_scores) if ssim_scores else -1,
                "num_videos": len(step_results["videos"]),
                # 首帧指标
                "first_frame_avg_lpips": np.mean(first_lpips_scores) if first_lpips_scores else -1,
                "first_frame_std_lpips": np.std(first_lpips_scores) if first_lpips_scores else -1,
                "first_frame_avg_psnr": np.mean(first_psnr_scores) if first_psnr_scores else -1,
                "first_frame_std_psnr": np.std(first_psnr_scores) if first_psnr_scores else -1,
                "first_frame_avg_ssim": np.mean(first_ssim_scores) if first_ssim_scores else -1,
                "first_frame_std_ssim": np.std(first_ssim_scores) if first_ssim_scores else -1,
                # 尾帧指标
                "last_frame_avg_lpips": np.mean(last_lpips_scores) if last_lpips_scores else -1,
                "last_frame_std_lpips": np.std(last_lpips_scores) if last_lpips_scores else -1,
                "last_frame_avg_psnr": np.mean(last_psnr_scores) if last_psnr_scores else -1,
                "last_frame_std_psnr": np.std(last_psnr_scores) if last_psnr_scores else -1,
                "last_frame_avg_ssim": np.mean(last_ssim_scores) if last_ssim_scores else -1,
                "last_frame_std_ssim": np.std(last_ssim_scores) if last_ssim_scores else -1,
            }
            
            print(f"\nStep {step} Summary:")
            print(f"  整体视频: LPIPS={step_results['summary']['avg_lpips']:.4f}±{step_results['summary']['std_lpips']:.4f}, "
                  f"PSNR={step_results['summary']['avg_psnr']:.2f}±{step_results['summary']['std_psnr']:.2f}, "
                  f"SSIM={step_results['summary']['avg_ssim']:.4f}±{step_results['summary']['std_ssim']:.4f}")
            print(f"  首帧: LPIPS={step_results['summary']['first_frame_avg_lpips']:.4f}±{step_results['summary']['first_frame_std_lpips']:.4f}, "
                  f"PSNR={step_results['summary']['first_frame_avg_psnr']:.2f}±{step_results['summary']['first_frame_std_psnr']:.2f}, "
                  f"SSIM={step_results['summary']['first_frame_avg_ssim']:.4f}±{step_results['summary']['first_frame_std_ssim']:.4f}")
            print(f"  尾帧: LPIPS={step_results['summary']['last_frame_avg_lpips']:.4f}±{step_results['summary']['last_frame_std_lpips']:.4f}, "
                  f"PSNR={step_results['summary']['last_frame_avg_psnr']:.2f}±{step_results['summary']['last_frame_std_psnr']:.2f}, "
                  f"SSIM={step_results['summary']['last_frame_avg_ssim']:.4f}±{step_results['summary']['last_frame_std_ssim']:.4f}")
        
        all_results[step] = step_results
        
        # 保存该步数的结果
        with open(step_output_dir / "results.json", "w") as f:
            json.dump(step_results, f, indent=2)
        
        # 清理GPU内存
        del pipe
        torch.cuda.empty_cache()
    
    # 保存所有结果
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 打印最终总结
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print("\nMetrics Explanation:")
    print("- LPIPS (Learned Perceptual Image Patch Similarity): 越低越好 (0-1，0表示完全相同)")
    print("- PSNR (Peak Signal-to-Noise Ratio): 越高越好 (通常>30dB为好)")
    print("- SSIM (Structural Similarity Index): 越高越好 (0-1，1表示完全相同)")
    print()
    
    # 创建比较表格
    print("\n整体视频指标:")
    print(f"{'Step':<8} {'LPIPS':<12} {'PSNR':<12} {'SSIM':<12} {'Videos':<8}")
    print("-" * 60)
    
    for step in model_steps:
        if step in all_results and all_results[step]["videos"]:
            summary = all_results[step]["summary"]
            print(f"{step:<8} "
                  f"{summary['avg_lpips']:<12.4f} "
                  f"{summary['avg_psnr']:<12.2f} "
                  f"{summary['avg_ssim']:<12.4f} "
                  f"{summary['num_videos']:<8}")
    
    print("\n首帧指标:")
    print(f"{'Step':<8} {'LPIPS':<12} {'PSNR':<12} {'SSIM':<12}")
    print("-" * 50)
    
    for step in model_steps:
        if step in all_results and all_results[step]["videos"]:
            summary = all_results[step]["summary"]
            print(f"{step:<8} "
                  f"{summary['first_frame_avg_lpips']:<12.4f} "
                  f"{summary['first_frame_avg_psnr']:<12.2f} "
                  f"{summary['first_frame_avg_ssim']:<12.4f}")
    
    print("\n尾帧指标:")
    print(f"{'Step':<8} {'LPIPS':<12} {'PSNR':<12} {'SSIM':<12}")
    print("-" * 50)
    
    for step in model_steps:
        if step in all_results and all_results[step]["videos"]:
            summary = all_results[step]["summary"]
            print(f"{step:<8} "
                  f"{summary['last_frame_avg_lpips']:<12.4f} "
                  f"{summary['last_frame_avg_psnr']:<12.2f} "
                  f"{summary['last_frame_avg_ssim']:<12.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("Generated videos are stored in respective step directories.")

if __name__ == "__main__":
    main()









