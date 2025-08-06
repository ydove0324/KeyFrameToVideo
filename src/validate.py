'''
python src/validate.py \
  --validation_file path/to/validation.json \
  --transformer_path path/to/transformer/model \
  --output_dir validation_results \
  --num_gpus 8  # number of GPUs to use
'''
import os
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import decord
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from typing import List, Tuple, Dict, Union
import cv2

# Import necessary modules
from WanIBQKeyFrame2VideoPipeline import WanIBQKeyFrame2VideoPipeline
from finetrainers.models.wan.transformer_wan import WanTransformer3DModel
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from transformers import CLIPVisionModel, CLIPImageProcessor, UMT5EncoderModel, AutoTokenizer
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.export_utils import export_to_video
from src.model.ibq_tokenizer import IBQ
from omegaconf import OmegaConf

# Import video evaluation metrics
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
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")

class VideoMetrics:
    """Video quality evaluation metrics calculation class"""
    
    def __init__(self, device: Union[str, torch.device] = 'cuda'):
        self.device = device
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    def calculate_metrics(self, real_video_path: str, generated_video_path: str, height: int, width: int, key_frames_indices: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics between two videos for key frames only"""
        metrics = {}
        
        # Convert key_frames_indices to flat list
        key_frames_indices = key_frames_indices.flatten().cpu().numpy().tolist()
        
        # Extract frames
        real_frames = self._extract_frames(real_video_path, height, width)
        gen_frames = self._extract_frames(generated_video_path, height, width)
        
        if not real_frames or not gen_frames:
            return {
                'lpips': -1.0,
                'psnr': -1.0,
                'ssim': -1.0,
                'rfid': -1.0
            }
        
        # Extract only key frames
        real_key_frames = [real_frames[i] for i in key_frames_indices if i < len(real_frames)]
        gen_key_frames = [gen_frames[i] for i in key_frames_indices if i < len(gen_frames)]
        
        if not real_key_frames or not gen_key_frames:
            return {
                'lpips': -1.0,
                'psnr': -1.0,
                'ssim': -1.0,
                'rfid': -1.0
            }
        
        # Calculate metrics for key frames only
        metrics['lpips'] = self._calculate_lpips(real_key_frames, gen_key_frames)
        metrics['psnr'] = self._calculate_psnr(real_key_frames, gen_key_frames)
        metrics['ssim'] = self._calculate_ssim(real_key_frames, gen_key_frames)
        metrics['rfid'] = self._calculate_rfid(real_key_frames, gen_key_frames)
        
        return metrics
    
    def _extract_frames(self, video_path: str, height: int, width: int) -> List[np.ndarray]:
        """Extract frames from **video** or return a single frame for **image** inputs."""

        img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"]
        suffix = Path(video_path).suffix.lower()

        # Image input ➜ single frame list
        if suffix in img_exts:
            try:
                from PIL import Image
                img = Image.open(video_path)
                img = img.convert('RGB')
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                img = np.array(img)
                return [img]
            except Exception as e:
                print(f"Failed to load image {video_path}: {e}")
                return []

        # Video input ➜ iterate through frames
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            frames.append(frame)

        cap.release()
        return frames
    
    def _calculate_lpips(self, real_frames: List[np.ndarray], gen_frames: List[np.ndarray]) -> float:
        """Calculate LPIPS score between two sets of frames"""
        if not LPIPS_AVAILABLE:
            return -1.0
        
        min_frames = min(len(real_frames), len(gen_frames))
        lpips_scores = []
        
        for i in range(min_frames):
            real_tensor = self._frame_to_tensor(real_frames[i])
            gen_tensor = self._frame_to_tensor(gen_frames[i])
            
            with torch.no_grad():
                score = self.lpips_fn(real_tensor, gen_tensor)
                lpips_scores.append(score.item())
        
        return float(np.mean(lpips_scores))
    
    def _calculate_psnr(self, real_frames: List[np.ndarray], gen_frames: List[np.ndarray]) -> float:
        """Calculate PSNR between two sets of frames"""
        min_frames = min(len(real_frames), len(gen_frames))
        psnr_scores = []
        
        for i in range(min_frames):
            mse = np.mean((real_frames[i].astype(np.float32) - gen_frames[i].astype(np.float32)) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            psnr_scores.append(psnr)
        
        return float(np.mean(psnr_scores))
    
    def _calculate_ssim(self, real_frames: List[np.ndarray], gen_frames: List[np.ndarray]) -> float:
        """Calculate SSIM between two sets of frames"""
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            print("Warning: scikit-image not available. Install with: pip install scikit-image")
            return -1.0
        
        min_frames = min(len(real_frames), len(gen_frames))
        ssim_scores = []
        
        for i in range(min_frames):
            real_gray = cv2.cvtColor(real_frames[i], cv2.COLOR_RGB2GRAY)
            gen_gray = cv2.cvtColor(gen_frames[i], cv2.COLOR_RGB2GRAY)
            score = ssim(real_gray, gen_gray, data_range=255)
            ssim_scores.append(score)
        
        return float(np.mean(ssim_scores))
    
    def _calculate_rfid(self, real_frames: List[np.ndarray], gen_frames: List[np.ndarray]) -> float:
        """Calculate FID score between real and generated key frames"""
        if not FID_AVAILABLE:
            return -1.0
            
        # Create temporary directories with absolute paths in /tmp
        import tempfile
        temp_base = tempfile.mkdtemp(prefix="rfid_calc_")
        temp_real_dir = Path(temp_base) / "real"
        temp_gen_dir = Path(temp_base) / "gen"
        temp_real_dir.mkdir(parents=True, exist_ok=True)
        temp_gen_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save frames as images with error checking
            for i, (real_frame, gen_frame) in enumerate(zip(real_frames, gen_frames)):
                try:
                    # Ensure frames are valid numpy arrays with correct dtype
                    real_frame = np.asarray(real_frame, dtype=np.uint8)
                    gen_frame = np.asarray(gen_frame, dtype=np.uint8)
                    
                    # Save images
                    real_path = temp_real_dir / f"frame_{i:04d}.png"
                    gen_path = temp_gen_dir / f"frame_{i:04d}.png"
                    
                    Image.fromarray(real_frame).save(str(real_path))
                    Image.fromarray(gen_frame).save(str(gen_path))
                    
                    # Verify files were written correctly
                    if not real_path.exists() or real_path.stat().st_size == 0:
                        print(f"Warning: Failed to write real frame {i}")
                        return -1.0
                    if not gen_path.exists() or gen_path.stat().st_size == 0:
                        print(f"Warning: Failed to write generated frame {i}")
                        return -1.0
                        
                except Exception as e:
                    print(f"Error saving frame {i}: {str(e)}")
                    return -1.0
            
            # Calculate FID score
            try:
                fid = fid_score.calculate_fid_given_paths(
                    [str(temp_real_dir), str(temp_gen_dir)],
                    batch_size=1,
                    device=self.device,
                    dims=2048
                )
                return float(fid)
            except Exception as e:
                print(f"Error calculating FID score: {str(e)}")
                return -1.0
        
        finally:
            # Cleanup temporary directories
            import shutil
            shutil.rmtree(temp_base, ignore_errors=True)
    
    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert frame to tensor normalized to [-1,1]"""
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensor = (tensor * 2.0 - 1.0).unsqueeze(0)
        return tensor.to(self.device)


    


def load_pipeline(model_id: str, transformer_path: str, device: Union[str, torch.device] = "cuda") -> WanIBQKeyFrame2VideoPipeline:
    """Load model pipeline"""
    print(f"Loading model from {transformer_path}...")
    
    # Load components
    transformer = WanTransformer3DModel.from_pretrained(
        transformer_path, 
        subfolder="transformer", 
        torch_dtype=torch.bfloat16
    )
    
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.bfloat16)
    image_processor = CLIPImageProcessor.from_pretrained(model_id, subfolder="image_processor")
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    
    # Load IBQ model
    tokenize_path = "/share/project/zhangfan/weights/Emu3.5-Tokenizer/IBQ-XL-f16c131k-FI"
    config = OmegaConf.load(os.path.join(tokenize_path, "fusimage_ibqgan_xl_131072_siglip.yaml"))
    ibq_model = IBQ(**config.model.init_args).to(dtype=torch.bfloat16)
    ckpt = torch.load(os.path.join(tokenize_path, "fusionimage_256_XL_f16c131k.ckpt"), weights_only=True)
    ibq_model.load_state_dict(ckpt["state_dict"])
    ibq_model.to(device)
    
    # Create pipeline
    pipe = WanIBQKeyFrame2VideoPipeline(
        transformer=transformer,
        vae=vae,
        image_encoder=image_encoder,
        image_processor=image_processor,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        ibq_model=ibq_model
    ).to(device)
    
    return pipe

def extract_keyframes(video_path: str, key_frames_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract key frame(s) from either a **video** or a **single image**.

    If `video_path` points to an image file, one frame is loaded and returned.
    """

    img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"]
    suffix = Path(video_path).suffix.lower()

    # -------------------------------------------------------------
    # Case 1: Input is an **image** ➜ single frame (F = 1)
    # -------------------------------------------------------------
    if suffix in img_exts:
        from PIL import Image
        import numpy as np

        img = Image.open(video_path).convert("RGB")
        frame = torch.from_numpy(np.array(img)).to("cuda")  # [H,W,3]
        frame = frame.permute(2, 0, 1)  # [3,H,W]

        key_frames = frame.unsqueeze(0).unsqueeze(0)  # [B=1,F=1,3,H,W]
        key_frames_indices_tensor = torch.tensor([[0]], device="cuda")
        return key_frames, key_frames_indices_tensor

    # -------------------------------------------------------------
    # Case 2: Input is a **video** ➜ gather requested key frames
    # -------------------------------------------------------------
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    key_frames = vr.get_batch(key_frames_indices).to("cuda")  # [F,H,W,3]
    key_frames = key_frames.permute(0, 3, 1, 2)  # [F,3,H,W]
    key_frames = key_frames.unsqueeze(0)  # [B,F,3,H,W] (B = 1)

    key_frames_indices_tensor = torch.tensor(key_frames_indices, device="cuda")
    return key_frames, key_frames_indices_tensor

def identity_collate(x):
    return x

def run_validation(rank, world_size, args):
    """Run validation on a single GPU"""
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    
    # Initialize metrics calculator
    metrics_calculator = VideoMetrics(device=device)
    
    # Load validation dataset
    dataset = ValidationDataset(args.validation_file)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=identity_collate
    )
    
    # Load pipeline
    pipe = load_pipeline(args.model_id, args.transformer_path, device)
    
    # Load encoder hidden states
    encoder_hidden_states = torch.load("debug_tensors/encoder_hidden_states_t1000.pt").to(device)
    
    # Create output directory with rank
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track metrics
    all_metrics = []
    
    # Run validation
    for batch in tqdm(dataloader, desc=f"Validating (GPU {rank})", disable=rank != 0):
        item = batch[0]  # Since batch_size=1, we take the first (and only) item

        # ------------------------------------------------------------------
        # Support either `image_path` or `video_path` (exactly one must exist)
        # ------------------------------------------------------------------
        has_video = "video_path" in item and item["video_path"] is not None
        has_image = "image_path" in item and item["image_path"] is not None
        assert has_video ^ has_image, "Exactly one of 'video_path' or 'image_path' must be provided."

        media_path = item["video_path"] if has_video else item["image_path"]

        # Default parameters for image inputs
        if has_image:
            num_frames = 1
            key_frames_indices = item.get("key_frames_indices", [[0]])
        else:
            num_frames = item["num_frames"]
            key_frames_indices = item["key_frames_indices"]

        if rank == 0:
            print(f"Processing {media_path} on GPU {rank}")
        
        # Extract keyframes
        key_frames, key_frames_indices = extract_keyframes(media_path, key_frames_indices)
        if rank == 0:
            print(f"Key frames: {key_frames.shape}")
            print(f"Key frames indices: {key_frames_indices.shape}")
        
        # Generate video
        generated_frames = pipe(
            encoder_hidden_states=encoder_hidden_states,
            key_frames=key_frames,
            key_frames_indices=key_frames_indices,
            height=item["height"],
            width=item["width"],
            num_frames=num_frames,
            num_inference_steps=50,
            guidance_scale=0,
            generator=torch.Generator(device).manual_seed(42)
        )
        if rank == 0:
            print(f"Generated frames Successfully")
        
        # Save generated video
        output_path = output_dir / f"{Path(media_path).stem}_generated.mp4"
        export_to_video(generated_frames, str(output_path), fps=16)
        
        # Calculate metrics
        metrics = metrics_calculator.calculate_metrics(
            media_path, 
            str(output_path),
            height=item["height"],
            width=item["width"],
            key_frames_indices=key_frames_indices
        )
        if rank == 0:
            print(f"Metrics Calculated Successfully")
        
        # Attach sample index so we can identify low-PSNR items later
        metrics["idx"] = item["idx"]
        metrics["media_path"] = media_path  # Save original media path for later reference
        all_metrics.append(metrics)
        
        if rank == 0:
            print(f"Metrics: LPIPS={metrics['lpips']:.4f}, PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}, RFID={metrics['rfid']:.4f}")
    
    # Gather metrics from all processes
    world_metrics = [None] * world_size
    dist.all_gather_object(world_metrics, all_metrics)
    
    if rank == 0:
        # Combine metrics from all processes
        combined_metrics = []
        for metrics_list in world_metrics:
            combined_metrics.extend(metrics_list)
        
        # Calculate and save average metrics
        avg_metrics = {
            metric: float(np.mean([m[metric] for m in combined_metrics]))
            for metric in ['lpips', 'psnr', 'ssim', 'rfid']
        }
        
        std_metrics = {
            metric: float(np.std([m[metric] for m in combined_metrics]))
            for metric in ['lpips', 'psnr', 'ssim', 'rfid']
        }

        # ------------------------------------------------------------------
        # Identify lowest 10 % PSNR samples
        # ------------------------------------------------------------------
        psnr_values = [m["psnr"] for m in combined_metrics]
        if len(psnr_values) > 0:
            psnr_threshold = float(np.percentile(psnr_values, 10))  # bottom 10 percentile
            lowest_psnr_metrics = [m for m in combined_metrics if m["psnr"] <= psnr_threshold]
            lowest_psnr_paths = [m.get("media_path", "") for m in lowest_psnr_metrics]

            print("\nLowest 10% PSNR threshold: {:.2f}".format(psnr_threshold))
            print("Paths with lowest 10% PSNR:", lowest_psnr_paths)

            # Add to results for further inspection
            results_extra = {
                "psnr_threshold_10pct": psnr_threshold,
                "lowest_10pct_psnr_paths": lowest_psnr_paths,
                "lowest_10pct_psnr_metrics": lowest_psnr_metrics,
            }
        else:
            results_extra = {
                "psnr_threshold_10pct": None,
                "lowest_10pct_psnr_paths": [],
                "lowest_10pct_psnr_metrics": [],
            }

        # Merge extra results
        results = {
            'average_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'all_metrics': combined_metrics,
            **results_extra
        }
        
        # Save final results
        with open(Path(args.output_dir) / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\nFinal Results:")
        print(f"LPIPS: {avg_metrics['lpips']:.4f} ± {std_metrics['lpips']:.4f}")
        print(f"PSNR: {avg_metrics['psnr']:.2f} ± {std_metrics['psnr']:.2f}")
        print(f"SSIM: {avg_metrics['ssim']:.4f} ± {std_metrics['ssim']:.4f}")
        print(f"RFID: {avg_metrics['rfid']:.4f} ± {std_metrics['rfid']:.4f}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Validate video generation models")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to validation dataset file")
    parser.add_argument("--model_id", type=str, default="/share/project/huangxu/model/Wan2.1-T2V-1.3B-diffusers")
    parser.add_argument("--transformer_path", type=str, required=True, help="Path to transformer model")
    parser.add_argument("--output_dir", type=str, default="validation_results")
    args = parser.parse_args()
    
    # Get world size and rank from environment variables (set by torchrun)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"World size: {world_size}, Rank: {rank}")
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    print(f"Rank {rank} initialized")
    
    try:
        # Run validation
        run_validation(rank, world_size, args)
    finally:
        # Clean up distributed environment
        dist.destroy_process_group()

class ValidationDataset(Dataset):
    def __init__(self, validation_file: str):
        """Initialize the validation dataset"""
        super().__init__()
        with open(validation_file, 'r') as f:
            self._data = json.load(f)["data"]
    
    def __len__(self) -> int:
        """Return the number of items in the dataset"""
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single item from the dataset"""
        item = self._data[idx]
        item["idx"] = idx  # Add index to the item
        return item

if __name__ == "__main__":
    print(f"Starting validation")
    main()











