#!/usr/bin/env python3
"""
Test VideoVAE reconstruction functionality
Reads video (first 16 frames) or image and performs direct reconstruction
"""

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from PIL import Image
import numpy as np
import decord
from pathlib import Path
import cv2
import argparse
from typing import Union, List
import os

def load_video_vae(model_id: str, device: str = "cuda") -> AutoencoderKLWan:
    """Load VideoVAE model"""
    print(f"Loading VideoVAE from {model_id}...")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, 
        subfolder="vae", 
        torch_dtype=torch.bfloat16
    ).to(device)
    return vae

def extract_frames_from_video(video_path: str, num_frames: int = 16, height: int = 480, width: int = 832) -> torch.Tensor:
    """Extract first N frames from video and resize"""
    print(f"Extracting first {num_frames} frames from {video_path}")
    
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    
    # Get first N frames
    frame_indices = list(range(min(num_frames, len(vr))))
    frames = vr.get_batch(frame_indices).to("cuda")  # [F, H, W, 3]
    
    # Resize frames
    frames = F.interpolate(
        frames.permute(0, 3, 1, 2),  # [F, 3, H, W]
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )
    
    # Normalize to [-1, 1]
    frames = (frames.float() / 255.0 - 0.5) / 0.5
    
    return frames  # [F, 3, H, W]

def load_image_as_frames(image_path: str, num_frames: int = 16, height: int = 480, width: int = 832) -> torch.Tensor:
    """Load single image and repeat as frames"""
    print(f"Loading image {image_path} and repeating as {num_frames} frames")
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image = torch.from_numpy(np.array(image)).to("cuda")  # [H, W, 3]
    
    # Convert to tensor format and normalize
    image = image.permute(2, 0, 1).float()  # [3, H, W]
    image = (image / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Repeat for num_frames
    frames = image.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # [F, 3, H, W]
    
    return frames

def reconstruct_video(vae: AutoencoderKLWan, frames: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Reconstruct video frames using VideoVAE"""
    print(f"Reconstructing {frames.shape[0]} frames...")
    
    # Add batch dimension and permute to [B, C, F, H, W] format
    frames = frames.unsqueeze(0)  # [1, F, 3, H, W]
    frames = frames.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W]
    frames = frames.to(device=device, dtype=torch.bfloat16)
    
    # Encode to latents
    with torch.no_grad():
        latents = vae.encode(frames).latent_dist.sample()
        print(f"Encoded latents shape: {latents.shape}")
        
        # Apply normalization like in the pipeline
        # latents_mean = torch.tensor(vae.config.latents_mean).to(device=device)
        # latents_std = torch.tensor(vae.config.latents_std).to(device=device)
        
        # # Denormalize latents before decoding
        # latents_mean = latents_mean.view(1, -1, 1, 1, 1)
        # latents_std = latents_std.view(1, -1, 1, 1, 1)
        # latents_denorm = ((latents.float() / latents_std) + latents_mean).to(latents.dtype)
        
        # Decode back to video
        reconstructed = vae.decode(latents).sample
        print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Convert back to [F, C, H, W] format
    reconstructed = reconstructed.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    return reconstructed.squeeze(0)  # [F, 3, H, W]

def _postprocess(video: torch.Tensor) -> List[Image.Image]:
    """Convert decoded tensor [B,C,F,H,W] in [-1,1] to PIL list."""
    video = video.squeeze(0).permute(1, 0, 2, 3)  # F × C × H × W
    video = (video / 2 + 0.5).clamp(0, 1)
    video = (video * 255).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(f.transpose(1, 2, 0), "RGB") for f in video]

def save_video_frames(frames: torch.Tensor, output_path: str, fps: int = 16):
    """Save video frames to MP4 file"""
    from diffusers.utils import export_to_video
    
    # Convert from [-1, 1] to [0, 255] uint8
    frames = (frames / 2 + 0.5).clamp(0, 1)  # [0, 1]
    frames = (frames * 255).to(torch.uint8)  # [0, 255]
    
    # Convert to PIL Images
    frames_pil = []
    for i in range(frames.shape[0]):
        frame = frames[i].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        frame_pil = Image.fromarray(frame, "RGB")
        frames_pil.append(frame_pil)
    
    # Save as video
    export_to_video(frames_pil, output_path, fps=fps)
    print(f"Saved reconstructed video to {output_path}")

def calculate_reconstruction_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Calculate reconstruction quality metrics"""
    # Convert to float for calculations
    original = original.float()
    reconstructed = reconstructed.float()
    
    # MSE
    mse = F.mse_loss(original, reconstructed)
    
    # PSNR
    mse_val = mse.item()
    if mse_val == 0:
        psnr = float('inf')
    else:
        psnr = 20 * torch.log10(torch.tensor(2.0) / torch.sqrt(mse))  # 2.0 because range is [-1, 1]
    
    # MAE
    mae = F.l1_loss(original, reconstructed)
    
    return {
        'mse': mse.item(),
        'psnr': psnr.item(),
        'mae': mae.item()
    }

def main():
    parser = argparse.ArgumentParser(description="Test VideoVAE reconstruction")
    parser.add_argument("--input_path", type=str, required=True, 
                       help="Path to video file or image file")
    parser.add_argument("--model_id", type=str, 
                       default="/share/project/huangxu/model/Wan2.1-T2V-1.3B-diffusers",
                       help="Path to model directory")
    parser.add_argument("--output_path", type=str, default="reconstructed_video.mp4",
                       help="Output video path")
    parser.add_argument("--num_frames", type=int, default=1,
                       help="Number of frames to process")
    parser.add_argument("--height", type=int, default=480,
                       help="Video height")
    parser.add_argument("--width", type=int, default=832,
                       help="Video width")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} does not exist")
        return
    
    # Load VideoVAE
    vae = load_video_vae(args.model_id, args.device)
    
    # Determine input type and load frames
    input_path = Path(args.input_path)
    img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    
    if input_path.suffix.lower() in img_exts:
        # Image input
        frames = load_image_as_frames(
            args.input_path, 
            args.num_frames, 
            args.height, 
            args.width
        )
    else:
        # Video input
        frames = extract_frames_from_video(
            args.input_path, 
            args.num_frames, 
            args.height, 
            args.width
        )
    
    print(f"Input frames shape: {frames.shape}")
    
    # Reconstruct video
    reconstructed = reconstruct_video(vae, frames, args.device)
    
    # Calculate metrics
    metrics = calculate_reconstruction_metrics(frames, reconstructed)
    print(f"\nReconstruction Metrics:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"MAE: {metrics['mae']:.6f}")
    
    # Save reconstructed video
    save_video_frames(reconstructed, args.output_path)
    
    print(f"\nTest completed successfully!")
    print(f"Original frames: {frames.shape}")
    print(f"Reconstructed frames: {reconstructed.shape}")

if __name__ == "__main__":
    main() 