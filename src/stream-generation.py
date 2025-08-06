from __future__ import annotations

"""Stream Video Generation Pipeline
=================================
A streaming video generation pipeline that uses sliding windows to generate
video segments continuously. It starts with key frames [0,8,16], then moves
to [8,16,24], [16,24,32], and so on until reaching the target frame.
"""

import torch
import decord
from PIL import Image
import numpy as np
from typing import List, Optional, Tuple
import os
from tqdm import tqdm
from diffusers.utils import export_to_video
import sys
sys.path.append("/share/project/huangxu/workspace/KeyFrameToVideo")
from src.WanIBQKeyFrame2VideoPipeline import WanIBQKeyFrame2VideoPipeline
import numpy as np


class StreamVideoGenerator:
    """Stream video generator using sliding window approach."""
    
    def __init__(self, pipeline: WanIBQKeyFrame2VideoPipeline):
        self.pipeline = pipeline
        self.generated_frames = []
        self.current_frame_count = 0
        
    def _load_video_frames(self, video_path: str, frame_indices: List[int]) -> torch.Tensor:
        """Load specific frames from video."""
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path)
        print(f"Total number of frames in video: {len(vr)}")
        frames = vr.get_batch(frame_indices).to(self.pipeline.device)
        # print(frames.shape)
        frames = frames.permute(0, 3, 1, 2)  # [F,3,H,W]
        return frames
    
    def _generate_segment(
        self,
        key_frames_indices: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        key_frames: Optional[torch.Tensor] = None,
        first_frame: Optional[torch.Tensor] = None,
        ibq_indices: Optional[torch.Tensor] = None,
        num_frames: int = 17,
        num_inference_steps: int = 50,
        guidance_scale: float = 0.0,
        height: int = 480,
        width: int = 832,
        seed: Optional[int] = None,
        use_first_frame: bool = False,
        return_tensors: bool = False,
        **kwargs,
    ):
        """Generate a video segment using the pipeline."""
        # print(f"key_frames_indices: {key_frames_indices}")
        # print(f"key_frames: {key_frames.shape}")
        return self.pipeline(
            key_frames=key_frames,
            first_frame=first_frame,
            ibq_indices=ibq_indices,
            key_frames_indices=key_frames_indices,
            encoder_hidden_states=encoder_hidden_states,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            seed=seed,
            use_first_frame=use_first_frame,
            return_tensors=return_tensors,
            **kwargs,
        )
    
    def generate_stream(
        self,
        video_path: Optional[str] = None,
        ibq_indices_path: Optional[str] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        start_frame: int = 0,
        end_frame: int = 160,
        window_size: int = 2,
        overlap_window_size: int = 0,
        frame_interval: int = 8,
        num_frames_per_segment: int = 17,
        num_inference_steps: int = 50,
        guidance_scale: float = 0.0,
        height: int = 480,
        width: int = 832,
        seed: Optional[int] = None,
        save_intermediate: bool = True,
        output_dir: str = "stream_output",
        using_i2v: Optional[bool] = True,
    ) -> List[Image.Image]:
        """
        Generate video stream using sliding window approach.
        
        Args:
            video_path: Path to source video
            encoder_hidden_states: Pre-encoded text embeddings
            start_frame: Starting frame index
            end_frame: Target end frame index
            window_size: Number of key frames per window (default: 3)
            frame_interval: Interval between key frames (default: 8)
            num_frames_per_segment: Number of frames to generate per segment
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            height: Video height
            width: Video width
            seed: Random seed for generation
            save_intermediate: Whether to save intermediate segments
            output_dir: Directory to save intermediate segments
            
        Returns:
            List of all generated frames
        """
        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)
        
        all_frames = []
        current_seed = seed
        if ibq_indices_path is not None:
            ibq_indices = torch.from_numpy(np.load(ibq_indices_path)).to(self.pipeline.device)
            print(f"ibq_indices.shape: {ibq_indices.shape}")
            total_frames_max = ibq_indices.shape[0] * frame_interval + 1
            end_frame = min(end_frame,start_frame + total_frames_max - 1)
        else:
            ibq_indices = None
        
        # Calculate number of segments needed with overlap
        # Each segment advances by frame_interval frames
        total_frames_needed = end_frame - start_frame + 1
        assert (total_frames_needed - 1) % frame_interval == 0, "total_frames_needed must be a multiple of frame_interval"
        total_window_size = (total_frames_needed - 1) // frame_interval
        # window_size + (num_segments - 1) * (window_size - overlap_window_size) = total_window_size
        num_segments = (total_window_size - window_size) // (window_size - overlap_window_size) + 1
        if num_segments < 1:
            num_segments = 1
        cached_key_frames = None
        cached_ibq_indices = None

        print(f"Generating {num_segments} segments from frame {start_frame} to {end_frame}")
        
        for segment_idx in tqdm(range(num_segments), desc="Generating segments"):
            # Calculate key frame indices for this segment with overlap
            # For segment 0: [0, 8, 16]
            # For segment 1: [8, 16, 24] 
            # For segment 2: [16, 24, 32]
            # etc.
            segment_start = start_frame + segment_idx * frame_interval * (window_size - overlap_window_size)
            key_frame_indices = [segment_start + i * frame_interval for i in range(window_size + 1)]
            
            # Ensure we don't exceed end_frame
            key_frame_indices = [idx for idx in key_frame_indices if idx <= end_frame]
            if len(key_frame_indices) < window_size + 1:
                # Pad with the last frame if needed
                while len(key_frame_indices) < window_size + 1:
                    key_frame_indices.append(key_frame_indices[-1])
            
            print(f"Segment {segment_idx + 1}/{num_segments}: Key frames {key_frame_indices}")
            print(f"  Relative indices: {[idx - segment_start for idx in key_frame_indices]}")
            
            # Load key frames from video
            if video_path is not None:
                key_frames = self._load_video_frames(video_path, key_frame_indices) # [F,3,H,W]
                key_frames = (key_frames.float() / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1] if input is [0, 255]
                key_frames = torch.nn.functional.interpolate(key_frames, size=(height, width), mode='bilinear', align_corners=False)
                
                key_frames = key_frames.unsqueeze(0)  # Add batch dimension [B,F,3,H,W]
                if using_i2v and cached_key_frames is not None:
                    key_frames[:,0] = cached_key_frames
            else:
                key_frames = None
            if ibq_indices is not None:
                ibq_indices_slices = ibq_indices[segment_idx*window_size:segment_idx*window_size + window_size + 1] # [F',H,W]
                if using_i2v and cached_ibq_indices is not None:
                    ibq_indices_slices[0] = cached_ibq_indices
                    
            else:
                ibq_indices_slices = None
            
            # Create key frame indices tensor (relative to segment start)
            # For segment 0: [0, 8, 16] -> [0, 8, 16]
            # For segment 1: [8, 16, 24] -> [0, 8, 16] (relative to segment start)
            relative_indices = [idx - segment_start for idx in key_frame_indices]
            key_frames_indices = torch.tensor(relative_indices, device=self.pipeline.device).unsqueeze(0)
            
            # Generate segment
            segment_frames,segment_frames_tensors = self._generate_segment(
                # key_frames=key_frames,
                first_frame=cached_key_frames,
                key_frames_indices=key_frames_indices,
                ibq_indices=ibq_indices_slices,
                encoder_hidden_states=encoder_hidden_states,
                num_frames=num_frames_per_segment,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                seed=current_seed,
                use_first_frame=using_i2v and (segment_idx != 0),
                return_tensors=True,
                normalize_key_frames=False
            )
            if using_i2v:
                # segment_frames_tensors: [B,3,F,H,W]
                cached_key_frames = segment_frames_tensors[:,:,(window_size - overlap_window_size) * frame_interval]  # [B,3,H,W]
                assert cached_key_frames.shape == (1, 3, height, width), "cached_key_frames shape is not correct"
                if ibq_indices is not None:
                    _, _ , cached_ibq_indices = self.pipeline._ibq_encode(cached_key_frames)
                    cached_ibq_indices = cached_ibq_indices.view((height//16, width // 16))
            # Save intermediate segment if requested
            if save_intermediate:
                segment_path = os.path.join(output_dir, f"segment_{segment_idx:03d}.mp4")
                export_to_video(segment_frames, segment_path, fps=16)
                print(f"Saved segment to {segment_path}")
            
            # Add frames to overall result (avoiding duplicates)
            if segment_idx == 0:
                # For first segment, add all frames
                all_frames.extend(segment_frames)
                print(f"  Added {len(segment_frames)} frames from first segment")
            else:
                # For subsequent segments, add frames starting from the overlap point
                # Since we have overlap of (window_size-1) frames, we skip the first (window_size-1) frames
                overlap_frames = overlap_window_size * frame_interval  # Number of overlapping frames
                new_frames = segment_frames[overlap_frames+1:]
                all_frames.extend(new_frames)
                print(f"  Added {len(new_frames)} new frames (skipped {overlap_frames} overlapping frames)")
            
            # Update seed for next segment (optional: use same seed for consistency)
            if current_seed is not None:
                current_seed += 1
        
        print(f"Generated {len(all_frames)} total frames")
        return all_frames


def main():
    """Main function to demonstrate stream generation."""
    # Load your pipeline components (same as in WanIBQKeyFrame2VideoPipeline.py)
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel
    from transformers import CLIPVisionModel, CLIPImageProcessor, UMT5EncoderModel, AutoTokenizer
    from omegaconf import OmegaConf
    from src.model.ibq_tokenizer import IBQ
    
    # Model paths
    transformer_path = "/share/project/huangxu/model/wan-ibq-key-frame-video-pretrain-first-frame-condition/model_weights/022250"
    model_id = "/share/project/huangxu/model/Wan2.1-T2V-1.3B-diffusers"
    tokenize_path = "/share/project/zhangfan/weights/Emu3.5-Tokenizer/IBQ-XL-f16c131k-FI"
    
    # Load models
    transformer = WanTransformer3DModel.from_pretrained(
        transformer_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.bfloat16
    )
    image_processor = CLIPImageProcessor.from_pretrained(model_id, subfolder="image_processor")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    
    # Load IBQ model
    config_name = "fusimage_ibqgan_xl_131072_siglip.yaml"
    config = OmegaConf.load(os.path.join(tokenize_path, config_name))
    ibq_model = IBQ(**config.model.init_args).to(dtype=torch.bfloat16)
    ckpt_name = "fusionimage_256_XL_f16c131k.ckpt"
    ckpt = torch.load(os.path.join(tokenize_path, ckpt_name), weights_only=True)
    ibq_model.load_state_dict(ckpt["state_dict"])
    
    # Initialize scheduler and pipeline
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5)
    pipeline = WanIBQKeyFrame2VideoPipeline(
        text_encoder=text_encoder,
        transformer=transformer,
        ibq_model=ibq_model,
        vae=vae,
        scheduler=scheduler,
        tokenizer=tokenizer,
        image_encoder=image_encoder,
        image_processor=image_processor,
    ).to("cuda")
    
    # Load encoder hidden states
    encoder_hidden_states = torch.load("debug_tensors/encoder_hidden_states_t1000.pt").to("cuda").to(torch.bfloat16)
    
    # Initialize stream generator
    stream_generator = StreamVideoGenerator(pipeline)
    
    # Video path
    # video_path = "/share/project/lzx/video_data/sora_test_videos/tokyo-in-the-snow.mp4"
    ibq_file_path = "/share/project/zhangfan/codes/Emu3.5/emu3p5_stage2_55k_topk1024_temp1.0_p1.0_cfg5.0_v4/output_token/rank3_of_world_size4_8.npy"
    # video_path = "/share/project/lzx/video_data/sora_test_videos/art-museum.mp4"
    # video_path = "/share/project/lzx/video_data/sora_test_videos/chinese-new-year-dragon.mp4"
    # video_path = "/share/project/lzx/video_data/sora_test_videos/cat-on-bed.mp4"
    # video_path = "_tmp/2495abec3c75f0d9af80056633a1eb268a48f7ca.mp4"
    # Generate stream
    all_frames = stream_generator.generate_stream(
        ibq_indices_path=ibq_file_path,
        encoder_hidden_states=encoder_hidden_states,
        start_frame=0,
        end_frame=48,
        window_size=2,
        overlap_window_size=0,
        frame_interval=8,
        num_frames_per_segment=17,
        num_inference_steps=50,
        guidance_scale=0.0,
        height=66*8,
        width=120*8,
        seed=42,
        save_intermediate=True,
        output_dir="stream_output",
    )
    
    # Save final result
    export_to_video(all_frames, "slide-window-output/emu3.5-test8-22250-train-first-frame-condition.mp4", fps=16)
    print("Stream generation completed!")


if __name__ == "__main__":
    main()
