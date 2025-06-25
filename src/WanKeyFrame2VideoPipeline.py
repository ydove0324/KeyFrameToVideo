from __future__ import annotations

"""Wan Key‑Frame‑to‑Video Pipeline
=================================
A minimal diffusion pipeline that reproduces the current *manual_inference* logic
for Wan T2V **first–last‑frame conditioned** generation, but without the text
prompt.  Instead, the *already‑encoded* text hidden states are provided
explicitly.

Only the functionality required by the user's existing notebook is
implemented.  It is **not** meant to be a fully‑featured Diffusers pipeline –
just enough to slot into your current workflow.
"""

from typing import List, Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
import numpy as np
import decord
from tqdm import tqdm

__all__ = ["WanKeyFrame2VideoPipeline"]


def _normalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    """Apply the same *affine* normalisation used during training."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    return ((latents.float() - latents_mean) * latents_std).to(latents)


def _denormalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    """Undo the affine transform."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    return ((latents.float() / latents_std) + latents_mean).to(latents)


class WanKeyFrame2VideoPipeline(DiffusionPipeline):
    """Key-frame (first + last) conditioned video generation without prompt.

    The pipeline accepts *pre-encoded* text embeddings (`encoder_hidden_states`) –
    you can obtain them however you like plus the first and last frames as
    PIL images.  Internally the images are encoded twice:

    1. To *CLIP embeddings* (for the cross-attention in the transformer).
    2. Into VAE *latents* (so that the denoising process is anchored at those
       key-frames).

    Only the parts strictly required by the user's notebook are implemented –
    no classifier-free guidance, safety-checker, etc.
    """

    def __init__(
        self,
        *,
        transformer,  # WanTransformer3DModel
        vae,  # AutoencoderKLWan
        image_encoder,  # CLIPVisionModel
        image_processor,  # CLIPImageProcessor
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.scheduler = scheduler

        # Register for save / from_pretrained – optional but handy
        self.register_modules(
            transformer=transformer,
            vae=vae,
            image_encoder=image_encoder,
            image_processor=image_processor,
            scheduler=scheduler,
        )

        # Pre‑compute mean / std constants for (de)normalisation
        self._latents_mean = torch.tensor(vae.config.latents_mean)
        self._latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std)

    # ---------------------------------------------------------------------
    # Utility helpers – kept private to avoid polluting the public API
    # ---------------------------------------------------------------------
    def _encode_first_last_frame(self, first_img: Image.Image, last_img: Image.Image) -> torch.Tensor:
        """Return concatenated CLIP embeddings [B, 257*2, hidden]."""
        first = self.image_processor(images=first_img, convert_to_rgb=False, do_rescale=False, return_tensors="pt").pixel_values
        last = self.image_processor(images=last_img, convert_to_rgb=False, do_rescale=False, return_tensors="pt").pixel_values
        first = first.to(dtype=self.image_encoder.dtype, device=self.device)
        last = last.to(dtype=self.image_encoder.dtype, device=self.device)
        with torch.no_grad():
            first_emb = self.image_encoder(first, output_hidden_states=True).hidden_states[-2]
            last_emb = self.image_encoder(last, output_hidden_states=True).hidden_states[-2]
        return torch.cat([first_emb, last_emb], dim=1)  # B × 514 × D

    def _postprocess(self, video: torch.Tensor) -> List[Image.Image]:
        """Convert decoded tensor [B,C,F,H,W] in [-1,1] to PIL list."""
        video = video.squeeze(0).permute(1, 0, 2, 3)  # F × C × H × W
        video = (video / 2 + 0.5).clamp(0, 1)
        video = (video * 255).to(torch.uint8).cpu().numpy()
        return [Image.fromarray(f.transpose(1, 2, 0), "RGB") for f in video]

    # ---------------------------------------------------------------------
    # Public call – the heart of the pipeline
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        *,
        encoder_hidden_states: torch.Tensor,  # pre‑encoded text embeddings
        first_image: Image.Image,
        last_image: Image.Image,
        height: int = 480,
        width: int = 832,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        save_debug_video_to: Optional[str] = None,  # path or None
    ) -> List[Image.Image]:
        """Generate a video conditioned on the provided key‑frames.

        Returns a *list* of `PIL.Image` frames (length == `num_frames`).  You can
        easily turn it into a `.mp4` via `diffusers.utils.export_to_video`.
        """
        device = self.device
        dtype = self.transformer.dtype

        # --------------------------------------------------------------
        # 1. Encode first & last images to CLIP embeddings for attention
        # --------------------------------------------------------------
        enc_img_states = self._encode_first_last_frame(first_image, last_image)

        # --------------------------------------------------------------
        # 2. Build the latent *conditioning* tensor + mask as in notebook
        # --------------------------------------------------------------
        import torchvision.transforms as T

        to_tensor = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # RGB → ‑1…1
        ])
        first_t = to_tensor(first_image).unsqueeze(0)
        last_t = to_tensor(last_image).unsqueeze(0)

        vid = torch.zeros(1, num_frames, 3, height, width, dtype=dtype, device=device)
        vid[:, 0] = first_t.to(dtype=dtype, device=device)
        vid[:, -1] = last_t.to(dtype=dtype, device=device)

        vid = vid.permute(0, 2, 1, 3, 4)  # B × C × F × H × W for VAE

        # -- VAE encode to (mu | logvar)
        moments = self.vae._encode(vid)
        mu, logvar = torch.chunk(moments.to(dtype=dtype), 2, dim=1)

        mu = _normalize_latents(mu, self._latents_mean, self._latents_std_inv)
        logvar = _normalize_latents(logvar, self._latents_mean, self._latents_std_inv)
        latent_condition = torch.cat([mu, logvar], dim=1)

        # Sample latent conditioning (posterior) – matches training behaviour
        posterior = DiagonalGaussianDistribution(latent_condition)
        latent_condition = posterior.sample(generator=generator)

        # -- build mask (first & last) [see notebook logic]
        temporal_downsample = 4
        mask = latent_condition.new_ones(latent_condition.shape[0], 1, num_frames, latent_condition.shape[3], latent_condition.shape[4])  # B ×1 × F × H/8 × W/8
        mask[:, :, 1:-1] = 0

        first_mask = mask[:, :, :1].clone().repeat_interleave(temporal_downsample, dim=2)
        latent_condition_mask = torch.cat([first_mask, mask[:, :, 1:]], dim=2)
        latent_condition_mask = latent_condition_mask.view(
            latent_condition.shape[0],
            -1,
            temporal_downsample,
            latent_condition.shape[-2],
            latent_condition.shape[-1],
        ).transpose(1, 2)

        # --------------------------------------------------------------
        # 3. Initialise noisy latents & set timesteps
        # --------------------------------------------------------------
        latents = torch.randn_like(latent_condition)
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # --------------------------------------------------------------
        # 4. Denoising loop
        # --------------------------------------------------------------
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="inference", unit="step")):
            lat_in = torch.cat([latents, latent_condition_mask, latent_condition], dim=1)
            transformer_out = self.transformer(
                hidden_states=lat_in,
                encoder_hidden_states=encoder_hidden_states.to(device=device, dtype=dtype),
                encoder_hidden_states_image=enc_img_states.to(device=device, dtype=dtype),
                timestep=t.unsqueeze(0).long(),
                return_dict=False,
            )[0]

            if i == 0 and save_debug_video_to is not None:
                # save "x₀" for inspection just like the notebook
                x0 = latents - transformer_out
                x0_denorm = _denormalize_latents(x0, self._latents_mean, self._latents_std_inv)
                video = self.vae.decode(x0_denorm).sample
                frames = self._postprocess(video)
                export_to_video(frames, save_debug_video_to, fps=16)

            latents = self.scheduler.step(transformer_out, t, latents, return_dict=False)[0]

        # --------------------------------------------------------------
        # 5. Decode final latents → frames
        # --------------------------------------------------------------
        latents = _denormalize_latents(latents, self._latents_mean, self._latents_std_inv)
        decoded = self.vae.decode(latents).sample
        return self._postprocess(decoded)


from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from transformers import CLIPVisionModel, CLIPImageProcessor
from finetrainers.models.wan.transformer_wan import WanTransformer3DModel,T2VModel2I2VModelConverter
from diffusers.utils import export_to_video

if __name__ == "__main__":
    # load your checkpoints as before …
    model_id = "/share/project/huangxu/Wan2.1-T2V-1.3B-diffusers"
    # transformer_path = "/share/project/huangxu/wan-t2v-debug-intern-video-clips/model_weights/007500"
    transformer_path = "/share/project/huangxu/wan-t2v-pexel-part2_0/model_weights/001500"


    transformer = WanTransformer3DModel.from_pretrained(transformer_path,subfolder="transformer", torch_dtype=torch.bfloat16)
    converter = T2VModel2I2VModelConverter(transformer)
    converter.convert()
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    # Load image encoder (required for first/last frame conditioning)
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.bfloat16)
    image_processor = CLIPImageProcessor.from_pretrained(model_id, subfolder="image_processor")

    encoder_hidden_states = torch.load("debug_tensors/encoder_hidden_states_t1000.pt").to("cuda")
    scheduler  = FlowMatchEulerDiscreteScheduler()
    pipe = WanKeyFrame2VideoPipeline(
        transformer=transformer,
        vae=vae,
        image_encoder=image_encoder,
        image_processor=image_processor,
        scheduler=scheduler,
    ).to("cuda")
    video_path = "pexel/a39e78046826c99432173630feec2456fe87ca43.mp4"
    # video_path = "validate_video/0CWZMaN4uAE_s006.mp4"
    
    # 从视频中提取第一帧和第16帧
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
    
    # 提取第一帧和第17帧
    first_pil, last_pil = extract_frames_from_video(video_path, first_frame_idx=0, last_frame_idx=16)
    
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
    export_to_video(frames, "_tmp_test.mp4", fps=16)