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

from typing import List, Optional, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
import numpy as np
import decord
from tqdm import tqdm
import os
from omegaconf import OmegaConf
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import sys
sys.path.append(".")
from src.model.ibq_tokenizer import IBQ

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


class WanIBQKeyFrame2VideoPipeline(DiffusionPipeline):
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
        ibq_model,  # IBQModel
        scheduler,  # FlowMatchEulerDiscreteScheduler
        text_encoder=None,  # T5EncoderModel
        tokenizer=None,  # UMT5Tokenizer
        image_encoder=None,  # CLIPVisionModel
        image_processor=None,  # CLIPImageProcessor
        device: Optional[torch.device]=None,
    ):
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.scheduler = scheduler
        self.ibq_model = ibq_model
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Register for save / from_pretrained – optional but handy
        self.register_modules(
            transformer=transformer,
            vae=vae,
            image_encoder=image_encoder,
            image_processor=image_processor,
            scheduler=scheduler,
            ibq_model=ibq_model,
            text_encoder=text_encoder,
        )

        # Pre‑compute mean / std constants for (de)normalisation
        self._latents_mean = torch.tensor(vae.config.latents_mean)
        self._latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std)
        if self.transformer.config.get("image_dim", None) is not None:
            self.using_cross_attn = True
        else:
            self.using_cross_attn = False

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[str, torch.device]):
        self._device = torch.device(device)
        return super().to(device)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()
        self.text_encoder.to(device)
        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds
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

    def _ibq_encode(self,images: torch.Tensor) -> torch.Tensor:
        quant, qloss, (_, _, indices) = self.ibq_model.encode(images)   # 归一化到[-1,1] 的 image tensor, [B, 3, H, W]
        return quant, qloss, indices    # quants: [TODO]

    def _postprocess(self, video: torch.Tensor) -> List[Image.Image]:
        """Convert decoded tensor [B,C,F,H,W] in [-1,1] to PIL list."""
        video = video.squeeze(0).permute(1, 0, 2, 3)  # F × C × H × W
        video = (video / 2 + 0.5).clamp(0, 1)
        video = (video * 255).to(torch.uint8).cpu().numpy()
        return [Image.fromarray(f.transpose(1, 2, 0), "RGB") for f in video]

    def _quant_to_3d_latent(self, key_frames_quants: torch.Tensor, key_frames_indices: torch.Tensor,num_frames: int) -> torch.Tensor:
        """Convert quantized latents to 3D latents."""
        B, F, C, H, W = key_frames_quants.shape
        
        # First upsample H and W dimensions by 2x using interpolate
        key_frames_quants = torch.nn.functional.interpolate(
            key_frames_quants.view(B*F, C, H, W),
            scale_factor=2,
            mode='nearest'
        ).view(B, F, C, H*2, W*2)
        
        # Create empty tensor of target size filled with zeros
        output = torch.zeros(B, C, (num_frames - 1) // 4 + 1, H*2, W*2, device=key_frames_quants.device, dtype=key_frames_quants.dtype)
        
        # For each batch, place the quants at the indices specified by key_frames_indices
        # Transform key frame indices according to the rule:
        # f(1) = 0, f(t) = (t-1)//4 + 1 for t > 1
        transformed_indices = torch.where(key_frames_indices == 1, 
                                        torch.zeros_like(key_frames_indices),
                                        (key_frames_indices - 1) // 4 + 1)
        # Check that transformed indices are unique within each batch
        for b in range(B):
            assert len(transformed_indices[b].unique()) == len(transformed_indices[b]), \
                f"Transformed key frame indices must be unique within each batch, but got duplicates in batch {b}"
        
        for b in range(B):
            output[b, :, transformed_indices[b].long()] = key_frames_quants[b].transpose(0, 1)
        
        return output
        

    @torch.no_grad()
    def __call__(
        self,
        key_frames: torch.Tensor,  # [B,F',3,H,W] F' 是 key_frames 的帧数
        key_frames_indices: torch.Tensor,  # [B,F'] F' 是 key_frames 的帧数
        encoder_hidden_states: Optional[torch.Tensor] = None,  # pre‑encoded text embeddings
        prompt: Optional[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,  # Added seed parameter
        save_debug_video_to: Optional[str] = None,  # path or None
        guidance_scale: float = 5.0,
        *args,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate a video conditioned on the provided key‑frames.

        Returns a *list* of `PIL.Image` frames (length == `num_frames`).  You can
        easily turn it into a `.mp4` via `diffusers.utils.export_to_video`.
        """
        device = self.device
        dtype = self.transformer.dtype
        if prompt is None and encoder_hidden_states is None:
            raise ValueError("prompt or encoder_hidden_states must be provided")
        if prompt is not None and encoder_hidden_states is not None:
            raise ValueError("prompt and encoder_hidden_states cannot be provided at the same time")
        if prompt is not None:
            encoder_hidden_states = self._get_t5_prompt_embeds(prompt)
            uncond_embeds = self._get_t5_prompt_embeds(prompt="")

        import torch.nn.functional as func_F
        B, F = key_frames.shape[0], key_frames.shape[1]
        assert key_frames_indices.shape == (B, F), f"key_frames_indices.shape : {key_frames_indices.shape}, should be (B, F) = ({B}, {F})"
        
        # First reshape key_frames to combine batch and frame dimensions
        key_frames = key_frames.view(B * F, 3, key_frames.shape[3], key_frames.shape[4])
        
        
        # Resize key_frames (already tensor) and normalize
        key_frames = (key_frames.float() / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1] if input is [0, 255]
        key_frames = func_F.interpolate(key_frames, size=(height, width), mode='bilinear', align_corners=False)
        key_frames = key_frames.to(device=device, dtype=dtype) # [B*F',3,H,W]
        key_frames_quants, key_frames_qloss, _ = self._ibq_encode(key_frames) # [B*F',256,H/16,W/16]
        if self.using_cross_attn:
            encoder_hidden_states_image = key_frames_quants.permute(0, 2, 1, 3, 4).flatten(2)   # [B,256,F'*H/16*W/16]
            encoder_hidden_states_image = encoder_hidden_states_image.permute(0, 2, 1)   # [B,F'*H/16*W/16,256]
            encoder_hidden_states_image = encoder_hidden_states_image.to(device=device, dtype=dtype)
        else:
            encoder_hidden_states_image = None
        key_frames_quants = key_frames_quants.view(B, F, 256, height//16, width//16) # Reshape back to [B,F',256,H/16,W/16]
        latent_condition = self._quant_to_3d_latent(key_frames_quants, key_frames_indices, num_frames) # [B,256,F,H/8,W/8]

        

        # Create mask with same shape as latent condition, initialized to zeros
        mask = latent_condition.new_zeros(latent_condition.shape[0], 1, num_frames, latent_condition.shape[3], latent_condition.shape[4]).to(device=device)  # B x 1 x F x H/8 x W/8
        
        # Set mask to 1 at key frame indices
        for b in range(latent_condition.shape[0]):
            mask[b, :, key_frames_indices[b].long()] = 1 # B x 1 x F x H/8 x W/8

        # Reshape mask similar to latent condition
        first_mask = mask[:, :, :1].clone().repeat_interleave(4, dim=2)  # Repeat first frame 4 times
        latent_condition_mask = torch.cat([first_mask, mask[:, :, 1:]], dim=2)
        latent_condition_mask = latent_condition_mask.view(
            latent_condition.shape[0],
            -1,
            4,  # Temporal downsample factor of 4
            latent_condition.shape[-2],
            latent_condition.shape[-1],
        ).transpose(1, 2)

        # --------------------------------------------------------------
        # 3. Initialise noisy latents & set timesteps
        # --------------------------------------------------------------
        # Set generator seed if provided
        if seed is not None:
            if generator is None:
                generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        latents = torch.randn(B, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8, device=device, dtype=dtype, generator=generator)    # 16 vae channels
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # --------------------------------------------------------------
        # 4. Denoising loop
        # --------------------------------------------------------------

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="inference", unit="step")):
            lat_in = torch.cat([latents, latent_condition_mask, latent_condition], dim=1)
            un_cond_lat_in = torch.cat([latents, torch.zeros_like(latent_condition_mask), torch.zeros_like(latent_condition)], dim=1)
            if guidance_scale > 0:
                transformer_out = self.transformer(
                    hidden_states=lat_in,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_image=encoder_hidden_states_image,
                    timestep=t.unsqueeze(0).long(),
                    return_dict=False,
                )[0]
                un_cond_transformer_out = self.transformer(
                    hidden_states=un_cond_lat_in,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_image=encoder_hidden_states_image,
                    timestep=t.unsqueeze(0).long(),
                    return_dict=False,
                )[0]
                transformer_out = transformer_out + guidance_scale * (transformer_out - un_cond_transformer_out)
            else:
                transformer_out = self.transformer(
                    hidden_states=lat_in,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_image=encoder_hidden_states_image,
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


from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanTransformer3DModel, WanPipeline
from transformers import CLIPVisionModel, CLIPImageProcessor, UMT5EncoderModel, AutoTokenizer
from diffusers.utils import export_to_video
import decord

if __name__ == "__main__":
    # load your checkpoints as before …
    transformer_path = "/share/project/huangxu/model/wan-ibq-key-frame-pixabay-img/model_weights/004000"
    model_id = "/share/project/huangxu/model/Wan2.1-T2V-1.3B-diffusers"


    transformer = WanTransformer3DModel.from_pretrained(transformer_path,subfolder="transformer", torch_dtype=torch.bfloat16)
    for name, param in transformer.named_parameters():
        if "patch_embedding.weight" in name:
            print(param.shape)

            part_param = param[:,16:,:]
            print(part_param.sum())
            print(part_param.min())
            print(part_param.max())
            print(part_param.norm(1))
            print(part_param.norm(1) / param.norm(1))

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    # Load image encoder (required for first/last frame conditioning)
    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.bfloat16)
    image_processor = CLIPImageProcessor.from_pretrained(model_id, subfolder="image_processor")
    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    encoder_hidden_states = torch.load("debug_tensors/encoder_hidden_states_t1000.pt").to("cuda")
    tokenize_path = "/share/project/zhangfan/weights/Emu3.5-Tokenizer/IBQ-XL-f16c131k-FI"
    config_name = "fusimage_ibqgan_xl_131072_siglip.yaml"
    config = OmegaConf.load(os.path.join(tokenize_path, config_name))

    # Initialize model with config
    ibq_model = IBQ(**config.model.init_args).to(dtype=torch.bfloat16)
    ckpt_name = "fusionimage_256_XL_f16c131k.ckpt"
    ckpt = torch.load(os.path.join(tokenize_path, ckpt_name), weights_only=True)

    # Load state dict
    ibq_model.load_state_dict(ckpt["state_dict"])
    ibq_model.to("cuda")
    scheduler =  FlowMatchEulerDiscreteScheduler()
    pipe = WanIBQKeyFrame2VideoPipeline(
        text_encoder=text_encoder,
        transformer=transformer,
        ibq_model=ibq_model,
        vae=vae,
        scheduler=scheduler,
        tokenizer=tokenizer,
    ).to("cuda")
    video_path = "a39e78046826c99432173630feec2456fe87ca43.mp4"
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    key_frames_indices = torch.Tensor([0]).to("cuda")
    key_frames_indices = key_frames_indices.unsqueeze(0)
    key_frames = vr.get_batch([0]).to("cuda")    # [F,H,W,3]
    key_frames = key_frames.permute(0, 3, 1, 2)    # [F,3,H,W]
    key_frames = key_frames.unsqueeze(0)    # [B,F,3,H,W] B = 1
    
    
    video = pipe(
        encoder_hidden_states=encoder_hidden_states,
        seed=42,
        key_frames=key_frames,
        key_frames_indices=key_frames_indices,
        save_debug_video_to="first_frame.mp4",
        height=480,
        width=832,
        num_frames=1,
        num_inference_steps=100,
        guidance_scale=0
    )
    # flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
    # pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    # video = pipe(
    #     prompt="beautiful girls",
    #     height=480,
    #     width=832,
    #     num_frames=49,
    #     guidance_scale=0
    # ).frames[0]
    export_to_video(video, "cfg0.mp4", fps=16)
