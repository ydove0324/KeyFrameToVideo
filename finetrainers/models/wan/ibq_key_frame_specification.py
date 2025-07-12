import functools
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
)
from diffusers import WanTransformer3DModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from torchvision import transforms

import finetrainers.functional as FF
from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.processors import ProcessorMixin, T5Processor
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import get_non_null_items, safetensors_torch_save_function
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from omegaconf import OmegaConf
import numpy as np
import sys
sys.path.append(".")
from src import WanIBQKeyFrame2VideoPipeline
from src.model.ibq_tokenizer import IBQ


logger = get_logger()

# Initialize the ToPILImage transform for video conversion
to_pil_image = transforms.ToPILImage(mode="RGB")


def _denormalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    """Undo the affine transform."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    return ((latents.float() / latents_std) + latents_mean).to(latents)


def export_video_tensor_to_file(video_tensor: torch.Tensor, output_path: str, fps: int = 16) -> None:
    """
    Export a video tensor to a video file.
    
    Args:
        video_tensor: Video tensor with shape [C, F, H, W] where C=3 (RGB channels), 
                     F=frames, H=height, W=width. Values should be in [-1, 1] range.
        output_path: Path to save the output video file (e.g., "output.mp4")
        fps: Frame rate for the output video
    """
    # Convert from [C, F, H, W] to [F, C, H, W] (frames first)
    video = video_tensor.permute(1, 0, 2, 3)  # [F, C, H, W]
    
    # Clamp values to [-1, 1] range and convert to float32
    video = video.to(dtype=torch.float32).clamp(-1, 1)
    
    # Convert from [-1, 1] to [0, 1] range for PIL Image
    video = (video + 1.0) / 2.0
    
    # Convert each frame to PIL Image
    video_frames = [to_pil_image(frame).convert("RGB") for frame in video]
    
    # Export to video file
    export_to_video(video_frames, output_path, fps=fps)


class WanLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - latents: The latents of the input image/video.
            - latents_mean: The channel-wise mean of the latent space.
            - latents_std: The channel-wise standard deviation of the latent space.
    """

    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 3

    def forward(
        self,
        vae: AutoencoderKLWan,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if image is not None:
            video = image.unsqueeze(1)

        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]

        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            # TODO(aryan): refactor in diffusers to have use_slicing attribute
            # if vae.use_slicing and video.shape[0] > 1:
            #     encoded_slices = [vae._encode(x_slice) for x_slice in video.split(1)]
            #     moments = torch.cat(encoded_slices)
            # else:
            #     moments = vae._encode(video)
            moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)

        return {self.output_names[0]: latents, self.output_names[1]: latents_mean, self.output_names[2]: latents_std}


class WanImageConditioningIBQLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode key frames into IBQ latents.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - ibq_latents: The IBQ-encoded latents of the key frames.
            - mask: The conditioning frame mask for the key frames.
    """

    def __init__(self, output_names: List[str],*,return_hidden_states: bool = False):
        super().__init__()
        self.output_names = output_names
        self.return_hidden_states = return_hidden_states

    def _ibq_encode(self, ibq_model, images: torch.Tensor):
        """Encode images using IBQ model with batch processing.
        
        Args:
            ibq_model: The IBQ model to use for encoding
            images: Input images tensor of shape [N, C, H, W]
            
        Returns:
            Tuple of (quant, qloss, indices) tensors
        """
        MAX_BATCH_SIZE = 2
        total_size = images.size(0)
        
        # Initialize lists to store results
        all_quants = []
        all_qlosses = []
        all_indices = []
        
        # Process in batches
        for start_idx in range(0, total_size, MAX_BATCH_SIZE):
            end_idx = min(start_idx + MAX_BATCH_SIZE, total_size)
            batch_images = images[start_idx:end_idx]
            
            # Encode current batch
            quant, qloss, (_, _, indices) = ibq_model.encode(batch_images)
            
            # Store results
            all_quants.append(quant)
            all_qlosses.append(qloss)
            all_indices.append(indices)
        
        # Concatenate all results
        final_quant = torch.cat(all_quants, dim=0)
        
        return final_quant, all_qlosses, all_indices

    def _quant_to_3d_latent(self, key_frames_quants: torch.Tensor, key_frames_indices: torch.Tensor, num_frames: int) -> torch.Tensor:
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

    def forward(
        self,
        ibq_model,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,   # [B,F,3,H,W]
        key_frames_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if video is None:
            raise ValueError("video must be provided for IBQ key frame processing")
        if key_frames_indices is None:
            raise ValueError("key_frames_indices must be provided for IBQ key frame processing")
        if key_frames_indices.ndim == 1:
            key_frames_indices = key_frames_indices.unsqueeze(0)

        device = ibq_model.device
        dtype = ibq_model.dtype

        # Extract key frames from video based on key_frames_indices
        # video: [B, F, C, H, W], key_frames_indices: [B, _F]
        B, F, C, H, W = video.shape
        _F = key_frames_indices.shape[1]  # Number of key frames
        
        # Extract key frames
        key_frames = []
        for b in range(B):
            batch_key_frames = video[b, key_frames_indices[b].long()]  # [_F, C, H, W]
            key_frames.append(batch_key_frames)
        key_frames = torch.stack(key_frames, dim=0)  # [B, _F, C, H, W]
        
        # Reshape key_frames to combine batch and frame dimensions
        key_frames = key_frames.view(B * _F, C, H, W)
       
        assert key_frames.max() < 1.0 + 2e-1 and key_frames.min() > -1.0 - 2e-1, "Key frames must be normalized to [-1, 1]"
        key_frames = key_frames.to(device=device, dtype=dtype)  # [B*_F, C, H, W]
        key_frames_quants, _, _ = self._ibq_encode(ibq_model, key_frames)  # [B*_F, 256, H/16, W/16]
        key_frames_quants = key_frames_quants.view(B, _F, 256, H//16,W//16)  # [B, _F, 256, H/16, W/16]
        if self.return_hidden_states:
            # Reshape to [B, 256, (H/16 * W/16 * _F)]
            ibq_encode_hidden_states = key_frames_quants.permute(0, 2, 1, 3, 4).flatten(2)
            ibq_encode_hidden_states = ibq_encode_hidden_states.permute(0, 2, 1)    # [B,F'*H/16*W/16,256]
            ibq_encode_hidden_states = ibq_encode_hidden_states.to(device=device, dtype=dtype)
        
        # Convert to 3D latents
        ibq_latents = self._quant_to_3d_latent(key_frames_quants, key_frames_indices, F)  # [B, 256, (num_frames-1)//4+1, H/8, W/8]
        
        # Create mask with same shape as latent condition, initialized to zeros
        mask = ibq_latents.new_zeros(ibq_latents.shape[0], 1, F, ibq_latents.shape[3], ibq_latents.shape[4]).to(device=device)  # B x 1 x F x H/8 x W/8
        
        # Set mask to 1 at key frame indices
        for b in range(ibq_latents.shape[0]):
            mask[b, :, key_frames_indices[b].long()] = 1  # B x 1 x F x H/8 x W/8

        # Reshape mask similar to latent condition
        first_mask = mask[:, :, :1].clone().repeat_interleave(4, dim=2)  # Repeat first frame 4 times
        latent_condition_mask = torch.cat([first_mask, mask[:, :, 1:]], dim=2)
        latent_condition_mask = latent_condition_mask.view(
            ibq_latents.shape[0],
            -1,
            4,  # Temporal downsample factor of 4
            ibq_latents.shape[-2],
            ibq_latents.shape[-1],
        ).transpose(1, 2)
        if self.return_hidden_states:
            return {
                self.output_names[0]: ibq_latents,
                self.output_names[1]: latent_condition_mask,
                self.output_names[2]: ibq_encode_hidden_states
            }
        else:
            return {
                self.output_names[0]: ibq_latents,
                self.output_names[1]: latent_condition_mask,
            }


class WanImageEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encoding image conditioning for Wan I2V training.

    Args:
        output_names (`List[str]`):
            The names of the outputs that the processor returns. The outputs are in the following order:
            - image_embeds: The CLIP vision model image embeddings of the input image.
    """

    def __init__(self, output_names: List[str], *, use_last_frame: bool = False):
        super().__init__()
        self.output_names = output_names
        self.use_last_frame = use_last_frame
        assert len(self.output_names) == 1

    def forward(
        self,
        image_encoder: CLIPVisionModel,
        image_processor: CLIPImageProcessor,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = image_encoder.device
        dtype = image_encoder.dtype
        last_image = None

        # We know the image here is in the range [-1, 1] (probably a little overshot if using bilinear interpolation), but
        # the processor expects it to be in the range [0, 1].
        image = image if video is None else video[:, 0]  # [B, F, C, H, W] -> [B, C, H, W] (take first frame)
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        # image = FF.normalize(image, min=0.0, max=1.0, dim=1)    # 这里的 normalize 为什么会这样呢
        assert image.ndim == 4, f"Expected 4D tensor, got {image.ndim}D tensor"

        if self.use_last_frame:
            last_image = image if video is None else video[:, -1]
            last_image = torch.clamp((last_image + 1.0) / 2.0, min=0.0, max=1.0)
            # last_image = FF.normalize(last_image, min=0.0, max=1.0, dim=1)
            # Process both images separately to maintain batch dimension
            first_image_processed = image_processor(images=image.float(), do_rescale=False, do_convert_rgb=False, return_tensors="pt")
            last_image_processed = image_processor(images=last_image.float(), do_rescale=False, do_convert_rgb=False, return_tensors="pt")
            
            first_image_processed = first_image_processed.to(device=device, dtype=dtype)
            last_image_processed = last_image_processed.to(device=device, dtype=dtype)
            
            # Get embeddings for both images
            first_image_embeds = image_encoder(**first_image_processed, output_hidden_states=True).hidden_states[-2]
            last_image_embeds = image_encoder(**last_image_processed, output_hidden_states=True).hidden_states[-2]
            
            # Concatenate in sequence dimension: [B, 257*2, hidden_dim]
            image_embeds = torch.cat([first_image_embeds, last_image_embeds], dim=1)
        else:
            image = image_processor(images=image.float(), do_rescale=False, do_convert_rgb=False, return_tensors="pt")
            image = image.to(device=device, dtype=dtype)
            image_embeds = image_encoder(**image, output_hidden_states=True)
            image_embeds = image_embeds.hidden_states[-2]
        return {self.output_names[0]: image_embeds}
        

class WanIBQKeyFrame2VideoModelSpecification(ModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        train_modules: List[str] = None,
        **kwargs,
    ) -> None:  # 在一开始的时候把 Conv3D 添加一层，然后再看看 cross-attn 在哪加入
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )

        if condition_model_processors is None:      # 要在这里定义清楚 condition 怎么写
            condition_model_processors = [T5Processor(["encoder_hidden_states", "__drop__"])]
        if latent_model_processors is None:             # ATTN! TODO, 搞清楚这里是怎么设计的
            latent_model_processors = [WanLatentEncodeProcessor(["latents", "latents_mean", "latents_std"])]
            if self.transformer_config.get("image_dim", None) is None:
                latent_model_processors.append(
                    WanImageConditioningIBQLatentEncodeProcessor(
                        ["ibq_latent_condition", "latent_condition_mask"],
                        return_hidden_states=False
                    )
                )
            else:
                latent_model_processors.append(
                    WanImageConditioningIBQLatentEncodeProcessor(
                        ["ibq_latent_condition", "latent_condition_mask","encoder_hidden_states_image"],
                        return_hidden_states=True
                    )
                )

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors
        self.train_modules = train_modules

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.vae_id is not None:
            vae = AutoencoderKLWan.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLWan.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        # Load IBQ model
        tokenize_path = "/share/project/zhangfan/weights/Emu3.5-Tokenizer/IBQ-XL-f16c131k-FI"
        config_name = "fusimage_ibqgan_xl_131072_siglip.yaml"
        config = OmegaConf.load(os.path.join(tokenize_path, config_name))
        ibq_model = IBQ(**config.model.init_args).to(dtype=torch.bfloat16)
        ckpt_name = "fusionimage_256_XL_f16c131k.ckpt"
        ckpt = torch.load(os.path.join(tokenize_path, ckpt_name), weights_only=True)

        # Load state dict
        ibq_model.load_state_dict(ckpt["state_dict"])

        # Ensure IBQ model uses the same device allocation pattern as other models
        # The device will be set later by the trainer's _move_components_to_device method

        models = {"vae": vae, "ibq_model": ibq_model}       # 这里 load 的都会被 sft_trainer 的 trainer 加载成 self.xxxx eg: self.ibq_model
        
        return models

    @property
    def ibq_model_dtype(self) -> torch.dtype:
        """Get the dtype for IBQ model, consistent with other model dtypes."""
        return torch.bfloat16

    def load_diffusion_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = WanTransformer3DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = WanTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        def wrap_trainable_modules():
            if self.train_modules:
                self.set_modules_trainable(transformer, self.train_modules)
            else:
                transformer.requires_grad = True
            return transformer
        transformer = wrap_trainable_modules()
        scheduler = FlowMatchEulerDiscreteScheduler()

        return {"transformer": transformer, "scheduler": scheduler}


    def load_pipeline(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        text_encoder: Optional[UMT5EncoderModel] = None,
        transformer: Optional[WanTransformer3DModel] = None,
        vae: Optional[AutoencoderKLWan] = None,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        image_encoder: Optional[CLIPVisionModel] = None,
        image_processor: Optional[CLIPImageProcessor] = None,
        ibq_model: Optional[IBQ] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Union[WanPipeline, WanImageToVideoPipeline, WanIBQKeyFrame2VideoPipeline]:
        if scheduler is None:
            scheduler = FlowMatchEulerDiscreteScheduler()
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "ibq_model": ibq_model
        }
        components = get_non_null_items(components)
        # if self.transformer_config.get("image_dim", None) is not None:  原始代码，这似乎是一个 bug
        pipe = WanIBQKeyFrame2VideoPipeline(**components)
        if pipe.text_encoder is not None:
            pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)
        self.vae = pipe.vae
        if not training:
            pipe.transformer.to(self.transformer_dtype)

        # Add IBQ model GPU allocation following the same pattern
        if hasattr(pipe, 'ibq_model') and pipe.ibq_model is not None:
            pipe.ibq_model.to(self.ibq_model_dtype)

        # TODO(aryan): add support in diffusers
        # if enable_slicing:
        #     pipe.vae.enable_slicing()
        # if enable_tiling:
        #     pipe.vae.enable_tiling()
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        caption: str,
        max_sequence_length: int = 512,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "caption": caption,
            "max_sequence_length": max_sequence_length,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_conditions(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    @torch.no_grad()
    def prepare_latents(    # 这个地方要改，不需要 vae，直接用 ibq 的 latent 来处理
        self,
        vae: AutoencoderKLWan,
        ibq_model,  # 替换 image_encoder
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        key_frames_indices: Optional[torch.Tensor] = None,  # 新增参数
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:       # image condition 放在 latent 里
        # 如果没有提供 key_frames_indices，默认使用首尾帧
        if key_frames_indices is None and video is not None:
            batch_size, num_frames = video.shape[0], video.shape[1]
            # 首尾帧索引：第1帧(索引0)和最后一帧
            first_frame_idx = torch.zeros(batch_size, 1, dtype=torch.long, device=video.device)
            last_frame_idx = torch.full((batch_size, 1), num_frames - 1, dtype=torch.long, device=video.device)
            key_frames_indices = torch.cat([first_frame_idx, last_frame_idx], dim=1)  # [B, 2]
        if video is None and image is not None:
            video = image.unsqueeze(1) # frame dimension to make [B,1,C,H,W]
            key_frames_indices = torch.zeros(image.shape[0], 1, dtype=torch.long, device=image.device) # [B,1] all zeros
            image = None
        
        conditions = {
            "vae": vae,
            "ibq_model": ibq_model,  # 替换 image_encoder 和 image_processor
            "image": image,
            "video": video,
            "key_frames_indices": key_frames_indices,  # 新增参数
            "generator": generator,
            # We must force this to False because the latent normalization should be done before
            # the posterior is computed. The VAE does not handle this any more:
            # https://github.com/huggingface/diffusers/pull/10998
            "compute_posterior": False,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    def forward(    # TODO: 这个地方要改
        self,
        transformer: WanTransformer3DModel,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        compute_posterior = False  # See explanation in prepare_latents    
        latent_condition = latent_condition_mask = None
        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
            latent_condition = latent_model_conditions.pop("latent_condition", None)
            latent_condition_mask = latent_model_conditions.pop("latent_condition_mask", None)
        else:
            latents = latent_model_conditions.pop("latents")
            original_latents = latents.clone()
            latents_mean = latent_model_conditions.pop("latents_mean")
            latents_std = latent_model_conditions.pop("latents_std")
            latent_condition = latent_model_conditions.pop("ibq_latent_condition", None)
            latent_condition_mask = latent_model_conditions.pop("latent_condition_mask", None)

            mu, logvar = torch.chunk(latents, 2, dim=1)
            mu = self._normalize_latents(mu, latents_mean, latents_std)
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)
            latents = torch.cat([mu, logvar], dim=1)

            posterior = DiagonalGaussianDistribution(latents)
            latents = posterior.sample(generator=generator)
            del posterior
        noise = torch.zeros_like(latents).normal_(generator=generator)
        noisy_latents = FF.flow_match_xt(latents, noise, sigmas)
        _noisy_latents = noisy_latents.clone()
        timesteps = (sigmas.flatten() * 1000.0).long()

        noisy_latents = torch.cat([noisy_latents, latent_condition_mask, latent_condition], dim=1)

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents)
        pred = transformer(     # 如果有 encoder_hidden_states_image 的话，自然会把 encoder_hidden_states_image 传进去
                **latent_model_conditions,
                **condition_model_conditions,
                timestep=timesteps,
                return_dict=False,
            )[0]
        target = FF.flow_match_target(noise, latents)
        DEBUG = False
        if DEBUG:
            # noise - x_0 = pred => x_0 = noise - pred
            debug_dir = "debug_tensors"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save hidden_states
            timestep_val = timesteps[0].item() if len(timesteps) > 0 else 0
            torch.save(
                latent_model_conditions["hidden_states"].detach().cpu(),
                os.path.join(debug_dir, f"hidden_states_t{timestep_val}.pt")
            )
            torch.save(
                condition_model_conditions["encoder_hidden_states"].detach().cpu(),
                os.path.join(debug_dir, f"encoder_hidden_states_t{timestep_val}.pt")
            )
            torch.save(
                noise.detach().cpu(),
                os.path.join(debug_dir, f"noise_t{timestep_val}.pt")
            )
            torch.save(
                latent_condition.detach().cpu(),
                os.path.join(debug_dir, f"latent_condition_t{timestep_val}.pt")
            )
            torch.save(
                latent_condition_mask.detach().cpu(),
                os.path.join(debug_dir, f"latent_condition_mask_t{timestep_val}.pt")
            )
            # Save encoder_hidden_states_image if it exists
            if "encoder_hidden_states_image" in latent_model_conditions:
                torch.save(
                    latent_model_conditions["encoder_hidden_states_image"].detach().cpu(),
                    os.path.join(debug_dir, f"encoder_hidden_states_image_t{timestep_val}.pt")
                )
            
            print(f"Saved debug tensors for timestep {timestep_val} to {debug_dir}/")
            x_0 = noise - pred
            x_0 = _denormalize_latents(x_0, latents_mean, latents_std)
            torch.save(x_0.detach().cpu(), os.path.join(debug_dir, f"x_0_t{timestep_val}.pt"))
            # print(f"x_0 shape: {x_0.shape}, noise shape: {noise.shape}, pred shape: {pred.shape}")
            # print(f"x_0: {x_0}, noise: {noise}, pred: {pred}")
            vae = AutoencoderKLWan.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype
            ).to(x_0.device)
            with torch.no_grad():
                video = vae.decode(x_0).sample[0]
            with torch.no_grad():
                video_2 = vae.decode(_denormalize_latents(_noisy_latents, latents_mean, latents_std).to(self.vae_dtype)).sample[0]
            print(f"video shape: {video.shape}")
            with torch.no_grad():
                no_img_nosiy_latents = torch.cat([_noisy_latents, torch.zeros_like(latent_condition_mask), torch.zeros_like(latent_condition)], dim=1)
                latent_model_conditions["hidden_states"] = no_img_nosiy_latents.to(latents)
                # latent_model_conditions["encoder_hidden_states_image"] = torch.zeros_like(latent_model_conditions["encoder_hidden_states_image"])
                pred_no_img = transformer(
                    **latent_model_conditions,
                    **condition_model_conditions,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
                x_0_no_img = noise - pred_no_img
                x_0_no_img = _denormalize_latents(x_0_no_img, latents_mean, latents_std)
                with torch.no_grad():
                    video_3 = vae.decode(x_0_no_img).sample[0]
            
            # Export video tensor to file using the helper function
            export_video_tensor_to_file(video, f"debug_video/debug_video_{timesteps[0]}.mp4", fps=16)
            export_video_tensor_to_file(video_2, f"debug_video/debug_noisy_video_{timesteps[0]}.mp4", fps=16)
            export_video_tensor_to_file(video_3, f"debug_video/debug_no_img_video_{timesteps[0]}.mp4", fps=16)

        return pred, target, sigmas

    def validation(
        self,
        pipeline: Union[WanPipeline, WanImageToVideoPipeline, WanIBQKeyFrame2VideoPipeline],
        prompt: str,
        image: Optional[PIL.Image.Image] = None,
        last_image: Optional[PIL.Image.Image] = None,
        video: Optional[List[PIL.Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        key_frames_indices: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[ArtifactType]:
        generation_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "key_frames_indices": key_frames_indices,
            "output_type": "pil",
        }
        if video is None:
            raise ValueError("video must be provided for WanIBQKeyFrame2Video validation.")
        # Extract key frames from video based on key_frames_indices
        # Convert video PIL Image list to tensor [B,F,H,W,C] format
        video_tensor = torch.stack([torch.from_numpy(np.array(frame)) for frame in video])  # [F,H,W,C]
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dim [1,F,H,W,C]

        # Extract key frames if indices provided
        if key_frames_indices is not None:
            # Gather key frames using indices and convert to [B,F,3,H,W] format
            key_frames = video_tensor.gather(1, key_frames_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, video_tensor.shape[2], video_tensor.shape[3], video_tensor.shape[4]))
            key_frames = key_frames.permute(0, 1, 4, 2, 3)  # [B,F,3,H,W]
            generation_kwargs["key_frames"] = key_frames
        else:
            raise ValueError("key_frames_indices must be provided for WanIBQKeyFrame2Video validation.")
        video = pipeline(**generation_kwargs)
        return [VideoArtifact(value=video)]

    def _save_lora_weights(
        self,
        directory: str,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
        metadata: Optional[Dict[str, str]] = None,
        *args,
        **kwargs,
    ) -> None:
        pipeline_cls = (
            WanImageToVideoPipeline if self.transformer_config.get("image_dim", None) is not None else WanPipeline
        )
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            pipeline_cls.save_lora_weights(
                directory,
                transformer_state_dict,
                save_function=functools.partial(safetensors_torch_save_function, metadata=metadata),
                safe_serialization=True,
            )
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model(    # TODO,mock 看看这个怎么搞
        self,
        directory: str,
        transformer: WanTransformer3DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = WanTransformer3DModel.from_config(self.transformer_config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def set_modules_trainable(self, transformer: WanTransformer3DModel, module_patterns: List[str]) -> None:
        """Set requires_grad for parameters based on module name patterns.
        Only parameters containing any of the patterns in their names will be set to trainable.
        
        Args:
            transformer (WanTransformer3DModel): The transformer model to modify
            module_patterns (List[str]): List of patterns to match against parameter names
        """
        # First set all parameters to non-trainable
        for param in transformer.parameters():
            param.requires_grad = False
            
        # Then set specified modules to trainable
        for name, param in transformer.named_parameters():
            if any(pattern in name for pattern in module_patterns):
                param.requires_grad = True
        
        # Log number of trainable parameters
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in transformer.parameters())
        logger.info(f"Number of trainable parameters: {trainable_params:,} out of {total_params:,}")
        logger.info(f"Training modules matching patterns: {module_patterns}")

