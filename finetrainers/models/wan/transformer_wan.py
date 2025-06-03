# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Optional, Tuple, Union

import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
import torch.cuda.amp as amp 


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor: # 把 image dim(CLIP) 转换成 Wan 的 dim
        with amp.autocast(dtype=torch.bfloat16):  # TODO: 先全部转成 bf16 来跑，先跑通
            hidden_states = self.norm1(encoder_hidden_states_image)
            hidden_states = self.ff(hidden_states)
            hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)       # 把这个 mock 进来

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs


class WanTransformerBlock(nn.Module):   # 把这个整个 mock 掉，想一下怎么拓展
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=WanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(x=norm_hidden_states, context=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(    # 需要把这个也mock一下
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )       # 把 hidden_states_image 正常输入即可
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class T2VModel2I2VModelConverter:
    def __init__(self, transformer: WanTransformer3DModel, image_dim: int = 1280, in_channels: int = 36,inner_dim: int = 1536, transformer_config: dict = None):
        self.transformer = transformer
        self.image_dim = image_dim
        self.in_channels = in_channels
        self.transformer_config = transformer_config
        self.inner_dim = inner_dim
    def convert(self):
        self.transformer = self.mock_conv3d(self.transformer, new_in_channels=self.in_channels)
        self.transformer = self.mock_crossattention(self.transformer)
        self.transformer = self.mock_condition_embedder(self.transformer)
        self.transformer.config.in_channels = self.in_channels
        self.transformer.config.image_dim = self.image_dim

    def mock_conv3d(self, transformer: WanTransformer3DModel, new_in_channels: int = 36) -> WanTransformer3DModel:
        """
        Mock the Conv3D layer in the transformer to support different input channels.
        """
        # Get original conv3d parameters
        original_conv3d = transformer.patch_embedding
        original_dtype = original_conv3d.weight.dtype
        original_device = original_conv3d.weight.device
        
        out_channels = original_conv3d.out_channels
        kernel_size = original_conv3d.kernel_size
        stride = original_conv3d.stride
        padding = original_conv3d.padding
        
        # Create new conv3d layer with desired input channels
        new_conv3d = nn.Conv3d(
            in_channels=new_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Convert to same dtype and device as original
        new_conv3d = new_conv3d.to(dtype=original_dtype, device=original_device)
        
        # Initialize weights
        with torch.no_grad():
            # Copy original weights for the first channels
            original_in_channels = original_conv3d.in_channels
            min_channels = min(original_in_channels, new_in_channels)
            
            new_conv3d.weight[:, :min_channels] = original_conv3d.weight[:, :min_channels]
            
            # If new_in_channels > original_in_channels, zero-initialize remaining channels
            if new_in_channels > original_in_channels:
                new_conv3d.weight[:, original_in_channels:] = 0.0
            
            # Copy bias
            new_conv3d.bias.data = original_conv3d.bias.data.clone()
        
        # Ensure all parameters require gradients
        new_conv3d.weight.requires_grad_(True)
        new_conv3d.bias.requires_grad_(True)
        
        # Replace the conv3d layer
        transformer.patch_embedding = new_conv3d
        
        return transformer

    def mock_condition_embedder(self, transformer: WanTransformer3DModel) -> WanTransformer3DModel:
        """
        Mock the condition embedder in the transformer to support different input channels.
        """
        original_condition_embedder = transformer.condition_embedder
        assert original_condition_embedder.image_embedder is None
        
        # Get dtype and device from existing parameters
        existing_param = next(original_condition_embedder.parameters())
        target_dtype = existing_param.dtype
        target_device = existing_param.device
        
        # Create and move to correct dtype/device
        image_embedder = WanImageEmbedding(self.image_dim, self.inner_dim)
        image_embedder = image_embedder.to(dtype=target_dtype, device=target_device)
        
        # Ensure all parameters in image_embedder require gradients
        for param in image_embedder.parameters():
            param.requires_grad_(True)
        
        original_condition_embedder.image_embedder = image_embedder
        return transformer

    def mock_crossattention(self, transformer: WanTransformer3DModel) -> WanTransformer3DModel:
        """
        Mock T2V cross attention to I2V cross attention by adding image-specific parameters.
        """
        def convert_attention_layer(layer):
            """Convert a single T2V cross attention layer to I2V cross attention"""
            if hasattr(layer, 'attn2') and layer.attn2 is not None:
                # Get the original cross attention
                original_attn = layer.attn2
                
                # Get dtype and device from existing parameters
                existing_param = next(original_attn.parameters())
                target_dtype = existing_param.dtype
                target_device = existing_param.device
                
                # Add image-specific parameters
                dim = original_attn.dim if hasattr(original_attn, 'dim') else original_attn.to_q.in_features
                
                # Add k_img and v_img linear layers
                original_attn.k_img = nn.Linear(dim, dim).to(dtype=target_dtype, device=target_device)
                original_attn.v_img = nn.Linear(dim, dim).to(dtype=target_dtype, device=target_device)
                
                # Add norm_k_img (check if original has qk_norm)
                has_qk_norm = hasattr(original_attn, 'norm_k') and not isinstance(original_attn.norm_k, nn.Identity)
                if has_qk_norm:
                    eps = original_attn.norm_k.eps if hasattr(original_attn.norm_k, 'eps') else 1e-6
                    original_attn.norm_k_img = WanRMSNorm(dim, eps=eps).to(dtype=target_dtype, device=target_device)
                else:
                    original_attn.norm_k_img = nn.Identity()
                
                # Initialize new parameters to ensure no change in original behavior
                with torch.no_grad():
                    # Initialize k_img and v_img weights to zero so img_x starts as zero
                    nn.init.zeros_(original_attn.k_img.weight)
                    nn.init.zeros_(original_attn.k_img.bias)
                    nn.init.zeros_(original_attn.v_img.weight)
                    nn.init.zeros_(original_attn.v_img.bias)
                    
                    # Initialize norm_k_img to identity if it's RMSNorm
                    if hasattr(original_attn.norm_k_img, 'weight'):
                        nn.init.ones_(original_attn.norm_k_img.weight)
                
                # Ensure all new parameters require gradients
                original_attn.k_img.weight.requires_grad_(True)
                original_attn.k_img.bias.requires_grad_(True)
                original_attn.v_img.weight.requires_grad_(True)
                original_attn.v_img.bias.requires_grad_(True)
                
                if hasattr(original_attn.norm_k_img, 'weight'):
                    original_attn.norm_k_img.weight.requires_grad_(True)

                # Update the forward method to handle both text and image context
                def new_forward(self, x, context):
                    # Define T5_CONTEXT_TOKEN_NUMBER (you may need to adjust this value)
                    T5_CONTEXT_TOKEN_NUMBER = 512  # Adjust based on your configuration TODO: 
                    
                    # Check if context includes image tokens
                    if context.shape[1] > T5_CONTEXT_TOKEN_NUMBER:
                        # I2V mode: split context into image and text parts
                        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
                        context_img = context[:, :image_context_length]
                        context_text = context[:, image_context_length:]
                        
                        b, n, d = x.size(0), self.heads, self.inner_dim // self.heads
                        
                        # Compute query, key, value for text
                        q = self.norm_q(self.to_q(x)).view(b, -1, n, d)
                        k = self.norm_k(self.to_k(context_text)).view(b, -1, n, d)
                        v = self.to_v(context_text).view(b, -1, n, d)
                        
                        # Compute key, value for image
                        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
                        v_img = self.v_img(context_img).view(b, -1, n, d)
                        from finetrainers.models.wan.attention import flash_attention
                        # Compute attention for both text and image
                        x_text = flash_attention(q, k, v, k_lens=None)
                        x_img = flash_attention(q, k_img, v_img, k_lens=None)
                        
                        # Combine results
                        x = x_text.flatten(2) + x_img.flatten(2)
                        x = self.to_out[0](x)
                        x = self.to_out[1](x)
                        return x
                    else:
                        # T2V mode: use original behavior
                        b, n, d = x.size(0), self.heads, self.inner_dim // self.heads
                        
                        q = self.norm_q(self.to_q(x)).view(b, -1, n, d)
                        k = self.norm_k(self.to_k(context)).view(b, -1, n, d)
                        v = self.to_v(context).view(b, -1, n, d)
                        
                        from diffusers.models.attention_processor import flash_attention
                        x = flash_attention(q, k, v, k_lens=context_lens)
                        
                        x = x.flatten(2)
                        x = self.to_out[0](x)
                        x = self.to_out[1](x)
                        return x
                
                # Bind the new forward method
                import types
                original_attn.forward = types.MethodType(new_forward, original_attn)
        
        # Recursively apply to all transformer blocks
        def apply_to_blocks(module):
            for name, child in module.named_children():
                if hasattr(child, 'attn2'):    # 这里叫 attn2
                    convert_attention_layer(child)
                else:
                    apply_to_blocks(child)
        
        apply_to_blocks(transformer)
        return transformer