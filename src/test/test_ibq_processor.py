class OriginalWanImageConditioningIBQLatentEncodeProcessor(ProcessorMixin):
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
        MAX_BATCH_SIZE = 8
        total_size = images.size(0)

        # Initialize lists to store results
        all_quants = []
        all_qlosses = []
        all_indices = []
        
        # Process in batches
        # print(f"Encoding {total_size} images in batches of {MAX_BATCH_SIZE}")
        import time
        
        for start_idx in range(0, total_size, MAX_BATCH_SIZE):
            t0 = time.time()
            end_idx = min(start_idx + MAX_BATCH_SIZE, total_size)
            batch_images = images[start_idx:end_idx]
            t1 = time.time()
            # Encode current batch
            quant, qloss, (_, _, indices) = ibq_model.encode(batch_images)
            t2 = time.time()
            # print(f"Batch {start_idx//MAX_BATCH_SIZE+1} encoded in {t2-t1} seconds, total time: {t2-t0}")
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
        # if image is not None:
        #     print(f"image is not None, image shape: {image.shape}")
        # else:
        #     print(f"image is None, video shape: {video.shape}")

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