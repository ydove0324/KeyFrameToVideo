import os
from omegaconf import OmegaConf
import torch
import sys
sys.path.append(".")
from src.model.ibq_tokenizer import IBQ, tokenize_image, reconstruct_image, process_image


def calc_quant(latent_condition):
    """Undo the 2× spatial up-sample applied in `_quant_to_3d_latent` and
    return the *first* key-frame quant.

    The incoming `latent_condition` has shape `[B, C, F, H*2, W*2]`, where
    `F` is the number of (down-sampled) temporal positions.  The original
    code up-sampled the spatial dimensions using nearest-neighbour
    interpolation (factor 2).  Recover the original quants by:

    1. Selecting the very first frame along the temporal axis.
    2. Picking every second pixel (step 2) along **H** and **W** – this works
       because nearest-neighbour simply duplicated values.
    """

    # Unpack expected dimensions from the incoming tensor
    B, C, F, H2, W2 = latent_condition.shape  # H2 = H * 2, W2 = W * 2

    # 1. Take the first frame (index 0)  →  [B, C, H2, W2]
    first_frame = latent_condition[:, :, 0]

    # 2. Reverse the 2× nearest-neighbour upsample by striding every 2 pixels
    #    along the spatial axes.
    quant = first_frame[:, :, ::2, ::2].contiguous()  # [B, C, H, W]

    return quant

if __name__ == "__main__":
    tokenize_path = "/share/project/zhangfan/weights/Emu3.5-Tokenizer/IBQ-XL-f16c131k-FI"
    config_name = "fusimage_ibqgan_xl_131072_siglip.yaml"
    config = OmegaConf.load(os.path.join(tokenize_path, config_name))

    # Initialize model with config
    model = IBQ(**config.model.init_args)
    # print(f"Working with z of shape {tuple(model.z_shape)} = {model.z_shape[0] * model.z_shape[1] * model.z_shape[2] * model.z_shape[3]} dimensions.")

    # Load checkpoint
    ckpt_name = "fusionimage_256_XL_f16c131k.ckpt"
    ckpt = torch.load(os.path.join(tokenize_path, ckpt_name), weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to("cuda")
    model.to(dtype=torch.bfloat16)
    # latent_condition = torch.load("debug_tensors/ibq_cloth_t1000.pt").to(device="cuda", dtype=torch.bfloat16)
    latent_condition = torch.load("debug_tensors/train_condition_latents.pt").to(device="cuda", dtype=torch.bfloat16)
    # latent_condition_mask = torch.load("debug_tensors/latent_condition_mask_t1000.pt").to(device="cuda", dtype=torch.float32)
    quant = calc_quant(latent_condition)
    reconstructed_image = reconstruct_image(quant, model)
    reconstructed_image.save("reconstructed_image.png")
