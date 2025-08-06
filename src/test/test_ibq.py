import os
from omegaconf import OmegaConf
import torch
import sys
sys.path.append(".")
from src.model.ibq_tokenizer import IBQ, tokenize_image, reconstruct_image, process_image
from src.model.ibq_tokenizer.utils import test_noise_robustness
import numpy as np

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

    # Load state dict
    model.load_state_dict(ckpt["state_dict"])
    model.to("cuda")
    print("All keys matched successfully>")

    # Set model to eval mode
    model.eval()
    
    # 读取npy文件
    npy_file_path = "/share/project/zhangfan/codes/Emu3.5/emu3p5_stage2_55k_topk1024_temp1.0_p1.0_cfg5.0_v4/output_token/rank0_of_world_size4_26.npy"
    ibq_indices = torch.from_numpy(np.load(npy_file_path)).to("cuda")
    print(f"Loaded ibq_indices with shape: {ibq_indices.shape}")
    # print(f"Data type: {ibq_indices.dtype}")
    embedding = model.get_embedding(indices=ibq_indices)
    print(embedding.shape)
    