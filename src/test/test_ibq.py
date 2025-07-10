import os
from omegaconf import OmegaConf
import torch
import sys
sys.path.append(".")
from src.model.ibq_tokenizer import IBQ, tokenize_image, reconstruct_image, process_image


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

    tokenizer = model
    import decord
    video_path = "validate_results/step_012000/pexel_part2_1_294123638bc0b4bd57e6d64951bebec51ae99cf9_generated.mp4"
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    first_frame = vr.get_batch([0])  # Shape: (1, H, W, 3)
    first_frame = first_frame.permute(0, 3, 1, 2)  # Shape: (1, 3, H, W)
    first_frame = torch.nn.functional.interpolate(first_frame, size=(480, 832), mode='bilinear', align_corners=False)  # Resize
    first_frame = first_frame.to("cuda")
    print(first_frame.shape)
    quant, qloss, indices = process_image(first_frame, model)
    print(quant.shape)
    reconstructed_image = reconstruct_image(quant, model)
    reconstructed_image.save("reconstructed_image.png")