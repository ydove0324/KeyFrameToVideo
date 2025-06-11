import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
from diffusers import FlowMatchEulerDiscreteScheduler
from finetrainers.models.wan.transformer_wan import WanTransformer3DModel,T2VModel2I2VModelConverter
from diffusers.utils import load_image
from transformers import UMT5EncoderModel, AutoTokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor
import PIL.Image
from typing import Optional, Union, List
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import numpy as np
import decord

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "/share/project/huangxu/Wan2.1-T2V-1.3B-diffusers"
transformer_path = "/share/project/huangxu/wan-t2v-debug-intern-video-clips/model_weights/007500"


# Load models
# transformer = WanTransformer3DModel.from_pretrained(transformer_path, subfolder="transformer", torch_dtype=torch.bfloat16)
# If you have custom transformer weights, load them like this:
transformer = WanTransformer3DModel.from_pretrained(transformer_path,subfolder="transformer", torch_dtype=torch.bfloat16)
converter = T2VModel2I2VModelConverter(transformer)
converter.convert()
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)

# Load text encoder and tokenizer
text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# Load image encoder (required for first/last frame conditioning)
image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.bfloat16)
image_processor = CLIPImageProcessor.from_pretrained(model_id, subfolder="image_processor")


scheduler = FlowMatchEulerDiscreteScheduler()

# Move models to GPU
device = "cuda"
transformer.to(device)
vae.to(device)
text_encoder.to(device)
image_encoder.to(device)
# pipe = WanImageToVideoPipeline.from_pretrained(pretrained_model_name_or_path="/share/project/huangxu/Wan2.1-T2V-1.3B-diffusers",transformer=transformer, torch_dtype=torch.bfloat16).to(device)

def load_video_as_tensor(video_path: str, height: int, width: int, max_frames: int = 17) -> torch.Tensor:
    """Loads a video and converts it to a tensor in the [-1, 1] range."""
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path, width=width, height=height)
    # 只读取前max_frames帧，如果视频帧数不足则读取所有帧
    num_frames_to_read = min(max_frames, len(vr))
    video = vr.get_batch(list(range(num_frames_to_read)))
    video = video.permute(0, 3, 1, 2).contiguous()
    video = video.float() / 127.5 - 1.0
    return video

def tensor_to_pil(tensor: torch.Tensor) -> PIL.Image.Image:
    """Converts a single image tensor from [-1, 1] to PIL Image."""
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    tensor = tensor.cpu().permute(1, 2, 0).numpy()
    image = PIL.Image.fromarray((tensor * 255).astype(np.uint8))
    return image

def normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
    latents = ((latents.float() - latents_mean) * latents_std).to(latents)
    return latents


def encode_text(prompt: str, tokenizer, text_encoder, max_length: int = 512):
    """Encode text prompt using T5 encoder"""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        text_embeddings = text_encoder(
            input_ids=text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )[0]
    
    return text_embeddings


def encode_first_last_frame(first_image: PIL.Image.Image, last_image: PIL.Image.Image, image_encoder, image_processor):
    """Encode first and last frame using CLIP vision encoder"""
    # Process both images separately to maintain batch dimension
    # first_image = torch.clamp((first_image + 1) / 2, 0, 1)
    # last_image = torch.clamp((last_image + 1) / 2, 0, 1)
    first_image_processed = image_processor(images=first_image,convert_to_rgb=False,do_rescale=False, return_tensors="pt").pixel_values
    last_image_processed = image_processor(images=last_image, convert_to_rgb=False, do_rescale=False, return_tensors="pt").pixel_values
    
    first_image_processed = first_image_processed.to(device=device, dtype=torch.bfloat16)
    last_image_processed = last_image_processed.to(device=device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        # Get embeddings for both images
        first_image_embeds = image_encoder(first_image_processed, output_hidden_states=True).hidden_states[-2]
        last_image_embeds = image_encoder(last_image_processed, output_hidden_states=True).hidden_states[-2]
        
        # Concatenate in sequence dimension: [B, 257*2, hidden_dim]
        image_embeds = torch.cat([first_image_embeds, last_image_embeds], dim=1)
    
    return image_embeds

def postprocess(video):
    video = video.squeeze(0)  # Remove batch dimension [C, F, H, W]
    video = video.permute(1, 0, 2, 3)  # [F, C, H, W]
    video = (video / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
    video = (video * 255).to(torch.uint8).cpu().numpy()
    
    pil_frames = []
    for frame in video:
        frame_pil = PIL.Image.fromarray(frame.transpose(1, 2, 0), 'RGB')
        pil_frames.append(frame_pil)
    
    return pil_frames

def manual_inference(
    prompt: str,
    first_image: PIL.Image.Image,
    last_image: PIL.Image.Image,
    height: int = 480,
    width: int = 832, 
    num_frames: int = 49,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    generator: Optional[torch.Generator] = None,
):
    """
    Manual inference function for first frame + last frame conditioning
    """
    # Encode text prompt
    print("Encoding text prompt...")
    encoder_hidden_states = encode_text(prompt, tokenizer, text_encoder)
    
    # Encode first and last frame
    print("Encoding first and last frame...")
    encoder_hidden_states_image = encode_first_last_frame(first_image, last_image, image_encoder, image_processor)
    
    # Process first and last frame for latent conditioning
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    # Convert both images to tensors
    first_tensor = transform(first_image).unsqueeze(0)  # [1, C, H, W]
    last_tensor = transform(last_image).unsqueeze(0)   # [1, C, H, W]
    
    # Create video tensor with first and last frame, zeros in between
    video_tensor = torch.zeros(1, num_frames, 3, height, width, dtype=torch.bfloat16, device=device)  
    video_tensor[:, 0] = first_tensor.to(device=device, dtype=torch.bfloat16)
    video_tensor[:, -1] = last_tensor.to(device=device, dtype=torch.bfloat16)
    
    # Encode to latents: [B, F, C, H, W] -> [B, C, F, H, W]
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
    
    with torch.no_grad():
        # Use mode() for deterministic encoding as in base_specification.py
        moments = vae._encode(video_tensor)
        latents = moments.to(dtype=torch.bfloat16)
        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)
        mu, logvar = torch.chunk(latents, 2, dim=1)
        mu = normalize_latents(mu, latents_mean, latents_std)
        logvar = normalize_latents(logvar, latents_mean, latents_std)
        latent_condition = torch.cat([mu, logvar], dim=1)

        posterior = DiagonalGaussianDistribution(latent_condition)
        latent_condition = posterior.sample(generator=generator)
    
    # Create conditioning mask - first and last frame only
    # Following the pattern from base_specification.py
    temporal_downsample = 4  # VAE temporal downsample factor
    
    # Create initial mask in original frame space
    mask = latent_condition.new_ones(latent_condition.shape[0], 1, num_frames, latent_condition.shape[3], latent_condition.shape[4])
    # Set first and last frame to 1, middle frames to 0 (use_last_frame=True case)
    mask[:, :, 1:-1] = 0
    
    # Process first frame mask with temporal downsample
    first_frame_mask = mask[:, :, :1]  # [B, 1, 1, H, W]
    first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=temporal_downsample)  # [B, 1, 4, H, W]
    
    # Concatenate with remaining frames
    mask = torch.cat([first_frame_mask, mask[:, :, 1:]], dim=2)  # [B, 1, 4+(F-1), H, W]
    
    # Reshape and transpose to match expected format
    mask = mask.view(latent_condition.shape[0], -1, temporal_downsample, latent_condition.shape[3], latent_condition.shape[4])
    latent_condition_mask = mask.transpose(1, 2)  # [B, 4, -1, H, W]
    
    # Calculate latent dimensions
    latent_height = height // 8
    latent_width = width // 8
    latent_frames = (num_frames - 1) // 4 + 1
    
    # Initialize random latents
    print("Initializing random latents...")
    latent_channels = 16  # Base latent channels
        
    latents_shape = (1, latent_channels, latent_frames, latent_height, latent_width)
    latents = torch.zeros_like(latent_condition).normal_(generator=generator)
    noise = latents.clone()
    
    # Set timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    print(f"Starting denoising loop with {num_inference_steps} steps...")
    
    # Denoising loop
    for i, t in enumerate(timesteps):
        # if i > 2:
            # break
        print(f"Step {i+1}/{num_inference_steps}, timestep: {t}")
        
        # Prepare input for transformer
        latent_model_input = latents
        
        # Concatenate conditioning latents and mask
        # Following the pattern from base_specification.py forward method
        latent_model_input = torch.cat([
            latent_model_input, 
            latent_condition_mask, 
            latent_condition
        ], dim=1)
        
        # Prepare transformer inputs
        # if i == 0:
            # latent_model_input = torch.load("debug_tensors/hidden_states_t1000.pt").to("cuda")
            # encoder_hidden_states_image = torch.load("debug_tensors/encoder_hidden_states_image_t1000.pt").to("cuda")
            # noise = torch.load("debug_tensors/noise_t1000.pt").to("cuda")
            # encoder_hidden_states = torch.load("debug_tensors/encoder_hidden_states_t1000.pt").to("cuda")
        transformer_inputs = {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_image": encoder_hidden_states_image,
            "timestep": t.unsqueeze(0).long(),
            "return_dict": False,
        }
        
        # Forward pass through transformer
        with torch.no_grad():
            pred = transformer(**transformer_inputs)[0]
            if i == 0:
                x_0 = noise - pred
                video = vae.decode(x_0).sample
                output_frames = postprocess(video)
                export_to_video(output_frames, "debug_output.mp4", fps=16)
        
        # Scheduler step
        latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    
    print("Decoding latents to video...")
    # Decode latents to video
    with torch.no_grad():
        video = vae.decode(latents).sample
    
    return postprocess(video)


# Example usage
if __name__ == "__main__":
    prompt = ""
    
    # Load first and last frame images
    video_path = "validate_video/3c841f88f857edf61a9b4b10ebcff3816c861835.mp4"
    video_tensor = load_video_as_tensor(video_path, height=480, width=832)
    first_frame = tensor_to_pil(video_tensor[0])
    last_frame = tensor_to_pil(video_tensor[16])
    # first_frame = video[0].resize((832, 480))
    # last_frame = video[16].resize((832, 480))
    
    print("Starting manual inference...")

    output_frames = manual_inference(
        prompt=prompt,
        first_image=first_frame,
        last_image=last_frame,
        height=480,
        width=832,
        num_frames=17,
        num_inference_steps=50,
        generator=torch.Generator(device=device).manual_seed(42)
    )
    # output_frames = pipe(prompt=prompt, image=first_frame, last_image=last_frame, height=480, width=832, num_frames=17, num_inference_steps=50, generator=torch.Generator(device=device).manual_seed(42)).frames[0]
    print("Exporting video...")
    export_to_video(output_frames, "first_last_frame_output.mp4", fps=16)
    print("Done! Video saved as 'first_last_frame_output.mp4'")

