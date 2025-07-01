import torch
from PIL import Image
from torchvision import transforms
import numpy as np

def process_image(image, model):
    """
    Process and encode image using IBQ tokenizer
    Args:
        image: Input image tensor
    Returns:
        quant: Quantized representation
        qloss: Quantization loss
        indices: Token indices
    """
    # Normalize image to [-1, 1] range
    image = image / 127.5 - 1.0
    
    # Encode image with no gradient computation
    with torch.no_grad():
        quant, qloss, (_, _, indices) = model.encode(image)
    
    return quant, qloss, indices


# Example usage function
def tokenize_image(image_path_or_tensor, model):
    """
    Tokenize an image using the IBQ model
    Args:
        image_path_or_tensor: Either a path to image file or image tensor
    Returns:
        tokens: Encoded image tokens
    """
    if isinstance(image_path_or_tensor, str):
        # If it's a path, load the image (you may need to implement image loading)
            # For now, assuming image tensor is passed directly
        image = Image.open(image_path_or_tensor)
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)  # Add batch dimension
        if image.shape[1] == 4:  # Handle RGBA images
            image = image[:, :3, :, :]  # Keep only RGB channels
    else:
        image = image_path_or_tensor
    
    # Process the image
    quant, qloss, indices = process_image(image, model)
    
    return {
        'quantized': quant,
        'loss': qloss,
        'tokens': indices
    }
def reconstruct_image(quant, model):
    """
    Reconstruct an image from quantized representation
    Args:
        quant: Quantized representation
    Returns:
        reconstructed_image: Reconstructed image
    """
    dec = model.decode(quant)
    image = torch.clamp(dec + 1.0, 0, 2) * 127.5
    image = image.squeeze(0)                # Remove batch dimension: (c, h, w)
    image = image.detach().cpu().float().numpy()    # Shape: (c, h, w)
    image = image.transpose(1, 2, 0)        # Change to HWC format: (h, w, c)
    image = Image.fromarray(image.astype(np.uint8))
    return image