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


def test_noise_robustness(quant, model, noise_variances=None, output_dir="./noise_test_results"):
    """
    Test robustness of IBQ tokenizer against Gaussian noise
    Args:
        quant: Quantized representation from IBQ model
        model: IBQ model for decoding
        noise_variances: List of noise variances to test (default: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5])
        output_dir: Directory to save reconstructed images
    Returns:
        results: Dictionary containing reconstructed images and their noise levels
    """
    import os
    from datetime import datetime
    
    # Default noise variances if not provided
    if noise_variances is None:
        noise_variances = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Testing noise robustness with variances: {noise_variances}")
    
    for i, variance in enumerate(noise_variances):
        print(f"Processing variance {variance}...")
        
        # Add Gaussian noise to quantized representation
        if variance == 0.0:
            # No noise case
            noisy_quant = quant.clone()
        else:
            # Add Gaussian noise with specified variance
            noise = torch.randn_like(quant) * torch.sqrt(torch.tensor(variance))
            noisy_quant = quant + noise
        
        # Reconstruct image from noisy quantized representation
        reconstructed_image = reconstruct_image(noisy_quant, model)
        
        # Save the reconstructed image
        filename = f"noise_variance_{variance:.3f}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        reconstructed_image.save(filepath)
        
        # Store results
        results[variance] = {
            'image': reconstructed_image,
            'filepath': filepath,
            'noise_level': variance
        }
        
        print(f"Saved: {filepath}")
    
    # Create a summary file
    summary_file = os.path.join(output_dir, f"noise_test_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("IBQ Tokenizer Noise Robustness Test Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test timestamp: {timestamp}\n")
        f.write(f"Number of noise levels tested: {len(noise_variances)}\n")
        f.write(f"Noise variances: {noise_variances}\n\n")
        f.write("Generated files:\n")
        for variance in noise_variances:
            f.write(f"- Variance {variance:.3f}: {results[variance]['filepath']}\n")
    
    print(f"\nNoise robustness test completed!")
    print(f"Results saved in: {output_dir}")
    print(f"Summary file: {summary_file}")
    
    return results


def visualize_noise_comparison(results, output_path="./noise_comparison.png"):
    """
    Create a visualization comparing all noise levels side by side
    Args:
        results: Results from test_noise_robustness function
        output_path: Path to save the comparison image
    """
    import matplotlib.pyplot as plt
    
    # Create subplot grid
    n_images = len(results)
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each image
    for i, (variance, result) in enumerate(results.items()):
        row = i // cols
        col = i % cols
        
        ax = axes[row, col]
        ax.imshow(result['image'])
        ax.set_title(f'Variance: {variance:.3f}')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Noise comparison visualization saved: {output_path}")