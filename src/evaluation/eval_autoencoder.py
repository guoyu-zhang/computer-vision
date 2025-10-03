#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import logging

# ------------------------------
# Setup Logger
# ------------------------------
logger = logging.getLogger("InferenceLogger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ------------------------------
# Enhanced UNet Architecture with Residual Blocks and Attention
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        self.res_block = ResidualBlock(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Adjust size if needed
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        if self.use_attention:
            skip = self.attention(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        x = self.res_block(x)
        return x

class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, features=[64, 128, 256, 512]):
        """
        Note: Your training script used 5 output channels. The conversion function later maps 
        predictions to 4 classes. Adjust the out_channels if needed.
        """
        super(EnhancedUNet, self).__init__()
        # Encoder
        self.encoder1 = ResidualBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ResidualBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck = ResidualBlock(features[3], features[3]*2)
        # Decoder
        self.up4 = UpBlock(in_channels=features[3]*2, out_channels=features[3], use_attention=True)
        self.up3 = UpBlock(in_channels=features[3], out_channels=features[2], use_attention=True)
        self.up2 = UpBlock(in_channels=features[2], out_channels=features[1], use_attention=True)
        self.up1 = UpBlock(in_channels=features[1], out_channels=features[0], use_attention=True)
        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        # Bottleneck
        b = self.bottleneck(p4)
        # Decoder with skip connections
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = self.final_conv(d1)
        return out

# ------------------------------
# Helper Functions: Convert and Colorize Masks
# ------------------------------
def convert_mask(mask_pil, mask_filename):
    """
    Convert a PIL image mask to a numpy array of class labels.
    Follows training logic:
      - Pixels with value 0.0 (background) → class 2
      - Pixels with value 1.0 (boundary) → class 3
      - For animal pixels, if mask filename starts with uppercase → cat (class 0), else dog (class 1)
    """
    mask_tensor = transforms.ToTensor()(mask_pil).squeeze(0)
    class_mask = torch.zeros_like(mask_tensor, dtype=torch.long)
    is_background = mask_tensor == 0.0
    is_boundary = mask_tensor == 1.0
    is_animal = (~is_background) & (~is_boundary)
    class_mask[is_background] = 2
    class_mask[is_boundary] = 3
    if mask_filename[0].isupper():
        class_mask[is_animal] = 0  # Cat
    else:
        class_mask[is_animal] = 1  # Dog
    return class_mask.cpu().numpy()

def colorize_mask(mask):
    """
    Convert a 2D numpy array of class labels to a color image.
    Colors:
      0: Cat (Red)
      1: Dog (Blue)
      2: Background (Green)
      3: Boundary (Yellow)
    """
    color_map = {
        0: (255, 0, 0),    # Cat → red
        1: (0, 255, 0),    # Dog → blue
        2: (0, 0, 255),    # Background → green
        3: (255, 255, 0)   # Boundary → yellow
    }
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        color_mask[mask == cls] = color
    return Image.fromarray(color_mask)

# ------------------------------
# Generate Inference Examples for Specific Images
# ------------------------------
def generate_specific_inference_examples(model, device, image_dir, mask_dir, output_dir, inference_images):
    """
    Runs inference on a given list of image filenames.
    - Each input image is resized to 256x256 for the model.
    - The predicted mask is then upsampled back to the original input image size.
    - Saves the original image, the colorized predicted mask, and the colorized ground truth mask.
    Assumes the corresponding mask is named based on the image filename. If not found,
    the code attempts to load the mask with a .png extension.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Transformation for input images: resize to (256,256) and convert to tensor.
    transform_input = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    model.eval()
    with torch.no_grad():
        for image_filename in inference_images:
            img_path = os.path.join(image_dir, image_filename)
            mask_path = os.path.join(mask_dir, image_filename)
            
            # Check if mask exists; if not, try replacing extension with .png
            if not os.path.exists(mask_path):
                base_filename = os.path.splitext(image_filename)[0]
                alternate_mask_path = os.path.join(mask_dir, base_filename + ".png")
                if os.path.exists(alternate_mask_path):
                    mask_path = alternate_mask_path
                    logger.info(f"Using alternate mask file: {mask_path}")
                else:
                    logger.error(f"Mask file for {image_filename} not found!")
                    continue
            
            if not os.path.exists(img_path):
                logger.error(f"Image file {img_path} not found!")
                continue

            # Load the original input image and record its size
            original_image = Image.open(img_path).convert("RGB")
            orig_width, orig_height = original_image.size
            target_size = (orig_height, orig_width)  # Note: PIL uses (height, width)

            # Preprocess the image: resize to 256x256 and convert to tensor
            image_tensor = transform_input(original_image).unsqueeze(0).to(device)
            
            # Run inference on resized input image
            logits = model(image_tensor)  # Expected output shape: [1, out_channels, 256, 256]
            # Upsample prediction to the original input image size using nearest neighbor interpolation
            logits_up = F.interpolate(logits, size=target_size, mode='nearest')
            pred_mask = torch.argmax(logits_up, dim=1).squeeze(0).cpu().numpy()
            color_pred = colorize_mask(pred_mask)
            
            # Process and colorize the ground truth mask (kept at its original size)
            gt_mask_pil = Image.open(mask_path)
            gt_mask = convert_mask(gt_mask_pil, image_filename)
            color_gt = colorize_mask(gt_mask)
            
            # Save original image, predicted mask, and ground truth mask
            base_filename = os.path.splitext(image_filename)[0]
            input_save_path = os.path.join(output_dir, f"{base_filename}_input.png")
            pred_save_path = os.path.join(output_dir, f"{base_filename}_pred.png")
            gt_save_path = os.path.join(output_dir, f"{base_filename}_gt.png")
            
            original_image.save(input_save_path)
            color_pred.save(pred_save_path)
            color_gt.save(gt_save_path)
            
            logger.info(f"Saved inference example for {image_filename}:")
            logger.info(f"   Original Image: {input_save_path}")
            logger.info(f"   Predicted Mask (resized to input image size): {pred_save_path}")
            logger.info(f"   Ground Truth Mask: {gt_save_path}")

# ------------------------------
# Main Function
# ------------------------------
def main():
    # Directories and checkpoint path (adjust these as needed)
    test_image_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/Test/color"
    test_mask_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/Test/label"
    output_dir = "inference_examples_specific"
    checkpoint_path = "/home/s1808795/CV_Assignment/task2final.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize the Enhanced UNet with 5 output channels (as used in your training)
    model = EnhancedUNet(in_channels=3, out_channels=4).to(device)
    
    # Optionally freeze encoder parameters if desired
    for name, param in model.named_parameters():
        if name.startswith("encoder") or name.startswith("bottleneck"):
            param.requires_grad = False

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info(f"Loaded model weights from {checkpoint_path}")
    else:
        logger.error(f"Checkpoint file {checkpoint_path} not found.")
        return

    # List of specific images for inference
    inference_images = [
        "basset_hound_198.jpg",
        "Bombay_99.jpg",
        "wheaten_terrier_96.jpg",
        "wheaten_terrier_199.jpg"
    ]
    
    # Generate inference examples only for the specified images
    generate_specific_inference_examples(model, device, test_image_dir, test_mask_dir, output_dir, inference_images)

if __name__ == '__main__':
    main()
