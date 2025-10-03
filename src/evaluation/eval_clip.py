import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import numpy as np
import clip  # CLIP model

# ------------------------------
# Setup Logger
# ------------------------------
logger = logging.getLogger("CLIPEvaluationLogger")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
f_handler = logging.FileHandler('clip_evaluation_log.txt', mode='w')
f_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

logger.info("===== Starting CLIP Segmentation Evaluation Process =====")

# ------------------------------
# 1. Define Model Building Blocks (same as in training code)
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
        """
        F_g: channels in gating signal (from decoder)
        F_l: channels in encoder feature map (skip connection)
        F_int: intermediate channel number (usually F_l//2)
        """
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
        # Handle size differences if any
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        if self.use_attention:
            skip = self.attention(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        x = self.res_block(x)
        return x

# ------------------------------
# 2. Re-define the CLIP Encoder Wrapper and the Segmentation Model 
# (from your training code architecture)
# ------------------------------
class CLIPEncoderWrapper(nn.Module):
    def __init__(self, clip_model):
        super(CLIPEncoderWrapper, self).__init__()
        # Use the stem and residual layers from the CLIP model (RN50)
        self.stem_conv1 = clip_model.visual.conv1
        self.stem_bn1 = clip_model.visual.bn1
        self.stem_relu1 = clip_model.visual.relu1
        self.stem_conv2 = clip_model.visual.conv2
        self.stem_bn2 = clip_model.visual.bn2
        self.stem_relu2 = clip_model.visual.relu2
        self.stem_conv3 = clip_model.visual.conv3
        self.stem_bn3 = clip_model.visual.bn3
        self.stem_relu3 = clip_model.visual.relu3
        self.stem_pool = clip_model.visual.avgpool
        
        self.layer1 = clip_model.visual.layer1   # Expected shape: [B, 256, H, W]
        self.layer2 = clip_model.visual.layer2   # Expected shape: [B, 512, H/2, W/2]
        self.layer3 = clip_model.visual.layer3   # Expected shape: [B, 1024, H/4, W/4]
        self.layer4 = clip_model.visual.layer4   # Expected shape: [B, 2048, H/8, W/8]

    def forward(self, x):
        x = x.to(self.stem_conv1.weight.dtype)
        features = {}
        x = self.stem_relu1(self.stem_bn1(self.stem_conv1(x)))
        x = self.stem_relu2(self.stem_bn2(self.stem_conv2(x)))
        x = self.stem_relu3(self.stem_bn3(self.stem_conv3(x)))
        x = self.stem_pool(x)
        features['stem'] = x
        x = self.layer1(x)
        features['layer1'] = x
        x = self.layer2(x)
        features['layer2'] = x
        x = self.layer3(x)
        features['layer3'] = x
        x = self.layer4(x)
        features['layer4'] = x
        return features

class CLIPSegmentationModel(nn.Module):
    def __init__(self, clip_model, out_channels=4):
        super(CLIPSegmentationModel, self).__init__()
        self.encoder = CLIPEncoderWrapper(clip_model)
        # Reduce channels from layer4 (2048) to a bottleneck (512)
        self.conv_reduce = nn.Conv2d(2048, 512, kernel_size=1)
        self.bottleneck_block = ResidualBlock(512, 512)
        # 1x1 convolutions for skip connections
        self.skip_conv_layer3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.skip_conv_layer2 = nn.Conv2d(512, 128, kernel_size=1)
        self.skip_conv_layer1 = nn.Conv2d(256, 64, kernel_size=1)
        self.skip_conv_stem   = nn.Conv2d(  64, 64, kernel_size=1)
        # UNet-style decoder with UpBlocks
        self.up4 = UpBlock(in_channels=512, out_channels=256, use_attention=True)
        self.up3 = UpBlock(in_channels=256, out_channels=128, use_attention=True)
        self.up2 = UpBlock(in_channels=128, out_channels=64, use_attention=True)
        self.up1 = UpBlock(in_channels=64, out_channels=64, use_attention=True)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        feats = self.encoder(x)
        stem = feats['stem'].float()
        layer1 = feats['layer1'].float()
        layer2 = feats['layer2'].float()
        layer3 = feats['layer3'].float()
        layer4 = feats['layer4'].float()
        
        bottleneck = self.conv_reduce(layer4)
        bottleneck = self.bottleneck_block(bottleneck)
        
        skip3 = self.skip_conv_layer3(layer3)
        skip2 = self.skip_conv_layer2(layer2)
        skip1 = self.skip_conv_layer1(layer1)
        skip0 = self.skip_conv_stem(stem)
        
        d4 = self.up4(bottleneck, skip3)
        d3 = self.up3(d4, skip2)
        d2 = self.up2(d3, skip1)
        d1 = self.up1(d2, skip0)
        out = self.final_conv(d1)
        return out

# ------------------------------
# 3. Define Test Dataset Class for Evaluation
# ------------------------------
class ClipSegmentationTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, clip_preprocess):
        """
        image_dir: Directory with test images
        mask_dir: Directory with corresponding mask images
        clip_preprocess: Preprocessing transform for CLIP model
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.clip_preprocess = clip_preprocess

        self.image_filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.mask_filenames = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(('.png', '.jpg'))
        ])
        assert len(self.image_filenames) == len(self.mask_filenames), "Image/mask count mismatch"
        self.mask_transform = transforms.ToTensor()
        self.img_tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Get the original size (PIL returns (width, height))
        original_size = image.size  
        
        # Process for CLIP model input
        clip_input = self.clip_preprocess(image)
        
        # For metrics and visualization, also get the original image tensor
        original_img = self.img_tensor_transform(image)
        
        # Process mask
        mask_tensor = self.mask_transform(mask).squeeze(0)
        # Create class mask based on value logic (2: background, 3: boundary)
        class_mask = torch.zeros_like(mask_tensor, dtype=torch.long)
        is_background = mask_tensor == 0.0
        is_boundary = mask_tensor == 1.0
        is_animal = (~is_background) & (~is_boundary)
        
        class_mask[is_background] = 2
        class_mask[is_boundary] = 3
        
        if mask_name[0].isupper():
            class_mask[is_animal] = 0  # Cat
        else:
            class_mask[is_animal] = 1  # Dog
        
        img_id = os.path.splitext(img_name)[0]
        mask_id = os.path.splitext(mask_name)[0]
        
        return clip_input, class_mask, original_size, img_id, mask_id, original_img

# Custom collate function for the test dataset
def clip_segmentation_test_collate_fn(batch):
    clip_inputs = torch.stack([item[0] for item in batch])
    class_masks = [item[1] for item in batch]  # keep as list
    original_sizes = [item[2] for item in batch]  # tuples (width, height)
    img_ids = [item[3] for item in batch]
    mask_ids = [item[4] for item in batch]
    original_imgs = [item[5] for item in batch]
    return clip_inputs, class_masks, original_sizes, img_ids, mask_ids, original_imgs

# ------------------------------
# 4. Colorize mask function (for visualization)
# ------------------------------
def colorize_mask(mask):
    color_map = {
        0: (255, 0, 0),    # Cat → red
        1: (0, 0, 255),    # Dog → blue
        2: (0, 255, 0),    # Background → green
        3: (255, 255, 0)   # Boundary → yellow
    }
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_mask[mask == label] = color
    return Image.fromarray(color_mask)

# ------------------------------
# 5. Evaluation Metrics Calculation Function
# ------------------------------
def evaluate_segmentation(model, test_loader, device, num_classes=4, save_examples=False):
    """
    Evaluate the segmentation model on test data and compute:
    - Per-class pixel accuracy
    - IoU and Dice scores per class
    """
    model.eval()
    
    intersections = torch.zeros(num_classes)
    unions = torch.zeros(num_classes)
    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)
    pixel_correct = torch.zeros(num_classes)
    pixel_total = torch.zeros(num_classes)
    
    if save_examples:
        examples_folder = "clip_seg_evaluation_examples"
        if not os.path.exists(examples_folder):
            os.makedirs(examples_folder)
        logger.info(f"Created folder for evaluation examples: {examples_folder}")
    
    with torch.no_grad():
        for batch_idx, (clip_inputs, masks, original_sizes, img_ids, mask_ids, original_imgs) in enumerate(test_loader):
            clip_inputs = clip_inputs.to(device)
            outputs = model(clip_inputs)  # logits with shape [B, out_channels, H, W]
            
            for i in range(clip_inputs.size(0)):
                target_mask = masks[i].to(device)
                # original_sizes[i] is (width, height); convert to (height, width)
                orig_w, orig_h = original_sizes[i]
                size_hw = (orig_h, orig_w)
                logits = outputs[i].unsqueeze(0)
                # Upsample logits to original image resolution (using nearest for segmentation)
                logits_resized = F.interpolate(logits, size=size_hw, mode='nearest')
                prediction = torch.argmax(logits_resized, dim=1).squeeze(0)
                
                # Compute metrics per class for this sample
                for c in range(num_classes):
                    gt_mask = (target_mask == c)
                    pred_mask = (prediction == c)
                    
                    intersection = (gt_mask & pred_mask).sum().float()
                    union = (gt_mask | pred_mask).sum().float()
                    tp = intersection
                    fp = pred_mask.sum().float() - tp
                    fn = gt_mask.sum().float() - tp
                    
                    intersections[c] += intersection.item()
                    unions[c] += union.item()
                    true_positives[c] += tp.item()
                    false_positives[c] += fp.item()
                    false_negatives[c] += fn.item()
                    
                    # Pixel accuracy update
                    correct = (gt_mask == pred_mask).sum().float()
                    total = gt_mask.numel()
                    pixel_correct[c] += correct.item()
                    pixel_total[c] += total
    
    # Compute per-class IoU and Dice scores
    epsilon = 1e-10
    iou_per_class = intersections / (unions + epsilon)
    dice_per_class = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives + epsilon)
    pixel_acc_per_class = pixel_correct / (pixel_total + epsilon)
    
    mean_iou = iou_per_class.mean().item()
    mean_dice = dice_per_class.mean().item()
    mean_pixel_acc = pixel_acc_per_class.mean().item()
    
    metrics = {
        'pixel_accuracy_per_class': pixel_acc_per_class.tolist(),
        'mean_pixel_accuracy': mean_pixel_acc,
        'iou_per_class': iou_per_class.tolist(),
        'mean_iou': mean_iou,
        'dice_per_class': dice_per_class.tolist(),
        'mean_dice': mean_dice
    }
    
    return metrics

# ------------------------------
# 6. Main Evaluation Function
# ------------------------------
def main():
    # Set paths (update these paths as needed)
    test_image_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/Test/color"
    test_mask_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/Test/label"
    model_path = "clip_segmentation_best_augment_everything.pth"  # Use the checkpoint saved from training
    num_classes = 4  # (Cat, Dog, Background, Boundary)
    batch_size = 4  # Adjust according to your hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load the CLIP model (using RN50, to match your training code)
    local_clip_dir = "/home/s1808795"  # Update if necessary
    logger.info("Loading CLIP model (RN50)...")
    clip_model, clip_preprocess = clip.load("RN50", device=device, download_root=local_clip_dir)
    
    # Prepare the test dataset and dataloader
    test_dataset = ClipSegmentationTestDataset(test_image_dir, test_mask_dir, clip_preprocess)
    logger.info(f"Number of test images: {len(test_dataset)}")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=clip_segmentation_test_collate_fn
    )
    
    # Initialize our CLIP segmentation model (must match the training architecture)
    logger.info("Initializing CLIP segmentation model...")
    seg_model = CLIPSegmentationModel(clip_model, out_channels=num_classes).to(device)
    
    # Load the trained model weights
    logger.info(f"Loading model weights from {model_path}...")
    seg_model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Start evaluation
    logger.info("Starting evaluation on test dataset...")
    metrics = evaluate_segmentation(seg_model, test_dataloader, device, num_classes, save_examples=True)
    
    # Log the evaluation metrics
    class_names = ["Cat", "Dog", "Background", "Boundary"]
    logger.info("\n--- CLIP Segmentation Evaluation Results ---")
    for i, cls_name in enumerate(class_names):
        logger.info(f"\nClass {i} ({cls_name}):")
        logger.info(f"  Pixel Accuracy: {metrics['pixel_accuracy_per_class'][i]:.4f}")
        logger.info(f"  IoU: {metrics['iou_per_class'][i]:.4f}")
        logger.info(f"  Dice/F1: {metrics['dice_per_class'][i]:.4f}")
    logger.info("\nMean Metrics:")
    logger.info(f"  Mean Pixel Accuracy: {metrics['mean_pixel_accuracy']:.4f}")
    logger.info(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    logger.info(f"  Mean Dice/F1: {metrics['mean_dice']:.4f}")
    
    # Optionally, save the metrics to a JSON file
    with open('clip_evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info("Evaluation metrics saved to clip_evaluation_metrics.json")

if __name__ == "__main__":
    main()
