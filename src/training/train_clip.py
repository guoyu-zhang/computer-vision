import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import numpy as np
import clip  # Import the CLIP model
import random
import torchvision.transforms.functional as TF

# ------------------------------
# Setup Logger
# ------------------------------
logger = logging.getLogger("TrainingLogger")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
# Changed log file name to include _augment
f_handler = logging.FileHandler('training_log_clip_augment.txt', mode='a')
f_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

logger.info("===== Starting CLIP-based Segmentation Process =====")

# ------------------------------
# 1. Augmentation Transform for Segmentation
# ------------------------------
from ..data.custom_dataset import SegmentationAugment

# ------------------------------
# 2. Dataset Classes
# ------------------------------
from ..data.custom_dataset import CustomDataset

# Modified ClipSegmentationDataset to integrate augmentation (if provided)
class ClipSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, clip_preprocess, transform=None):
        """
        image_dir: Directory with input images.
        mask_dir: Directory with corresponding mask images.
        clip_preprocess: Preprocessing transform for the CLIP model.
        transform: Augmentation transform that takes (image, mask) and returns (aug_image, aug_mask).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.clip_preprocess = clip_preprocess
        self.transform = transform

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

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply augmentation if provided (only for training)
        if self.transform:
            image, mask = self.transform(image, mask)

        # Preprocess image for CLIP model
        clip_input = self.clip_preprocess(image)
        mask_tensor = self.mask_transform(mask).squeeze(0)

        # Create a class mask based on mask tensor values
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
        
        # For visualization, use the (augmented) image converted to tensor
        original_img = transforms.ToTensor()(image)
            
        return clip_input, class_mask, img_id, mask_id, original_img

# ------------------------------
# 3. Dice Loss Implementation (unchanged)
# ------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        batch_size = targets.size(0)
        num_classes = logits.size(1)
        
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        dice_scores = []
        for cls in range(num_classes):
            pred_cls = probs[:, cls, :, :]
            target_cls = targets_one_hot[:, cls, :, :]
            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(1.0 - dice.mean())
            
        return torch.stack(dice_scores).mean()

# Combined Loss Function (unchanged)
class CombinedLoss(nn.Module):
    def __init__(self, weights=None, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, logits, targets):
        ce_loss_val = self.ce_loss(logits, targets)
        dice_loss_val = self.dice_loss(logits, targets)
        return self.alpha * ce_loss_val + self.beta * dice_loss_val

# ------------------------------
# 4. Helper function to compute class weights (unchanged)
# ------------------------------
def compute_class_weights(dataset, num_classes=4):
    logger.info("Computing class weights...")
    class_count = torch.zeros(num_classes)
    
    for i in range(len(dataset)):
        _, mask, _, _, _ = dataset[i]
        for c in range(num_classes):
            class_count[c] += (mask == c).sum().item()
    
    total_pixels = class_count.sum()
    class_frequencies = class_count / total_pixels
    class_weights = 1.0 / (class_frequencies + 1e-5)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    logger.info(f"Class counts: {class_count}")
    logger.info(f"Class frequencies: {class_frequencies}")
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights

# ------------------------------
# 5. Colorize mask function (unchanged)
# ------------------------------
def colorize_mask(mask):
    import numpy as np
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
# 6. New CLIP Encoder & Segmentation Model
# ------------------------------

from ..models.enhanced_unet import ResidualBlock, AttentionBlock, UpBlock
# --- Modified CLIP Encoder Wrapper for ModifiedResNet ---
class CLIPEncoderWrapper(nn.Module):
    def __init__(self, clip_model):
        super(CLIPEncoderWrapper, self).__init__()
        # In the ModifiedResNet used by CLIP, the stem consists of three conv layers followed by an avgpool.
        self.stem_conv1 = clip_model.visual.conv1    # Conv layer 1 (stride=2)
        self.stem_bn1 = clip_model.visual.bn1
        self.stem_relu1 = clip_model.visual.relu1      # Note: attribute is 'relu1'
        self.stem_conv2 = clip_model.visual.conv2      # Second conv, stride=1 typically
        self.stem_bn2 = clip_model.visual.bn2
        self.stem_relu2 = clip_model.visual.relu2
        self.stem_conv3 = clip_model.visual.conv3      # Third conv layer
        self.stem_bn3 = clip_model.visual.bn3
        self.stem_relu3 = clip_model.visual.relu3
        # Instead of a maxpool, ModifiedResNet uses an average pool in the stem.
        self.stem_pool = clip_model.visual.avgpool      # nn.AvgPool2d(2)
        
        # Residual layers as in the CLIP model.
        self.layer1 = clip_model.visual.layer1   # Expected output: [B, 256, H, W]
        self.layer2 = clip_model.visual.layer2   # Expected output: [B, 512, H/2, W/2]
        self.layer3 = clip_model.visual.layer3   # Expected output: [B, 1024, H/4, W/4]
        self.layer4 = clip_model.visual.layer4   # Expected output: [B, 2048, H/8, W/8]

    def forward(self, x):
        # Convert input to same dtype as the model weights.
        x = x.to(self.stem_conv1.weight.dtype)
        
        features = {}
        # Stem: three conv layers + avgpool.
        x = self.stem_relu1(self.stem_bn1(self.stem_conv1(x)))
        x = self.stem_relu2(self.stem_bn2(self.stem_conv2(x)))
        x = self.stem_relu3(self.stem_bn3(self.stem_conv3(x)))
        x = self.stem_pool(x)  # Reduces resolution
        features['stem'] = x   # e.g., [B, width, H/?, W/?]
        
        x = self.layer1(x)
        features['layer1'] = x
        x = self.layer2(x)
        features['layer2'] = x
        x = self.layer3(x)
        features['layer3'] = x
        x = self.layer4(x)
        features['layer4'] = x
        return features

# --- Modified CLIP-Based Segmentation Model using a UNet-style decoder ---
# Note: We add 1x1 conv layers to adjust the skip connection channels.
class CLIPSegmentationModel(nn.Module):
    def __init__(self, clip_model, out_channels=4):
        super(CLIPSegmentationModel, self).__init__()
        self.encoder = CLIPEncoderWrapper(clip_model)
        
        # Reduce channels from layer4 (2048) to a smaller bottleneck (512)
        self.conv_reduce = nn.Conv2d(2048, 512, kernel_size=1)
        # A bottleneck residual block operating on the reduced feature map.
        self.bottleneck_block = ResidualBlock(512, 512)
        
        # 1x1 convolutions to reduce skip connection feature dimensions.
        self.skip_conv_layer3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.skip_conv_layer2 = nn.Conv2d(512, 128, kernel_size=1)
        self.skip_conv_layer1 = nn.Conv2d(256, 64, kernel_size=1)
        self.skip_conv_stem   = nn.Conv2d(  64, 64, kernel_size=1)
        
        # UNet-style decoder using your existing UpBlock.
        # up4: combines bottleneck and layer3 skip.
        self.up4 = UpBlock(in_channels=512, out_channels=256, use_attention=True)
        # up3: combines upsampled features and layer2 skip.
        self.up3 = UpBlock(in_channels=256, out_channels=128, use_attention=True)
        # up2: combines features with layer1 skip.
        self.up2 = UpBlock(in_channels=128, out_channels=64, use_attention=True)
        # up1: combines features with stem skip.
        self.up1 = UpBlock(in_channels=64, out_channels=64, use_attention=True)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        feats = self.encoder(x)
        stem = feats['stem'].float()    
        layer1 = feats['layer1'].float()   
        layer2 = feats['layer2'].float()   
        layer3 = feats['layer3'].float()   
        layer4 = feats['layer4'].float()   
        
        # Reduce layer4 channels before applying the bottleneck.
        bottleneck = self.conv_reduce(layer4)  # [B, 512, 7,7]
        bottleneck = self.bottleneck_block(bottleneck)  # [B, 512, 7,7]
        
        # Apply 1x1 convs to adjust skip features.
        skip3 = self.skip_conv_layer3(layer3)   # [B, 256, 14,14]
        skip2 = self.skip_conv_layer2(layer2)     # [B, 128, 28,28]
        skip1 = self.skip_conv_layer1(layer1)     # [B, 64, 56,56]
        skip0 = self.skip_conv_stem(stem)           # [B, 64, 56,56]
        
        # Decoder path: progressively upsample and fuse with skip features.
        d4 = self.up4(bottleneck, skip3)  # Upsample from 7->14; output: [B,256,14,14]
        d3 = self.up3(d4, skip2)          # Upsample from 14->28; output: [B,128,28,28]
        d2 = self.up2(d3, skip1)          # Upsample from 28->56; output: [B,64,56,56]
        d1 = self.up1(d2, skip0)          # Upsample from 56->? (as per UpBlock) 
        out = self.final_conv(d1)         # Final segmentation logits
        return out

# 7. CLIP-based Segmentation Training with Validation
# ------------------------------
def train_clip_segmentation():
    logger.info("=== Starting CLIP-based Segmentation Training ===")
    # Changed training paths to include _augment
    train_image_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/train_resized/color"
    train_mask_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/train_resized/label"
    json_path = "/home/s1808795/CV_Assignment/Dataset_filtered/original_sizes.json"
    num_epochs = 1000
    batch_size = 8
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    num_classes = 4  # Cat, Dog, Background, Boundary

    # Early stopping parameters based on validation loss
    val_patience = 10
    best_val_loss = float('inf')
    val_patience_counter = 0

    with open(json_path, 'r') as f:
        original_sizes = json.load(f)

    # Load CLIP model using a ResNet backbone
    logger.info("Loading CLIP model (RN50)...")
    local_clip_dir = "/home/s1808795"  # Updated path
    # You can switch among "RN50", "RN101", "RN50x4", etc.
    clip_model, clip_preprocess = clip.load("RN50", device=device, download_root=local_clip_dir)
    
    # Create an instance of the augmentation transform for training
    augmentation_transform = SegmentationAugment(apply_color_jitter=True)
    
    # Training dataset (with augmentation)
    train_dataset = ClipSegmentationDataset(train_image_dir, train_mask_dir, clip_preprocess, transform=None)
    logger.info(f"Number of images in CLIP training dataset: {len(train_dataset)}")
    
    # Compute class weights for weighted loss using training data
    class_weights = compute_class_weights(train_dataset, num_classes=num_classes)
    class_weights = class_weights.to(device)
    
    criterion = CombinedLoss(
        weights=class_weights,
        alpha=0.5,  # Weight for CrossEntropyLoss
        beta=0.5    # Weight for DiceLoss
    )
    logger.info(f"Using combined loss with weights: CE={0.5}, Dice={0.5}")

    # Custom collate function remains the same
    def clip_segmentation_collate_fn(batch):
        clip_inputs = torch.stack([item[0] for item in batch], dim=0)
        masks = [item[1] for item in batch]
        img_ids = [item[2] for item in batch]
        mask_ids = [item[3] for item in batch]
        original_imgs = [item[4] for item in batch]
        return clip_inputs, masks, img_ids, mask_ids, original_imgs

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=clip_segmentation_collate_fn
    )

    # Create validation dataset and dataloader (without augmentation)
    val_image_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/val_resized/color"
    val_mask_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/val_resized/label"
    val_dataset = ClipSegmentationDataset(val_image_dir, val_mask_dir, clip_preprocess, transform=None)
    logger.info(f"Number of images in CLIP validation dataset: {len(val_dataset)}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,  # Use a smaller batch size if needed
        shuffle=False,
        collate_fn=clip_segmentation_collate_fn
    )

    # Initialize our new CLIPSegmentationModel (which uses the RN50 encoder and UNet-style decoder)
    seg_model = CLIPSegmentationModel(clip_model, out_channels=num_classes).to(device)
    
    # Freeze CLIP encoder parameters if desired (here we freeze the whole encoder)
    for param in seg_model.encoder.parameters():
        param.requires_grad = False
    logger.info("CLIP encoder parameters have been frozen.")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, seg_model.parameters()), lr=learning_rate)
    logger.info("Initialized optimizer with trainable parameters.")

    train_losses = []
    class_accuracies = {i: [] for i in range(num_classes)}
    
    # Changed inference folder name to include _augment
    seg_inference_folder = "clip_seg_inference_examples_augment_everything"
    if not os.path.exists(seg_inference_folder):
        os.makedirs(seg_inference_folder)
        logger.info(f"Created folder for inference examples: {seg_inference_folder}")
    
    for epoch in range(num_epochs):
        seg_model.train()
        running_loss = 0.0
        class_correct = torch.zeros(num_classes).to(device)
        class_total = torch.zeros(num_classes).to(device)
        
        logger.info(f"CLIP Segmentation Epoch {epoch+1}/{num_epochs} - Starting training...")
        
        for batch_idx, (clip_inputs, masks, img_ids, mask_ids, original_imgs) in enumerate(train_dataloader):
            clip_inputs = clip_inputs.to(device)
            optimizer.zero_grad()
            
            logits = seg_model(clip_inputs)
            
            batch_loss = 0.0
            for i in range(clip_inputs.size(0)):
                orig_size = original_sizes.get(img_ids[i], None)
                if orig_size is None:
                    raise ValueError(f"Original size for {img_ids[i]} not found in JSON")
                
                logit = logits[i].unsqueeze(0)  # [1, 4, H, W]
                logit_up = F.interpolate(
                    logit, 
                    size=tuple(orig_size), 
                    mode='nearest'
                )
                
                target_mask = masks[i].unsqueeze(0).to(device)  # [1, H, W]
                sample_loss = criterion(logit_up, target_mask)
                batch_loss += sample_loss
                
                pred = torch.argmax(logit_up, dim=1)  # [1, H, W]
                for c in range(num_classes):
                    class_mask = (target_mask == c)
                    class_total[c] += class_mask.sum().item()
                    if class_mask.sum() > 0:
                        class_correct[c] += ((pred == c) & class_mask).sum().item()
            
            batch_loss = batch_loss / clip_inputs.size(0)
            batch_loss.backward()
            optimizer.step()
            
            running_loss += batch_loss.item() * clip_inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        
        for c in range(num_classes):
            if class_total[c] > 0:
                acc = 100.0 * class_correct[c] / class_total[c]
                class_accuracies[c].append(acc)
                logger.info(f"Training - Class {c} accuracy: {acc:.2f}%")
            else:
                class_accuracies[c].append(0.0)
        
        logger.info(f"Epoch {epoch+1} Completed - Average Training Loss: {epoch_loss:.4f}")

        # ------------------------------
        # Validation Phase
        # ------------------------------
        seg_model.eval()
        val_running_loss = 0.0
        val_class_correct = torch.zeros(num_classes).to(device)
        val_class_total = torch.zeros(num_classes).to(device)
        with torch.no_grad():
            for batch_idx, (clip_inputs_val, masks_val, img_ids_val, mask_ids_val, original_imgs_val) in enumerate(val_dataloader):
                clip_inputs_val = clip_inputs_val.to(device)
                logits_val = seg_model(clip_inputs_val)
                batch_loss_val = 0.0
                for i in range(clip_inputs_val.size(0)):
                    # For validation, we use the mask size as the target size
                    orig_size = masks_val[i].shape  # (H, W)
                    logit_val = logits_val[i].unsqueeze(0)
                    logit_up_val = F.interpolate(logit_val, size=tuple(orig_size), mode='nearest')
                    target_mask_val = masks_val[i].unsqueeze(0).to(device)
                    sample_loss_val = criterion(logit_up_val, target_mask_val)
                    batch_loss_val += sample_loss_val
                    pred_val = torch.argmax(logit_up_val, dim=1)
                    for c in range(num_classes):
                        class_mask_val = (target_mask_val == c)
                        val_class_total[c] += class_mask_val.sum().item()
                        if class_mask_val.sum() > 0:
                            val_class_correct[c] += ((pred_val == c) & class_mask_val).sum().item()
                batch_loss_val = batch_loss_val / clip_inputs_val.size(0)
                val_running_loss += batch_loss_val.item() * clip_inputs_val.size(0)
        avg_val_loss = val_running_loss / len(val_dataset)
        logger.info(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")
        for c in range(num_classes):
            if val_class_total[c] > 0:
                val_acc = 100.0 * val_class_correct[c] / val_class_total[c]
                logger.info(f"Validation - Class {c} accuracy: {val_acc:.2f}%")
        
        # ------------------------------
        # Early Stopping Check based on Validation Loss
        # ------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            val_patience_counter = 0
            logger.info(f"Epoch {epoch+1} - New best validation loss: {best_val_loss:.4f}")
            torch.save(seg_model.state_dict(), "clip_segmentation_best_augment_everything.pth")
        else:
            val_patience_counter += 1
            logger.info(f"Epoch {epoch+1} - No improvement on validation. Patience counter: {val_patience_counter}/{val_patience}")
            if val_patience_counter >= val_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        if epoch in [0,1,2,3,5,10,30,60,100]:
            # ------------------------------
            # Inference Example (using a training sample)
            # ------------------------------
            clip_input, sample_mask, sample_img_id, sample_mask_id, original_img = train_dataset[0]
            seg_model.eval()
            with torch.no_grad():
                sample_tensor = clip_input.unsqueeze(0).to(device)
                logits = seg_model(sample_tensor)
                orig_size = original_sizes.get(sample_img_id, None)
                if orig_size is None:
                    raise ValueError(f"Original size for {sample_img_id} not found in JSON")
                logits_up = F.interpolate(logits, size=tuple(orig_size), mode='nearest')
                pred_mask = torch.argmax(logits_up, dim=1)
                pred_mask = pred_mask.squeeze(0).cpu().numpy()
            seg_model.train()
            
            # Colorize the predicted mask
            color_pred = colorize_mask(pred_mask)
            
            # Convert the input image tensor to a PIL image
            to_pil = transforms.ToPILImage()
            input_img_pil = to_pil(original_img)
            
            # Process and save the ground truth mask
            sample_mask_np = sample_mask.cpu().numpy()
            color_gt = colorize_mask(sample_mask_np)
            
            input_save_path = os.path.join(seg_inference_folder, f"epoch_{epoch+1}_input.png")
            pred_save_path = os.path.join(seg_inference_folder, f"epoch_{epoch+1}_pred.png")
            gt_save_path = os.path.join(seg_inference_folder, f"epoch_{epoch+1}_gt.png")
            input_img_pil.save(input_save_path)
            color_pred.save(pred_save_path)
            color_gt.save(gt_save_path)
            logger.info(f"Saved CLIP segmentation inference example for epoch {epoch+1} including ground truth")

    seg_checkpoint = "clip_segmentation_finetuned_augment_everything.pth"
    torch.save(seg_model.state_dict(), seg_checkpoint)
    logger.info(f"Final CLIP segmentation model saved to {seg_checkpoint}")
    
    metrics = {
        'loss': train_losses,
        'class_accuracies': class_accuracies
    }
    with open('clip_training_metrics_augment_everything.json', 'w') as f:
        json.dump(metrics, f)
    logger.info("Training metrics saved to clip_training_metrics_augment.json")

# ------------------------------
# 8. Main Function
# ------------------------------
if __name__ == '__main__':
    train_clip_segmentation()
