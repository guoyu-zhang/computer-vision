import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import logging
import numpy as np

# ------------------------------
# Setup Logger
# ------------------------------
logger = logging.getLogger("TrainingLogger")
logger.setLevel(logging.INFO)
# Console handler
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
# File handler
f_handler = logging.FileHandler('training_log_augments.txt', mode='a')
f_handler.setLevel(logging.INFO)
# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

logger.info("===== Starting Training Process =====")

# ------------------------------
# 0. Augmentation Class
# ------------------------------
from src.data.custom_dataset import SegmentationAugment

from src.models.enhanced_unet import EnhancedUNet

# ------------------------------
# 2. Dataset Classes
# ------------------------------
# For autoencoder pre-training (only images are needed)
class AutoencoderDataset(Dataset):
    def __init__(self, image_dir, transform=None, augmentation=None):
        self.image_dir = image_dir
        self.filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        image = Image.open(img_path).convert("RGB")
        # Apply augmentation if provided (pass same image twice for autoencoder)
        if self.augmentation is not None:
            image, _ = self.augmentation(image, image)
        if self.transform:
            image = self.transform(image)
        # For autoencoder, target is the input image itself
        return image

# For segmentation fine-tuning
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augmentation=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augmentation = augmentation

        self.image_filenames = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.mask_filenames = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(('.png', '.jpg'))
        ])
        assert len(self.image_filenames) == len(self.mask_filenames), "Image/mask count mismatch"

        self.img_transform = transform
        self.mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # Apply augmentation if provided
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)

        if self.img_transform:
            image = self.img_transform(image)

        mask_tensor = self.mask_transform(mask).squeeze(0)

        # Initialize final class mask with default value 4 (unknown)
        class_mask = torch.full_like(mask_tensor, fill_value=4, dtype=torch.long)
        is_background = mask_tensor == 0.0
        is_boundary = mask_tensor == 1.0
        is_animal = (~is_background) & (~is_boundary)

        class_mask[is_background] = 2   # Background
        class_mask[is_boundary] = 3     # Boundary

        if mask_name[0].isupper():
            class_mask[is_animal] = 0   # Cat
        else:
            class_mask[is_animal] = 1   # Dog

        img_id = os.path.splitext(img_name)[0]
        mask_id = os.path.splitext(mask_name)[0]
        return image, class_mask, img_id, mask_id

# ------------------------------
# 3. Dice Loss Implementation
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

# Combined Loss Function with ignore_index=4 for unknown regions
class CombinedLoss(nn.Module):
    def __init__(self, weights=None, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=3)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, logits, targets):
        ce_loss_val = self.ce_loss(logits, targets)
        dice_loss_val = self.dice_loss(logits, targets)
        return self.alpha * ce_loss_val + self.beta * dice_loss_val

# ------------------------------
# 4. Helper Function to Compute Class Weights
# ------------------------------
def compute_class_weights(dataset, num_classes=4):
    logger.info("Computing class weights...")
    class_count = torch.zeros(num_classes)
    for i in range(len(dataset)):
        _, mask, _, _ = dataset[i]
        for c in range(num_classes):
            class_count[c] += (mask == c).sum().item()
    total_pixels = class_count.sum()
    class_frequencies = class_count / total_pixels
    class_weights = 1.0 / (class_frequencies + 1e-5)
    class_weights = class_weights / class_weights.sum() * num_classes
    if num_classes == 5:
        class_weights[4] = 0.0
    logger.info(f"Class counts: {class_count}")
    logger.info(f"Class frequencies: {class_frequencies}")
    logger.info(f"Class weights: {class_weights}")
    return class_weights

# ------------------------------
# 5. Pre-training Autoencoder (with Augmentation)
# ------------------------------
def pretrain_autoencoder():
    logger.info("=== Starting Autoencoder Pre-training ===")
    image_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/train_resized/color"
    num_epochs = 7
    batch_size = 8
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    auto_patience = 3
    best_auto_loss = float('inf')
    patience_counter = 0

    inference_folder = "pretrain_inference_examples_augments_enhanced"
    if not os.path.exists(inference_folder):
        os.makedirs(inference_folder)
        logger.info(f"Created folder for inference examples: {inference_folder}")

    transform = transforms.Compose([transforms.ToTensor(),])
    augmentation = SegmentationAugment(apply_color_jitter=True)

    dataset = AutoencoderDataset(image_dir, transform=transform, augmentation=augmentation)
    logger.info(f"Number of images in autoencoder dataset: {len(dataset)}")
    sample_image = dataset[0]
    logger.info(f"Sample image shape from autoencoder dataset: {sample_image.size()}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate EnhancedUNet for autoencoder pretraining with reconstruction (out_channels=3)
    model = EnhancedUNet(in_channels=3, out_channels=3).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Starting training...")
        for batch_idx, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            # logger.info(f"Epoch {epoch+1} Batch {batch_idx+1}/{len(dataloader)} - Batch Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataset)
        logger.info(f"Epoch {epoch+1} Completed - Average Loss: {epoch_loss:.4f}")

        if epoch_loss < best_auto_loss:
            best_auto_loss = epoch_loss
            patience_counter = 0
            logger.info(f"Epoch {epoch+1} - New best loss: {best_auto_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Epoch {epoch+1} - No improvement. Patience counter: {patience_counter}/{auto_patience}")
            if patience_counter >= auto_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        sample_img = dataset[0]
        model.eval()
        with torch.no_grad():
            input_tensor = sample_img.unsqueeze(0).to(device)
            output_tensor = model(input_tensor)
        model.train()
        to_pil = transforms.ToPILImage()
        input_img_pil = to_pil(input_tensor.squeeze(0).cpu().clamp(0, 1))
        input_save_path = os.path.join(inference_folder, f"epoch_{epoch+1}_input.png")
        input_img_pil.save(input_save_path)
        logger.info(f"Saved input example for epoch {epoch+1} to {inference_folder}")
        output_img_pil = to_pil(output_tensor.squeeze(0).cpu().clamp(0, 1))
        output_save_path = os.path.join(inference_folder, f"epoch_{epoch+1}_output.png")
        output_img_pil.save(output_save_path)
        logger.info(f"Saved output example for epoch {epoch+1} to {inference_folder}")

    autoencoder_checkpoint = "autoencoder_pretrained_augment_enhanced.pth"
    torch.save(model.state_dict(), autoencoder_checkpoint)
    logger.info(f"Pretrained autoencoder saved to {autoencoder_checkpoint}")

# ------------------------------
# 6. Improved Segmentation Fine Tuning
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

def fine_tune_segmentation():
    logger.info("=== Starting Improved Segmentation Fine Tuning ===")
    image_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/train_resized/color"
    mask_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/train_resized/label"
    json_path = "/home/s1808795/CV_Assignment/Dataset_filtered/original_sizes.json"
    num_epochs = 1000
    batch_size = 8
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    num_classes = 4  # Cat, Dog, Background, Boundary, Unknown

    seg_patience = 10
    best_seg_loss = float('inf')
    seg_patience_counter = 0

    with open(json_path, 'r') as f:
        original_sizes = json.load(f)

    transform = transforms.Compose([transforms.ToTensor(),])
    augmentation = SegmentationAugment(apply_color_jitter=True)

    train_dataset = SegmentationDataset(image_dir, mask_dir, transform=transform, augmentation=augmentation)
    logger.info(f"Number of images in training dataset: {len(train_dataset)}")
    
    def segmentation_collate_fn(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        masks = [item[1] for item in batch]
        img_ids = [item[2] for item in batch]
        mask_ids = [item[3] for item in batch]
        return images, masks, img_ids, mask_ids

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=segmentation_collate_fn)
    
    # Validation Dataset & Dataloader
    val_image_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/val_resized/color"
    val_mask_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/val_resized/label"
    val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, transform=transform, augmentation=None)
    logger.info(f"Number of images in validation dataset: {len(val_dataset)}")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=segmentation_collate_fn)

    class_weights = compute_class_weights(train_dataset, num_classes=num_classes)
    class_weights = class_weights.to(device)
    
    criterion = CombinedLoss(
        weights=class_weights,
        alpha=0.5,  # CrossEntropyLoss weight
        beta=0.5    # DiceLoss weight
    )
    logger.info(f"Using combined loss with weights: CE={0.5}, Dice={0.5}")

    # Instantiate EnhancedUNet for segmentation fine tuning with out_channels=num_classes
    seg_model = EnhancedUNet(in_channels=3, out_channels=num_classes).to(device)

    # Load pre-trained autoencoder weights
    autoencoder_checkpoint = "autoencoder_pretrained_augment_enhanced.pth"
    pretrained_dict = torch.load(autoencoder_checkpoint, map_location=device)
    seg_model_dict = seg_model.state_dict()
    # Transfer encoder and bottleneck weights
    pretrained_encoder_dict = {k: v for k, v in pretrained_dict.items() if k.startswith("encoder") or k.startswith("bottleneck")}
    seg_model_dict.update(pretrained_encoder_dict)
    seg_model.load_state_dict(seg_model_dict)
    logger.info("Loaded pre-trained encoder and bottleneck weights into segmentation model.")

    # Freeze encoder layers initially
    for name, param in seg_model.named_parameters():
        if name.startswith("encoder") or name.startswith("bottleneck"):
            param.requires_grad = False
    logger.info("Encoder and bottleneck parameters have been frozen.")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, seg_model.parameters()), lr=learning_rate)

    train_losses = []
    class_accuracies = {i: [] for i in range(num_classes)}
    
    for epoch in range(num_epochs):
        seg_model.train()
        running_loss = 0.0
        class_correct = torch.zeros(num_classes).to(device)
        class_total = torch.zeros(num_classes).to(device)
        
        logger.info(f"Segmentation Epoch {epoch+1}/{num_epochs} - Starting training...")
        
        for batch_idx, (imgs, masks, img_ids, _) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            logits = seg_model(imgs)
            
            batch_loss = 0.0
            for i in range(imgs.size(0)):
                orig_size = original_sizes.get(img_ids[i], None)
                if orig_size is None:
                    raise ValueError(f"Original size for {img_ids[i]} not found in JSON")
                logit = logits[i].unsqueeze(0)
                logit_up = F.interpolate(logit, size=tuple(orig_size), mode='nearest')
                target_mask = masks[i].unsqueeze(0).to(device)
                sample_loss = criterion(logit_up, target_mask)
                batch_loss += sample_loss
                
                pred = torch.argmax(logit_up, dim=1)
                for c in range(num_classes):
                    class_mask = (target_mask == c)
                    class_total[c] += class_mask.sum().item()
                    if class_mask.sum() > 0:
                        class_correct[c] += ((pred == c) & class_mask).sum().item()
            
            batch_loss = batch_loss / imgs.size(0)
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item() * imgs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        for c in range(num_classes):
            if class_total[c] > 0:
                acc = 100.0 * class_correct[c] / class_total[c]
                class_accuracies[c].append(acc.item())
                logger.info(f"Class {c} accuracy: {acc:.2f}%")
            else:
                class_accuracies[c].append(0.0)
        logger.info(f"Epoch {epoch+1} Completed - Average Training Loss: {epoch_loss:.4f}")
        
        # Validation Loop
        seg_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, masks, img_ids, _ in val_dataloader:
                images = images.to(device)
                batch_val_loss = 0.0
                logits = seg_model(images)
                for i in range(images.size(0)):
                    orig_size = original_sizes.get(img_ids[i], None)
                    if orig_size is None:
                        raise ValueError(f"Original size for {img_ids[i]} not found in JSON")
                    logit = logits[i].unsqueeze(0)
                    logit_up = F.interpolate(logit, size=tuple(orig_size), mode='nearest')
                    target_mask = masks[i].unsqueeze(0).to(device)
                    sample_loss = criterion(logit_up, target_mask)
                    batch_val_loss += sample_loss
                total_val_loss += batch_val_loss / images.size(0)
        val_loss = total_val_loss / len(val_dataloader)
        logger.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")
        
        if epoch_loss < best_seg_loss:
            best_seg_loss = epoch_loss
            seg_patience_counter = 0
            logger.info(f"Epoch {epoch+1} - New best segmentation loss: {best_seg_loss:.4f}")
            torch.save(seg_model.state_dict(), "segmentation_best_augments_enhanced.pth")
        else:
            seg_patience_counter += 1
            logger.info(f"Epoch {epoch+1} - No improvement. Patience counter: {seg_patience_counter}/{seg_patience}")
            if seg_patience_counter >= seg_patience:
                logger.info(f"Early stopping triggered for segmentation at epoch {epoch+1}")
                break

        # Inference example on a training sample after each epoch
        sample_img, sample_mask, sample_img_id, sample_mask_id = train_dataset[0]
        seg_model.eval()
        with torch.no_grad():
            sample_tensor = sample_img.unsqueeze(0).to(device)
            logits = seg_model(sample_tensor)
            orig_size = original_sizes.get(sample_img_id, None)
            if orig_size is None:
                raise ValueError(f"Original size for {sample_img_id} not found in JSON")
            logits_up = F.interpolate(logits, size=tuple(orig_size), mode='nearest')
            pred_mask = torch.argmax(logits_up, dim=1).squeeze(0).cpu().numpy()
        seg_model.train()
        
        if epoch in [1,5,10,50,100,200,300]:
            color_pred = colorize_mask(pred_mask)
            to_pil = transforms.ToPILImage()
            input_img_pil = to_pil(sample_tensor.squeeze(0).cpu().clamp(0, 1))
            
            seg_inference_folder = "seg_inference_examples_augments_enhanced"
            if not os.path.exists(seg_inference_folder):
                os.makedirs(seg_inference_folder)
                
            input_save_path = os.path.join(seg_inference_folder, f"epoch_{epoch+1}_input.png")
            pred_save_path = os.path.join(seg_inference_folder, f"epoch_{epoch+1}_pred.png")
            input_img_pil.save(input_save_path)
            color_pred.save(pred_save_path)
            logger.info(f"Saved segmentation inference input and prediction for epoch {epoch+1}")

    seg_checkpoint = "segmentation_finetuned_augment_enhanced.pth"
    torch.save(seg_model.state_dict(), seg_checkpoint)
    logger.info(f"Final segmentation model saved to {seg_checkpoint}")
    
    metrics = {
        'loss': train_losses,
        'class_accuracies': class_accuracies
    }
    with open('training_metrics_augments_enhanced.json', 'w') as f:
        json.dump(metrics, f)
    logger.info("Training metrics saved to training_metrics_augments.json")

# ------------------------------
# 7. Main: Run Pre-training and Fine-tuning
# ------------------------------
if __name__ == '__main__':
    # Uncomment the following line to run autoencoder pre-training
    # pretrain_autoencoder()
    
    # Run segmentation fine-tuning
    fine_tune_segmentation()
