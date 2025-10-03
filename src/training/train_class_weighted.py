import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from ..data.custom_dataset import CustomDataset
from ..models.enhanced_unet import EnhancedUNet

# Training script for Enhanced UNet
if __name__ == "__main__":
    base_dir = "/home/s2103701/Dataset_filtered"

    # ====== Load training and validation datasets ======
    train_dataset = CustomDataset(
        image_dir=os.path.join(base_dir, "train_randaug", "color"),
        mask_dir=os.path.join(base_dir, "train_randaug", "label"),
    )

    # Custom collate function to clean up names and stack images
    def custom_collate_fn(batch):
        images, masks, img_names, mask_names = zip(*batch)
        images = torch.stack(images)
        clean_img_names = [re.sub(r'(_aug_\d+)$', '', name) for name in img_names]
        clean_mask_names = [re.sub(r'(_aug_\d+)$', '', name) for name in mask_names]
        return images, list(masks), clean_img_names, clean_mask_names

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    val_dataset = CustomDataset(
        image_dir=os.path.join(base_dir, "val_resized", "color"),
        mask_dir=os.path.join(base_dir, "val_resized", "label"),
    )
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    # ====== Initialize model (with multi-GPU support if available) ======
    model = EnhancedUNet(in_channels=3, out_channels=4)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ====== Load original image sizes for resizing output masks ======
    with open(os.path.join(base_dir, "original_sizes.json"), "r") as f:
        original_sizes = json.load(f)

    # Resize output logits back to original image dimensions
    def resize_multiclass_logits(logits, img_names, original_sizes_dict):
        resized_logits = []
        for i in range(len(logits)):
            name = img_names[i]
            orig_h, orig_w = original_sizes_dict[name]
            resized = F.interpolate(
                logits[i].unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )
            resized_logits.append(resized.squeeze(0))  # shape: (4, H, W)
        return resized_logits

    # ====== Compute class weights for loss balancing ======
    print("Computing class weights...")
    mask_dir = os.path.join(base_dir, "train_randaug", "label")
    num_classes = 4
    class_counts = torch.zeros(num_classes)

    for filename in tqdm(sorted(os.listdir(mask_dir))):
        if filename.endswith(".png"):
            mask = Image.open(os.path.join(mask_dir, filename)).convert("L")
            mask_np = np.array(mask).astype(np.float32) / 255.0

            mapped = np.zeros_like(mask_np, dtype=np.int64)
            is_background = (mask_np == 0.0)
            is_boundary = (mask_np == 1.0)
            is_catdog = (~is_background) & (~is_boundary)

            if filename[0].isupper():
                mapped[is_catdog] = 0  # Cat
            else:
                mapped[is_catdog] = 1  # Dog
            mapped[is_background] = 2  # Background
            mapped[is_boundary] = 3    # Boundary

            for cls in range(num_classes):
                class_counts[cls] += (mapped == cls).sum()

    print("Class pixel counts:", class_counts.tolist())

    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum()
    print("Class weights:", weights.tolist())

    # ====== Training configuration ======
    NUM_EPOCHS = 500
    PRINT_INTERVAL = 10
    BEST_MODEL_PATH = "/home/s2103701/Model/best_enhanced_unet_500_epochs_aug_class_weighted.pth"
    best_val_loss = float("inf")
    patience = 10
    early_stop_counter = 0

    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ====== Training loop ======
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for images, masks, img_names, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
            images = images.to(device)
            masks = [m.to(device) for m in masks]
            optimizer.zero_grad()

            logits = model(images)
            resized_logits = resize_multiclass_logits(logits, img_names, original_sizes)

            total_loss = 0
            for pred, gt in zip(resized_logits, masks):
                pred = pred.unsqueeze(0)
                gt = gt.unsqueeze(0)
                loss = loss_fn(pred, gt)
                total_loss += loss

            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # ====== Validation ======
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, masks, img_names, _ in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                images = images.to(device)
                masks = [m.to(device) for m in masks]

                logits = model(images)
                resized_logits = resize_multiclass_logits(logits, img_names, original_sizes)

                for pred, gt in zip(resized_logits, masks):
                    pred = pred.unsqueeze(0)
                    gt = gt.unsqueeze(0)
                    val_loss = loss_fn(pred, gt)
                    total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ====== Early Stopping ======
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model at epoch {epoch+1} (val loss: {avg_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break

    print("Training complete. Best validation loss:", best_val_loss)