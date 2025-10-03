import os
import csv
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..models.enhanced_unet import EnhancedUNet
from ..data.robustness_custom_dataset import RobustnessCustomDataset

# Set target image size
TARGET_WIDTH = 256
TARGET_HEIGHT = 256
NUM_CLASSES = 4
IGNORED_CLASS = 3
CLASS_NAMES = ["Cat", "Dog", "Background", "Boundary"]
BATCH_SIZE = 32
BEST_MODEL_PATH = "/Users/bin/Desktop/CV_Assignment/Model/best_enhanced_unet_500_epochs_aug.pth"
base_dir = "/home/s2103701/Dataset_filtered"
BASE_DIR = "/Users/bin/Desktop/CV_Assignment/Dataset_filtered"
ORIGINAL_SIZES_PATH = os.path.join(BASE_DIR, "original_sizes_test.json")
SAVE_DIR = "Robustness_Results"

# === Custom collate (image, mask, img_name, mask_name)
def custom_collate_fn(batch):
    images, masks, img_names, mask_names = zip(*batch)
    images = torch.stack(images)
    return images, list(masks), list(img_names), list(mask_names)

# Compute Dice score between prediction and ground truth
def dice_score(preds, targets):
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1)

    preds_flat = preds.contiguous().view(preds.size(0), -1)
    targets_flat = targets.contiguous().view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2. * intersection) / (union)
    return dice.mean().item()

# Evaluate model on one test set (perturbation level)
def evaluate_model(model, dataloader, device, original_sizes):
    model.eval()
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks, img_names, _ in tqdm(dataloader):
            images = images.to(device)
            masks = [m.to(device).long() for m in masks]  # each mask: (H, W)

            logits = model(images)  # (B, C, H, W)

            for i in range(len(images)):
                name = img_names[i]
                orig_h, orig_w = original_sizes[name]  # restore original size

                # Resize prediction to original size
                logit = logits[i].unsqueeze(0)
                resized_logit = F.interpolate(logit, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                pred_mask = torch.argmax(resized_logit.squeeze(0), dim=0).long()  # (H, W)

                # Resize ground truth mask
                gt_mask = F.interpolate(masks[i].unsqueeze(0).unsqueeze(0).float(),
                                        size=(orig_h, orig_w),
                                        mode="nearest").squeeze().long()

                # Ignore boundary class
                valid_mask = gt_mask != IGNORED_CLASS
                filtered_pred = pred_mask[valid_mask]
                filtered_target = gt_mask[valid_mask]

                if filtered_pred.numel() == 0:
                    continue

                score = dice_score(filtered_pred.unsqueeze(0), filtered_target.unsqueeze(0))
                total_dice += score
                num_batches += 1

    return total_dice / num_batches if num_batches > 0 else 0.0

# Loop over all perturbations and levels
def run_robustness_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(ORIGINAL_SIZES_PATH, "r") as f:
        original_sizes = json.load(f)

    # Load best model
    model = EnhancedUNet(in_channels=3, out_channels=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    transform = None
    results_csv = os.path.join(SAVE_DIR, "mean_dice_scores.csv")

    perturbations = [
        'gaussian_noise', 'gaussian_blur', 'contrast_increase', 'contrast_decrease',
        'brightness_increase', 'brightness_decrease', 'occlusion', 'salt_pepper'
    ]

    # Save Dice scores to a CSV file
    with open(results_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Perturbation", "Level", "Mean Dice Score"])

        for perturb in perturbations:
            dice_scores = []
            for level in range(10):
                dataset = RobustnessCustomDataset(
                    image_dir=os.path.join(BASE_DIR, "test_resized", "color"),
                    mask_dir=os.path.join(BASE_DIR, "test_resized", "label"),
                    transform=transform,
                    perturbation=perturb,
                    level=level
                )
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

                # Run evaluation
                mean_dice = evaluate_model(model, dataloader, device, original_sizes)
                dice_scores.append(mean_dice)
                print(f"{perturb} - Level {level}: Dice = {mean_dice:.4f}")

                writer.writerow([perturb, level, mean_dice])

if __name__ == '__main__':
    run_robustness_evaluation()