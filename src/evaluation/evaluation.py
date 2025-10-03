import os
import cv2
import shutil
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import DiceScore
import os, json
from tqdm import tqdm
from custom_dataset import CustomDataset
from enhanced_unet import EnhancedUNet

# Set target dimensions
TARGET_WIDTH = 256
TARGET_HEIGHT = 256
target_size = (TARGET_WIDTH, TARGET_HEIGHT)

MODEL_PATH = '/Users/bin/Desktop/CV_Assignment/Model/best_enhanced_unet_500_epochs_aug.pth'

# -----------------------------------------
# Resizing Test Set Functions
# -----------------------------------------

def resize_and_save(src_path, dst_path, target_size):
    """
    Resize an image and save it (don't record size here anymore).
    """
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Error reading:", src_path)
        return
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(dst_path, resized)

def process_data(color_source, label_source, resized_color_dest, resized_label_dest, original_sizes_dict):
    """
    Processes a dataset split:
    - Resizes color images and saves them.
    - Copies label masks unchanged, and stores their original sizes with clean keys.
    """
    # Resize color images
    for filename in sorted(os.listdir(color_source)):
        if filename.lower().endswith(".jpg"):
            src_path = os.path.join(color_source, filename)
            dst_path = os.path.join(resized_color_dest, filename)
            resize_and_save(src_path, dst_path, target_size)

    # Copy masks and record original sizes
    for filename in sorted(os.listdir(label_source)):
        if filename.lower().endswith(".png"):
            src_path = os.path.join(label_source, filename)
            dst_path = os.path.join(resized_label_dest, filename)

            # Load mask
            mask = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print("Error reading mask:", src_path)
                continue

            # Clean filename key: remove suffix like ".png"
            img_key = os.path.splitext(filename)[0]
            original_sizes_dict[img_key] = list(mask.shape)

            shutil.copy2(src_path, dst_path)



# -----------------------------------------
# Evaluation
# -----------------------------------------
def evaluation(model_path):
    # === Config ===
    NUM_CLASSES = 4
    IGNORED_CLASS = 3
    CLASS_NAMES = ["Cat", "Dog", "Background", "Boundary"]
    BATCH_SIZE = 32

    base_dir = "/Users/bin/Desktop/CV_Assignment/Dataset_filtered"
    original_sizes_path = os.path.join(base_dir, "original_sizes_test.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model ===
    model = EnhancedUNet(in_channels=3,out_channels=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # === Load original sizes ===
    with open(original_sizes_path, "r") as f:
        original_sizes = json.load(f)

    # === Custom collate (image, mask, img_name, mask_name)
    def custom_collate_fn(batch):
        images, masks, img_names, mask_names = zip(*batch)
        images = torch.stack(images)
        return images, list(masks), list(img_names), list(mask_names)

    # === Load test dataset ===
    test_dataset = CustomDataset(
        image_dir=os.path.join(base_dir, "test_resized", "color"),
        mask_dir=os.path.join(base_dir, "test_resized", "label"),
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    # === Metrics ===
    iou_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average=None).to(device)
    dice_metric = DiceScore(num_classes=NUM_CLASSES, average=None).to(device)

    # === Buffers for predictions and targets
    iou_preds, iou_targets = [], []
    dice_preds, dice_targets = [], []

    # Per-class accuracy stats
    per_class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long).to(device)
    per_class_total = torch.zeros(NUM_CLASSES, dtype=torch.long).to(device)

    # === Evaluation loop ===
    with torch.no_grad():
        for images, masks, img_names, _ in tqdm(test_loader):
            images = images.to(device)
            masks = [m.to(device).long() for m in masks]  # each mask: (H, W)

            logits = model(images)  # (B, C, 256, 256)

            for i in range(len(images)):
                name = img_names[i]
                orig_h, orig_w = original_sizes[name]
                logit = logits[i].unsqueeze(0)  # (1, C, H, W)
                resized_logit = F.interpolate(logit, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
                pred_mask = torch.argmax(resized_logit.squeeze(0), dim=0).long()  # (H, W)

                # Resize ground truth mask (safe way)
                gt_mask = F.interpolate(masks[i].unsqueeze(0).unsqueeze(0).float(),
                                        size=(orig_h, orig_w),
                                        mode="nearest").squeeze().long()

                # Filter out ignored class
                valid_mask = gt_mask != IGNORED_CLASS
                filtered_pred = pred_mask[valid_mask]
                filtered_target = gt_mask[valid_mask]

                if filtered_pred.numel() == 0:
                    continue  # skip fully-ignored images

                # Collect for metrics
                iou_preds.append(filtered_pred)
                iou_targets.append(filtered_target)
                dice_preds.append(filtered_pred)
                dice_targets.append(filtered_target)

                # Per-class pixel accuracy
                for cls in range(NUM_CLASSES):
                    if cls == IGNORED_CLASS:
                        continue
                    cls_mask = filtered_target == cls
                    per_class_total[cls] += cls_mask.sum()
                    per_class_correct[cls] += ((filtered_pred == cls) & cls_mask).sum()

    # === Compute scores
    all_preds = torch.cat(iou_preds)
    all_targets = torch.cat(iou_targets)

    iou_scores = iou_metric(all_preds, all_targets)
    dice_scores = dice_metric(
        F.one_hot(all_preds, NUM_CLASSES).permute(1, 0).unsqueeze(0),
        F.one_hot(all_targets, NUM_CLASSES).permute(1, 0).unsqueeze(0)
    )

    # === Print results
    print("\nPer-Class Evaluation on Test Set:")
    for i in range(NUM_CLASSES):
        if i == IGNORED_CLASS:
            continue
        print(f"Class {i} ({CLASS_NAMES[i]}): IoU = {iou_scores[i]:.4f}, Dice = {dice_scores[i]:.4f}")

    valid_iou_scores = [iou_scores[i] for i in range(NUM_CLASSES) if i != IGNORED_CLASS]
    valid_dice_scores = [dice_scores[i] for i in range(NUM_CLASSES) if i != IGNORED_CLASS]

    mean_iou = torch.stack(valid_iou_scores).mean()
    mean_dice = torch.stack(valid_dice_scores).mean()

    print(f"\nMean IoU (excluding Boundary): {mean_iou:.4f}")
    print(f"Mean Dice (excluding Boundary): {mean_dice:.4f}")

    # === Per-class pixel accuracy
    print(f"\nPixel Accuracy Per Class (excluding Boundary):")
    per_class_accuracies = []
    
    for i in range(NUM_CLASSES):
        if i == IGNORED_CLASS:
            continue
        if per_class_total[i] == 0:
            acc = 0.0
        else:
            acc = per_class_correct[i].float() / per_class_total[i]
        per_class_accuracies.append(acc)
        print(f"Class {i} ({CLASS_NAMES[i]}): Pixel Accuracy = {acc:.4f}")
    
    # === Mean Per-Class Accuracy
    mean_accuracy = torch.stack(per_class_accuracies).mean()
    print(f"\nMean Per-Class Pixel Accuracy (excluding Boundary): {mean_accuracy:.4f}")




if __name__ == "__main__":
    base_dir = "/Users/bin/Desktop/CV_Assignment/Dataset_filtered"

    # -----------------------------------------
    # Resize the Test Set
    # -----------------------------------------

    # Directories
    test_dir = os.path.join(base_dir, "Test")
    resized_test_color_dir = os.path.join(base_dir, "test_resized", "color")
    resized_test_label_dir = os.path.join(base_dir, "test_resized", "label")

    # Create folders
    os.makedirs(resized_test_color_dir, exist_ok=True)
    os.makedirs(resized_test_label_dir, exist_ok=True)

    # Dictionary for test original sizes
    original_sizes_test = {}

    # Source folders
    test_color_source = os.path.join(test_dir, "color")
    test_label_source = os.path.join(test_dir, "label")

    # Process test data
    process_data(
        color_source=test_color_source,
        label_source=test_label_source,
        resized_color_dest=resized_test_color_dir,
        resized_label_dest=resized_test_label_dir,
        original_sizes_dict=original_sizes_test
    )

    # Save test sizes JSON
    test_size_json_path = os.path.join(base_dir, "original_sizes_test.json")
    with open(test_size_json_path, "w") as f:
        json.dump(original_sizes_test, f, indent=4)

    print(f"Test data processed. Test mask sizes saved to {test_size_json_path}")

    evaluation(model_path=MODEL_PATH)