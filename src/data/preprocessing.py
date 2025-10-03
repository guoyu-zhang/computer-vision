import os
import random
import json
import shutil
import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from .custom_dataset import SegmentationAugment

# ===== Path configuration =====
base_dir = "/Users/bin/Desktop/CV_Assignment/Dataset_filtered"

# ====== Cropping Function ======
def crop_and_save(original_sizes, crop_prob=0.35, crop_size=(324, 324)):
    input_image_dir = os.path.join(base_dir, "train", "color")
    input_mask_dir = os.path.join(base_dir, "train", "label")
    output_image_dir = os.path.join(base_dir, "train_crop", "color")
    output_mask_dir = os.path.join(base_dir, "train_crop", "label")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    image_filenames = sorted([f for f in os.listdir(input_image_dir) if f.endswith(".jpg")])
    for img_name in image_filenames:
        img_path = os.path.join(input_image_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(input_mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img_width, img_height = image.size
        key_orig = os.path.splitext(img_name)[0]
        original_sizes[key_orig] = [img_height, img_width]

        out_img_path_orig = os.path.join(output_image_dir, img_name)
        out_mask_path_orig = os.path.join(output_mask_dir, mask_name)
        image.save(out_img_path_orig, format="JPEG")
        mask.save(out_mask_path_orig, format="PNG")

        do_crop = random.random() < crop_prob
        use_crop = do_crop and (img_width >= crop_size[0]) and (img_height >= crop_size[1])
        if use_crop:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
            cropped_image = TF.crop(image, i, j, h, w)
            cropped_mask = TF.crop(mask, i, j, h, w)

            base_name = os.path.splitext(img_name)[0]
            new_img_name = f"crop_{base_name}.jpg"
            new_mask_name = f"crop_{base_name}.png"

            out_img_path_crop = os.path.join(output_image_dir, new_img_name)
            out_mask_path_crop = os.path.join(output_mask_dir, new_mask_name)

            cropped_image.save(out_img_path_crop, format="JPEG")
            cropped_mask.save(out_mask_path_crop, format="PNG")

            original_sizes[os.path.splitext(new_img_name)[0]] = [crop_size[1], crop_size[0]]

    print("Cropping complete. Files saved to 'train_crop'.")

# ====== Resize Function ======
def resize_and_save(src_path, dst_path, target_size):
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Error reading:", src_path)
        return
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(dst_path, resized)

def process_data(color_source, label_source, resized_color_dest, resized_label_dest, original_sizes_dict, target_size=(256, 256)):
    for filename in sorted(os.listdir(color_source)):
        if filename.lower().endswith(".jpg"):
            src_path = os.path.join(color_source, filename)
            dst_path = os.path.join(resized_color_dest, filename)
            resize_and_save(src_path, dst_path, target_size)

    for filename in sorted(os.listdir(label_source)):
        if filename.lower().endswith(".png"):
            src_path = os.path.join(label_source, filename)
            dst_path = os.path.join(resized_label_dest, filename)
            mask = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print("Error reading mask:", src_path)
                continue
            img_key = os.path.splitext(filename)[0]
            original_sizes_dict[img_key] = list(mask.shape)
            shutil.copy2(src_path, dst_path)

# ====== Augmentation Function ======
def apply_augmentation():
    input_img_dir = os.path.join(base_dir, "train_resized", "color")
    input_mask_dir = os.path.join(base_dir, "train_resized", "label")
    output_img_dir = os.path.join(base_dir, "train_randaug", "color")
    output_mask_dir = os.path.join(base_dir, "train_randaug", "label")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    augmentor = SegmentationAugment()
    num_augments = 3
    augmentation_prob = 0.35

    image_filenames = sorted([f for f in os.listdir(input_img_dir) if f.endswith(".jpg")])

    for img_name in image_filenames:
        mask_name = img_name.replace(".jpg", ".png")
        img_path = os.path.join(input_img_dir, img_name)
        mask_path = os.path.join(input_mask_dir, mask_name)

        original_image = Image.open(img_path).convert("RGB")
        original_mask = Image.open(mask_path).convert("L")

        out_img_path = os.path.join(output_img_dir, img_name)
        out_mask_path = os.path.join(output_mask_dir, mask_name)
        original_image.save(out_img_path, format="JPEG")
        original_mask.save(out_mask_path, format="PNG")

        if random.random() < augmentation_prob:
            for i in range(1, num_augments + 1):
                aug_img, aug_mask = augmentor(original_image.copy(), original_mask.copy())
                base_name = os.path.splitext(img_name)[0]
                aug_img_name = f"{base_name}_aug_{i}.jpg"
                aug_mask_name = f"{base_name}_aug_{i}.png"

                aug_img.save(os.path.join(output_img_dir, aug_img_name), format="JPEG")
                aug_mask.save(os.path.join(output_mask_dir, aug_mask_name), format="PNG")

    print("Data augmentation complete. Saved to 'train_randaug'.")

# ====== Main ======
if __name__ == "__main__":
    original_sizes = {}

    # Step 1: Crop
    crop_and_save(original_sizes)

    # Step 2: Resize
    resized_train_color_dir = os.path.join(base_dir, "train_resized", "color")
    resized_train_label_dir = os.path.join(base_dir, "train_resized", "label")
    resized_val_color_dir = os.path.join(base_dir, "val_resized", "color")
    resized_val_label_dir = os.path.join(base_dir, "val_resized", "label")

    for d in [resized_train_color_dir, resized_train_label_dir, resized_val_color_dir, resized_val_label_dir]:
        os.makedirs(d, exist_ok=True)

    process_data(
        os.path.join(base_dir, "train_crop", "color"),
        os.path.join(base_dir, "train_crop", "label"),
        resized_train_color_dir,
        resized_train_label_dir,
        original_sizes
    )

    process_data(
        os.path.join(base_dir, "val", "color"),
        os.path.join(base_dir, "val", "label"),
        resized_val_color_dir,
        resized_val_label_dir,
        original_sizes
    )

    with open(os.path.join(base_dir, "original_sizes.json"), "w") as f:
        json.dump(original_sizes, f, indent=4)

    print("Resizing complete. Sizes saved to JSON.")

    # Step 3: Apply Augmentation
    apply_augmentation()