import random
import torchvision.transforms.functional as TF

class SegmentationAugment:
    def __init__(self, apply_color_jitter=True):
        # Whether to apply random color jitter to the image
        self.apply_color_jitter = apply_color_jitter
        
        # Define color jitter transformation: slight random changes in brightness, contrast, and saturation
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        )

    def __call__(self, image, mask):
        """
        Apply a single randomly chosen spatial transform to both image and mask.
        Note: augmentations regarding color changes are not applied to mask.
        """
        # List of available spatial augmentation operations
        ops = ['flip', 'rotate', 'affine', 'none']

        # Randomly pick one transformation to apply
        op = random.choice(ops)

        # Apply horizontal flip to both image and mask
        if op == 'flip':
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Apply small random rotation to both image and mask
        elif op == 'rotate':
            angle = random.uniform(-15, 15)  # Random angle between -15 and 15 degrees
            image = TF.rotate(image, angle, fill=0)     # Fill empty area in image with black (0)
            mask = TF.rotate(mask, angle, fill=255)     # Fill empty area in mask with 255 (treated as "unknown")

        # Apply affine transformation with translation and shear, no scaling or rotation
        elif op == 'affine':
            image = TF.affine(image, angle=0, translate=(10, 10), scale=1.0, shear=10, fill=0)
            mask = TF.affine(mask, angle=0, translate=(10, 10), scale=1.0, shear=10, fill=255)

        # Optionally apply color jitter to the image (never to the mask)
        if self.apply_color_jitter:
            image = self.color_jitter(image)

        # Return the transformed image and mask pair
        return image, mask







import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        image_dir: Directory containing input images
        mask_dir: Directory containing corresponding grayscale masks
        transform: A custom transform function that takes (image, mask) and returns (augmented_image, augmented_mask)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        assert len(self.image_filenames) == len(self.mask_filenames), "Image and mask count do not match"

        self.img_tensor_transform = transforms.ToTensor()
        self.mask_to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply any augmentation transformations
        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert image and mask to tensors
        image = self.img_tensor_transform(image)
        mask_tensor = self.mask_to_tensor(mask).squeeze(0)  # shape: H x W

        # Initialize an empty tensor to hold class labels for each pixel
        class_mask = torch.zeros_like(mask_tensor, dtype=torch.long)
        
        # Define pixel classification rules based on grayscale values:
        is_background = mask_tensor == 0.0
        is_boundary = mask_tensor == 1.0
        is_catdog = ~(is_background | is_boundary)  # All other pixels

        class_mask[is_background] = 2  # Label 2: Background
        class_mask[is_boundary] = 3    # Label 3: Object boundary

        # Use the filename to determine the class:
        # If the image filename starts with an uppercase letter, assign as Cat; otherwise Dog
        if img_name[0].isupper():
            class_mask[is_catdog] = 0  # Label 0: Cat
        else:
            class_mask[is_catdog] = 1  # Label 1: Dog

        return image, class_mask, os.path.splitext(img_name)[0], os.path.splitext(mask_name)[0]