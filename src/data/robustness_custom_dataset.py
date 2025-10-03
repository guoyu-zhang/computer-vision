import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import os
from skimage.util import random_noise

class RobustnessCustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, perturbation=None, level=0):
        """
        Args:
            image_dir (str): Directory containing input images.
            mask_dir (str): Directory containing corresponding mask images.
            transform (callable, optional): Optional transform to apply to both image and mask.
            perturbation (str, optional): Type of perturbation to apply.
            level (int): Strength level (0–9) for the perturbation.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.perturbation = perturbation
        self.level = level

        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        assert len(self.image_filenames) == len(self.mask_filenames), "Image/mask count mismatch"

        self.img_tensor_transform = transforms.ToTensor()
        self.mask_to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply perturbation to image only
        image_np = np.array(image)
        if self.perturbation:
            image_np = self.apply_perturbation(image_np, self.perturbation, self.level)
        image = Image.fromarray(image_np.astype(np.uint8))

        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert image and mask to tensor
        image = self.img_tensor_transform(image)
        mask_tensor = self.mask_to_tensor(mask).squeeze(0)

        # Create 4-class label mask: [Cat, Dog, Background, Boundary]
        class_mask = torch.zeros_like(mask_tensor, dtype=torch.long)
        is_background = mask_tensor == 0.0
        is_boundary = mask_tensor == 1.0
        is_catdog = ~(is_background | is_boundary)

        class_mask[is_background] = 2
        class_mask[is_boundary] = 3
        class_mask[is_catdog] = 0 if img_name[0].isupper() else 1

        return image, class_mask, os.path.splitext(img_name)[0], os.path.splitext(mask_name)[0]

    def apply_perturbation(self, image, perturbation, level):
        """
        Applies a specific perturbation to the image at a given intensity level.
        """
        if perturbation == 'gaussian_noise':
            stds = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
            noise = np.random.normal(0, stds[level], image.shape)
            image = np.clip(image + noise, 0, 255)

        elif perturbation == 'gaussian_blur':
            from scipy.ndimage import convolve
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
            for _ in range(level):
                image = convolve(image, kernel[..., None])

        elif perturbation == 'contrast_increase':
            factors = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.2, 1.25]
            image = np.clip(image * factors[level], 0, 255)

        elif perturbation == 'contrast_decrease':
            factors = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
            image = np.clip(image * factors[level], 0, 255)

        elif perturbation == 'brightness_increase':
            values = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
            image = np.clip(image + values[level], 0, 255)

        elif perturbation == 'brightness_decrease':
            values = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
            image = np.clip(image - values[level], 0, 255)

        elif perturbation == 'occlusion':
            sizes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
            size = sizes[level]
            h, w, _ = image.shape
            x = random.randint(0, max(0, w - size))
            y = random.randint(0, max(0, h - size))
            image[y:y+size, x:x+size] = 0  # black square patch

        elif perturbation == 'salt_pepper':
            amounts = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
            image = (random_noise(image / 255.0, mode='s&p', amount=amounts[level]) * 255).astype(np.uint8)

        return image
