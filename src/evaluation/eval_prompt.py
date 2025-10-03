import os
import random
import logging
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# Import CLIP for text prompt encoding.
import clip

# ------------------------------
# Setup Logger
# ------------------------------
logger = logging.getLogger("EvaluationLogger")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
f_handler = logging.FileHandler('evaluation_log.txt', mode='a')
f_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

logger.info("===== Starting Evaluation for Point-Based Segmentation Model =====")

# ------------------------------
# 0. Augmentation Class (if needed)
# ------------------------------
class SegmentationAugment:
    def __init__(self, apply_color_jitter=True):
        self.apply_color_jitter = apply_color_jitter
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def __call__(self, image, mask):
        ops = ['flip', 'rotate', 'affine', 'none']
        op = random.choice(ops)
        same = (image is mask)

        if op == 'flip':
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        elif op == 'rotate':
            angle = random.uniform(-15, 15)
            if same:
                image = TF.rotate(image, angle, fill=0)
                mask = TF.rotate(mask, angle, fill=0)
            else:
                image = TF.rotate(image, angle, fill=0)
                mask = TF.rotate(mask, angle, fill=(255, 255, 255))  # for boundary regions
        elif op == 'affine':
            if same:
                image = TF.affine(image, angle=0, translate=(10, 10), scale=1.0, shear=10, fill=0)
                mask = TF.affine(mask, angle=0, translate=(10, 10), scale=1.0, shear=10, fill=0)
            else:
                image = TF.affine(image, angle=0, translate=(10, 10), scale=1.0, shear=10, fill=0)
                mask = TF.affine(mask, angle=0, translate=(10, 10), scale=1.0, shear=10, fill=(255, 255, 255))
        # 'none': do nothing

        if self.apply_color_jitter and random.random() < 0.5:
            image = self.color_jitter(image)

        return image, mask

# ------------------------------
# 0.1 SAM-style Prompt Encoder Components
# ------------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.ln = nn.LayerNorm(num_features)
    def forward(self, x):
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2)

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        if scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats))
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1  # normalize to [-1,1]
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        grid_y = grid_y / h
        grid_x = grid_x / w
        grid = torch.stack([grid_x, grid_y], dim=-1)  # shape: (h, w, 2)
        pe = self._pe_encoding(grid)
        return pe.permute(2, 0, 1)  # (C, H, W)

    def forward_with_coords(self, coords: torch.Tensor, image_size: tuple) -> torch.Tensor:
        B, N, _ = coords.shape
        coords = coords.clone()
        coords[..., 0] = coords[..., 0] / image_size[1]
        coords[..., 1] = coords[..., 1] / image_size[0]
        return self._pe_encoding(coords)  # returns (B, N, C)

class SAMPromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple,
        input_image_size: tuple,
        activation: type = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings: int = 4  # (0: positive, 1: negative, 2: top-left box, 3: bottom-right box)
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool = False) -> torch.Tensor:
        points = points + 0.5  
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        B, N, C = point_embedding.shape
        for b in range(B):
            for n in range(N):
                if labels[b, n] == -1:
                    point_embedding[b, n] += self.not_a_point_embed.weight.squeeze(0)
                elif labels[b, n] == 1:
                    point_embedding[b, n] += self.point_embeddings[0].weight.squeeze(0)
                elif labels[b, n] == 0:
                    point_embedding[b, n] += self.point_embeddings[1].weight.squeeze(0)
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5  
        B = boxes.shape[0]
        boxes = boxes.view(B, 2, 2)  # (B, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(boxes, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight.squeeze(0)
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight.squeeze(0)
        box_embedding = corner_embedding.mean(dim=1, keepdim=True)  # (B, 1, embed_dim)
        return box_embedding

    def forward(self, points: tuple = None, boxes: torch.Tensor = None) -> torch.Tensor:
        embeddings = []
        if points is not None:
            coords, labels = points
            point_embeds = self._embed_points(coords, labels, pad=(boxes is None))
            embeddings.append(point_embeds)
        if boxes is not None:
            box_embeds = self._embed_boxes(boxes)
            embeddings.append(box_embeds)
        if len(embeddings) > 0:
            combined = torch.cat(embeddings, dim=1)  # (B, total_prompts, embed_dim)
            combined = combined.mean(dim=1, keepdim=True)  # (B, 1, embed_dim)
            return combined
        else:
            bs = 1
            return self.not_a_point_embed.weight.unsqueeze(0).expand(bs, -1).unsqueeze(1)

# ------------------------------
# New Component: Text Prompt Encoder
# ------------------------------
class TextPromptEncoder(nn.Module):
    """
    Uses a pre-trained CLIP text encoder to get text embeddings and then
    projects them to the same prompt embedding dimension.
    """
    def __init__(self, clip_model, prompt_embed_dim: int):
        super().__init__()
        self.clip_model = clip_model.eval()  # set to eval
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.prompt_projection = nn.Linear(512, prompt_embed_dim)

    def forward(self, text_prompts):
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        text_tokens = clip.tokenize(text_prompts).to(next(self.clip_model.parameters()).device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(text_tokens)
        text_embeddings = text_embeddings.float()  # (B, 512)
        text_embeddings = self.prompt_projection(text_embeddings)  # (B, prompt_embed_dim)
        return text_embeddings.unsqueeze(1)  # (B, 1, prompt_embed_dim)

# ------------------------------
# 1. Define the Point-based Segmentation Model with Multi-modal Prompts
# ------------------------------
class PointSegUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, prompt_embed_dim=32,
                 image_embedding_size=(16, 16), input_image_size=(256, 256),
                 clip_model=None):
        super(PointSegUNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.encoder1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck = conv_block(256, 512)
        # SAM-style Prompt Encoder (for points and boxes)
        self.prompt_encoder = SAMPromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=input_image_size,
            activation=nn.GELU
        )
        # Text prompt encoder if a CLIP model is provided.
        if clip_model is not None:
            self.text_prompt_encoder = TextPromptEncoder(clip_model, prompt_embed_dim)
            self.prompt_fusion = nn.Linear(prompt_embed_dim * 2, prompt_embed_dim)
        else:
            self.text_prompt_encoder = None

        # Fuse bottleneck features with prompt features.
        self.bottleneck_fuse = nn.Sequential(
            nn.Conv2d(512 + prompt_embed_dim, 512, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = conv_block(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x, prompt_points=None, prompt_boxes=None, text_prompt=None):
        # Encoder forward
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)

        # Process prompt information.
        if prompt_points is not None or prompt_boxes is not None:
            multimodal_prompt = self.prompt_encoder(points=prompt_points, boxes=prompt_boxes)
        else:
            multimodal_prompt = None

        if text_prompt is not None and self.text_prompt_encoder is not None:
            text_embedding = self.text_prompt_encoder(text_prompt)
        else:
            text_embedding = None

        if multimodal_prompt is not None and text_embedding is not None:
            combined = torch.cat([multimodal_prompt, text_embedding], dim=-1)
            combined = self.prompt_fusion(combined)
        elif multimodal_prompt is not None:
            combined = multimodal_prompt
        elif text_embedding is not None:
            combined = text_embedding
        else:
            combined = None

        if combined is not None:
            prompt_embed = combined.squeeze(1)
            spatial_size = b.shape[-2:]
            prompt_features = prompt_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spatial_size[0], spatial_size[1])
        else:
            prompt_features = torch.zeros_like(b)

        # Fuse bottleneck features and prompt features.
        b_cat = torch.cat((b, prompt_features), dim=1)
        b_fused = self.bottleneck_fuse(b_cat)
        # Decoder forward
        u4 = self.upconv4(b_fused)
        d4 = self.decoder4(torch.cat((u4, e4), dim=1))
        u3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat((u3, e3), dim=1))
        u2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat((u2, e2), dim=1))
        u1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat((u1, e1), dim=1))
        output = self.final_conv(d1)
        return output

# ------------------------------
# 2. Define the Dataset for Evaluation
# ------------------------------
class PointSegmentationDataset(Dataset):
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

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)
        if self.transform:
            image = self.transform(image)
        
        # Convert mask to segmentation labels.
        mask_np = np.array(mask)
        h, w, _ = mask_np.shape
        label_mask = np.zeros((h, w), dtype=np.int64)
        
        if img_name[0].isupper():
            obj_label = 1  # cat
            text_prompt = "cat"
            # Map mask colors to labels for cats
            object_region = (mask_np[..., 0] == 128) & (mask_np[..., 1] == 0) & (mask_np[..., 2] == 0)
        else:
            obj_label = 2  # dog
            text_prompt = "dog"
            # Check if you need different color mapping for dogs
            # For example, if dogs are colored (0,128,0) in your masks:
            object_region = (mask_np[..., 0] == 0) & (mask_np[..., 1] == 128) & (mask_np[..., 2] == 0)

        # # Now assign the label
        label_mask[object_region] = obj_label
        
        # # Map mask colors: (128,0,0) for the object region.
        # object_region = (mask_np[..., 0] == 128) & (mask_np[..., 1] == 0) & (mask_np[..., 2] == 0)
        # label_mask[object_region] = obj_label
        # White color (255,255,255) for boundaries.
        boundary_region = (mask_np[..., 0] == 255) & (mask_np[..., 1] == 255) & (mask_np[..., 2] == 255)
        label_mask[boundary_region] = 3
        seg_mask = torch.from_numpy(label_mask)
        
        target_size = (256, 256)
        indices = np.argwhere(object_region)
        if indices.size > 0:
            idx_choice = random.choice(indices.tolist())
            label = 1  # positive prompt
            y_min, x_min = indices.min(axis=0)
            y_max, x_max = indices.max(axis=0)
            y_min_new = int((y_min / h) * target_size[0])
            x_min_new = int((x_min / w) * target_size[1])
            y_max_new = int((y_max / h) * target_size[0])
            x_max_new = int((x_max / w) * target_size[1])
            prompt_box = torch.tensor([float(x_min_new), float(y_min_new), float(x_max_new), float(y_max_new)])
        else:
            indices = np.argwhere(label_mask == 0)
            if indices.size > 0:
                idx_choice = random.choice(indices.tolist())
            else:
                idx_choice = [0, 0]
            label = 0  # negative prompt
            prompt_box = torch.tensor([0.0, 0.0, 0.0, 0.0])
        
        # For point prompt, swap (y,x) to (x,y)
        y, x = idx_choice[0], idx_choice[1]
        y_new = int((y / h) * target_size[0])
        x_new = int((x / w) * target_size[1])
        prompt_point = torch.tensor([[float(x_new), float(y_new)]])
        prompt_label = torch.tensor([label], dtype=torch.long)
        
        img_id = os.path.splitext(img_name)[0]
        mask_id = os.path.splitext(mask_name)[0]
        
        return image, prompt_point, prompt_label, prompt_box, seg_mask, img_id, mask_id, text_prompt

def point_segmentation_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    prompt_points = torch.stack([item[1] for item in batch], dim=0)
    prompt_labels = torch.stack([item[2] for item in batch], dim=0)
    prompt_boxes = torch.stack([item[3] for item in batch], dim=0)
    masks = [item[4] for item in batch]
    img_ids = [item[5] for item in batch]
    mask_ids = [item[6] for item in batch]
    text_prompts = [item[7] for item in batch]
    return images, (prompt_points, prompt_labels, prompt_boxes, text_prompts), masks, img_ids, mask_ids

# ------------------------------
# 3. Evaluation Metric Functions
# ------------------------------
def evaluate_model(model, dataloader, device, num_classes=4):
    model.eval()
    total_pixels = 0
    correct_pixels = 0
    total_intersection = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)
    total_pred = np.zeros(num_classes, dtype=np.float64)
    total_gt = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for images, prompt_tuple, masks, _, _ in dataloader:
            images = images.to(device)
            prompt_points, prompt_labels, prompt_boxes, text_prompts = prompt_tuple
            prompt_points = prompt_points.to(device)
            prompt_labels = prompt_labels.to(device)
            prompt_boxes = prompt_boxes.to(device)

            # Forward pass: your model accepts text prompt info as list of strings.
            logits = model(images,
                           prompt_points=(prompt_points, prompt_labels),
                           prompt_boxes=prompt_boxes,
                           text_prompt=text_prompts)
            
            for i in range(images.size(0)):
                logit = logits[i].unsqueeze(0)
                target_mask = masks[i].unsqueeze(0).to(device)
                if logit.shape[-2:] != target_mask.shape[-2:]:
                    logit = F.interpolate(logit, size=target_mask.shape[-2:], mode='bilinear', align_corners=False)
                pred_mask = torch.argmax(logit, dim=1)
                total_pixels += target_mask.numel()
                correct_pixels += (pred_mask == target_mask).sum().item()

                for c in range(num_classes):
                    pred_c = (pred_mask == c)
                    target_c = (target_mask == c)
                    intersection = (pred_c & target_c).sum().item()
                    union = (pred_c | target_c).sum().item()
                    total_intersection[c] += intersection
                    total_union[c] += union
                    total_pred[c] += pred_c.sum().item()
                    total_gt[c] += target_c.sum().item()

    global_pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    iou_scores = {}
    dice_scores = {}
    per_class_pixel_accuracy = {}
    for c in range(num_classes):
        iou = total_intersection[c] / total_union[c] if total_union[c] > 0 else 0.0
        dice = (2 * total_intersection[c]) / (total_pred[c] + total_gt[c]) if (total_pred[c] + total_gt[c]) > 0 else 0.0
        # Compute per-class pixel accuracy as intersection divided by total ground truth pixels for the class.
        class_pixel_accuracy = total_intersection[c] / total_gt[c] if total_gt[c] > 0 else 0.0
        iou_scores[c] = iou
        dice_scores[c] = dice
        per_class_pixel_accuracy[c] = class_pixel_accuracy

    return global_pixel_accuracy, iou_scores, dice_scores, per_class_pixel_accuracy

# ------------------------------
# Update the Main Evaluation Script
# ------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # ---------------
    # Load CLIP model (use same parameters/settings as training)
    # ---------------
    local_clip_dir = "/home/s1808795"  # update as needed
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, download_root=local_clip_dir)

    # ---------------
    # Initialize the segmentation model with same parameters as in training.
    # ---------------
    seg_model = PointSegUNet(in_channels=3,
                             out_channels=4,
                             prompt_embed_dim=32,
                             image_embedding_size=(16, 16),
                             input_image_size=(256, 256),
                             clip_model=clip_model)
    seg_model = seg_model.to(device)

    # ---------------
    # Load the trained model checkpoint.
    # ---------------
    checkpoint_path = "/home/s1808795/CV_Assignment/best_point_segmentation_model_catanddog.pth"
    if os.path.exists(checkpoint_path):
        seg_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model checkpoint from {checkpoint_path}")
        logger.info(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Exiting...")
        logger.error(f"Checkpoint not found at {checkpoint_path}. Exiting...")
        return

    # ---------------
    # Create the validation dataset and dataloader.
    # ---------------
    val_image_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/val_resized/color"
    val_mask_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/val_resized/label"
    transform = transforms.Compose([transforms.ToTensor()])
    augmentation = None  # No augmentation during evaluation.
    val_dataset = PointSegmentationDataset(val_image_dir, val_mask_dir, transform=transform, augmentation=augmentation)
    batch_size = 16
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=point_segmentation_collate_fn)

    # ---------------
    # Run evaluation.
    # ---------------
    global_pixel_acc, iou_scores, dice_scores, per_class_pixel_acc = evaluate_model(seg_model, val_dataloader, device, num_classes=4)

    # ---------------
    # Print evaluation results.
    # ---------------
    print("========== Evaluation Results ==========")
    print(f"Overall Pixel Accuracy: {global_pixel_acc * 100:.2f}%")
    logger.info(f"Overall Pixel Accuracy: {global_pixel_acc * 100:.2f}%")
    for c in range(4):
        print(f"Class {c}: IoU = {iou_scores[c]:.4f}, Dice (F1) = {dice_scores[c]:.4f}, Pixel Accuracy = {per_class_pixel_acc[c] * 100:.2f}%")
        logger.info(f"Class {c}: IoU = {iou_scores[c]:.4f}, Dice (F1) = {dice_scores[c]:.4f}, Pixel Accuracy = {per_class_pixel_acc[c] * 100:.2f}%")
    print("==========================================")

if __name__ == "__main__":
    main()
