import os
import random
import logging
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# Import CLIP for text prompt encoding.
import clip

# ------------------------------
# Setup Logger
# ------------------------------
logger = logging.getLogger("TrainingLogger")
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
f_handler = logging.FileHandler('training_log.txt', mode='a')
f_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

logger.info("===== Starting End-to-End Point-Based Segmentation Training =====")

# ------------------------------
# 0. Augmentation Class
# ------------------------------
from ..data.custom_dataset import SegmentationAugment

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
        return x.permute(0, 3, 1, 2)  # back to (B, C, H, W)

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
        self.num_point_embeddings: int = 4  # indices: 0 (positive), 1 (negative), 2 (top-left box), 3 (bottom-right box)
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
        self.clip_model = clip_model.eval()  # set to evaluation mode
        # Freeze CLIP parameters.
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
        text_embeddings = text_embeddings.unsqueeze(1)  # (B, 1, prompt_embed_dim)
        return text_embeddings

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
        # Instantiate TextPromptEncoder if a CLIP model is provided.
        if clip_model is not None:
            self.text_prompt_encoder = TextPromptEncoder(clip_model, prompt_embed_dim)
            self.prompt_fusion = nn.Linear(prompt_embed_dim * 2, prompt_embed_dim)
        else:
            self.text_prompt_encoder = None

        # Fusion layer for bottleneck features and prompt features.
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
        # Image encoder.
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)
        
        # Process point and box prompts.
        if prompt_points is not None or prompt_boxes is not None:
            multimodal_prompt = self.prompt_encoder(points=prompt_points, boxes=prompt_boxes)
        else:
            multimodal_prompt = None

        # Process text prompt.
        if text_prompt is not None and self.text_prompt_encoder is not None:
            text_embedding = self.text_prompt_encoder(text_prompt)
        else:
            text_embedding = None

        # Fuse available prompt embeddings.
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

        # Fuse bottleneck features with prompt features.
        b_cat = torch.cat((b, prompt_features), dim=1)
        b_fused = self.bottleneck_fuse(b_cat)
        # Decoder.
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
# 2. Define the Dataset for Point-based Segmentation with Bounding Boxes and Text Prompts
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
        # Load mask as RGB (since colors matter for mapping)
        mask = Image.open(mask_path).convert("RGB")
        
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)
        if self.transform:
            image = self.transform(image)
        
        # Convert mask (RGB) to a segmentation label mask.
        mask_np = np.array(mask)
        h, w, _ = mask_np.shape
        label_mask = np.zeros((h, w), dtype=np.int64)
        
        # Determine object class based on filename:
        # If filename starts with uppercase → cat (label 1); else → dog (label 2)
        # Determine object class based on filename:
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

        # Now assign the label
        label_mask[object_region] = obj_label
                
        # Map mask colors to labels:
        # Object region in mask is colored (128,0,0)
        # object_region = (mask_np[..., 0] == 128) & (mask_np[..., 1] == 0) & (mask_np[..., 2] == 0)
        # label_mask[object_region] = obj_label
        # Boundary is white (255,255,255)
        boundary_region = (mask_np[..., 0] == 255) & (mask_np[..., 1] == 255) & (mask_np[..., 2] == 255)
        label_mask[boundary_region] = 3
        # Background remains 0.
        seg_mask = torch.from_numpy(label_mask)
        
        target_size = (256, 256)
        # Compute object prompt box and point using the object_region mask.
        indices = np.argwhere(object_region)
        if indices.size > 0:
            idx_choice = random.choice(indices.tolist())
            label = 1  # positive for object point prompt
            y_min, x_min = indices.min(axis=0)
            y_max, x_max = indices.max(axis=0)
            y_min_new = int((y_min / h) * target_size[0])
            x_min_new = int((x_min / w) * target_size[1])
            y_max_new = int((y_max / h) * target_size[0])
            x_max_new = int((x_max / w) * target_size[1])
            prompt_box = torch.tensor([float(x_min_new), float(y_min_new), float(x_max_new), float(y_max_new)])
        else:
            # If no object region, select a random background point.
            indices = np.argwhere(label_mask == 0)
            if indices.size > 0:
                idx_choice = random.choice(indices.tolist())
            else:
                idx_choice = [0, 0]
            label = 0  # negative point prompt
            prompt_box = torch.tensor([0.0, 0.0, 0.0, 0.0])
        
        # For the point prompt, swap (y,x) to (x,y) coordinates.
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
    prompt_points = torch.stack([item[1] for item in batch], dim=0)   # (B, 1, 2)
    prompt_labels = torch.stack([item[2] for item in batch], dim=0)   # (B, 1)
    prompt_boxes = torch.stack([item[3] for item in batch], dim=0)    # (B, 4)
    masks = [item[4] for item in batch]
    img_ids = [item[5] for item in batch]
    mask_ids = [item[6] for item in batch]
    text_prompts = [item[7] for item in batch]  # list of strings
    return images, (prompt_points, prompt_labels, prompt_boxes, text_prompts), masks, img_ids, mask_ids

# ------------------------------
# 3. Loss Functions and Utility Functions
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

def compute_class_weights(dataset, num_classes=4):
    logger.info("Computing class weights...")
    class_count = torch.zeros(num_classes)
    for i in range(len(dataset)):
        _, _, _, _, mask, _, _, _ = dataset[i]
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

def colorize_mask(mask):
    """Convert a segmentation mask (4 classes) to a color image for visualization.
       Mapping: 0: Background (black), 1: Cat (128,0,0), 2: Dog (0,128,0), 3: Boundary (white)."""
    color_map = {
        0: (0, 0, 0),           # Background: black
        1: (128, 0, 0),         # Cat: use the original red-ish color
        2: (0, 128, 0),         # Dog: a different color (greenish)
        3: (255, 255, 255)      # Boundary: white
    }
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_mask[mask == label] = color
    return Image.fromarray(color_mask)

# ------------------------------
# 4. End-to-End Training Function
# ------------------------------
def train_point_segmentation():
    logger.info("=== Starting Point-Based Segmentation Training ===")
    # Define paths (adjust these paths as needed)
    train_image_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/train_resized/color"
    train_mask_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/train_resized/label"
    val_image_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/val_resized/color"
    val_mask_dir = "/home/s1808795/CV_Assignment/Dataset_filtered/val_resized/label"

    num_epochs = 1000
    batch_size = 16
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------------------
    # Load CLIP model for text prompt encoding.
    logger.info("Loading CLIP model...")
    local_clip_dir = "/home/s1808795"  # Update as needed
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, download_root=local_clip_dir)

    transform = transforms.Compose([transforms.ToTensor()])
    augmentation = SegmentationAugment(apply_color_jitter=True)

    train_dataset = PointSegmentationDataset(train_image_dir, train_mask_dir, transform=transform, augmentation=augmentation)
    val_dataset = PointSegmentationDataset(val_image_dir, val_mask_dir, transform=transform, augmentation=None)
    logger.info(f"Number of training images: {len(train_dataset)}")
    logger.info(f"Number of validation images: {len(val_dataset)}")

    debug_mode = False
    if debug_mode:
        train_dataset = torch.utils.data.Subset(train_dataset, list(range(5)))
        logger.info("Debug mode enabled: using only 5 training images.")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=point_segmentation_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=point_segmentation_collate_fn)

    class_weights = compute_class_weights(train_dataset, num_classes=4)
    class_weights = class_weights.to(device)

    criterion = CombinedLoss(weights=class_weights, alpha=0.5, beta=0.5)
    # Pass the loaded CLIP model to our segmentation network.
    seg_model = PointSegUNet(in_channels=3, out_channels=4,
                             prompt_embed_dim=32,
                             image_embedding_size=(16, 16),
                             input_image_size=(256, 256),
                             clip_model=clip_model).to(device)
    optimizer = optim.Adam(seg_model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(num_epochs):
        seg_model.train()
        running_loss = 0.0
        for batch_idx, (imgs, prompt_tuple, masks, img_ids, mask_ids) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            prompt_points, prompt_labels, prompt_boxes, text_prompts = prompt_tuple
            prompt_points = prompt_points.to(device)
            prompt_labels = prompt_labels.to(device)
            prompt_boxes = prompt_boxes.to(device)
            logits = seg_model(imgs, prompt_points=(prompt_points, prompt_labels),
                               prompt_boxes=prompt_boxes, text_prompt=text_prompts)
            losses = []
            for i in range(imgs.size(0)):
                logit = logits[i].unsqueeze(0)
                target_mask = masks[i].unsqueeze(0).to(device)
                if logit.shape[-2:] != target_mask.shape[-2:]:
                    logit = F.interpolate(logit, size=target_mask.shape[-2:], mode='bilinear', align_corners=False)
                sample_loss = criterion(logit, target_mask)
                losses.append(sample_loss)
            batch_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}")

        seg_model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, prompt_tuple, masks, img_ids, mask_ids in val_dataloader:
                images = images.to(device)
                prompt_points, prompt_labels, prompt_boxes, text_prompts = prompt_tuple
                prompt_points = prompt_points.to(device)
                prompt_labels = prompt_labels.to(device)
                prompt_boxes = prompt_boxes.to(device)
                logits = seg_model(images, prompt_points=(prompt_points, prompt_labels),
                                     prompt_boxes=prompt_boxes, text_prompt=text_prompts)
                batch_val_loss = 0.0
                for i in range(images.size(0)):
                    logit = logits[i].unsqueeze(0)
                    target_mask = masks[i].unsqueeze(0).to(device)
                    if logit.shape[-2:] != target_mask.shape[-2:]:
                        logit = F.interpolate(logit, size=target_mask.shape[-2:], mode='bilinear', align_corners=False)
                    sample_loss = criterion(logit, target_mask)
                    batch_val_loss += sample_loss
                running_val_loss += batch_val_loss / images.size(0)
            val_loss = running_val_loss / len(val_dataloader)
            val_losses.append(val_loss)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save(seg_model.state_dict(), "best_point_segmentation_model_catanddog.pth")
            logger.info(f"New best validation loss: {val_loss:.4f}. Best model saved.")
        else:
            no_improvement_count += 1
            logger.info(f"No improvement count: {no_improvement_count}")

        if no_improvement_count >= 10:
            logger.info("Early stopping triggered after 10 epochs without improvement.")
            break

    
        # if epoch in [0,1,4,6,10,20,40,80,100,150,200,250,300]:
        #     # Inference Example Composite Images per Epoch (up to 5 examples)
        #     seg_model.eval()
        #     with torch.no_grad():
        #         for i in [1,700]:
        #             sample = val_dataset[i]
        #             sample_img, sample_prompt_point, sample_prompt_label, sample_prompt_box, sample_mask, sample_img_id, sample_mask_id, sample_text_prompt = sample
        #             sample_tensor = sample_img.unsqueeze(0).to(device)
        #             sample_prompt_points = sample_prompt_point.unsqueeze(0).to(device)
        #             sample_prompt_labels = sample_prompt_label.unsqueeze(0).to(device)
        #             sample_prompt_boxes = sample_prompt_box.unsqueeze(0).to(device)
        #             text_prompt = sample_text_prompt  # text prompt is a string

        #             logits = seg_model(sample_tensor, prompt_points=(sample_prompt_points, sample_prompt_labels),
        #                                 prompt_boxes=sample_prompt_boxes, text_prompt=text_prompt)
        #             if logits.shape[-2:] != sample_mask.shape:
        #                 logits = F.interpolate(logits, size=sample_mask.shape, mode='bilinear', align_corners=False)
        #             pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        #             color_pred = colorize_mask(pred_mask)
        #             color_gt = colorize_mask(sample_mask.numpy())
        #             to_pil = transforms.ToPILImage()
        #             input_img_pil = to_pil(sample_tensor.squeeze(0).cpu().clamp(0, 1))
                    
        #             prompt_np = sample_prompt_point.squeeze(0).cpu().numpy()
        #             x_coord, y_coord = int(prompt_np[0]), int(prompt_np[1])
        #             radius = 10
        #             input_with_prompt = input_img_pil.copy()
        #             draw_overlay = ImageDraw.Draw(input_with_prompt)
        #             draw_overlay.ellipse((x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius), fill=(255, 255, 0))
        #             box = sample_prompt_box.cpu().numpy()
        #             if not np.allclose(box, 0.0):
        #                 draw_overlay.rectangle((box[0], box[1], box[2], box[3]), outline=(0, 255, 255), width=2)
                    
        #             size1 = input_img_pil.size
        #             size2 = input_img_pil.size
        #             size3 = color_pred.size
        #             size4 = color_gt.size
        #             composite_width = size1[0] + size2[0] + size3[0] + size4[0]
        #             composite_height = max(size1[1], size2[1], size3[1], size4[1])
        #             composite = Image.new('RGB', (composite_width, composite_height))
        #             x_offset = 0
        #             composite.paste(input_img_pil, (x_offset, 0))
        #             x_offset += size1[0]
        #             composite.paste(input_with_prompt, (x_offset, 0))
        #             x_offset += size2[0]
        #             composite.paste(color_pred, (x_offset, 0))
        #             x_offset += size3[0]
        #             composite.paste(color_gt, (x_offset, 0))
                    
        #             seg_inference_folder = "point_seg_inference_examples"
        #             if not os.path.exists(seg_inference_folder):
        #                 os.makedirs(seg_inference_folder)
        #             safe_text = text_prompt.replace(" ", "_")
        #             composite_save_path = os.path.join(seg_inference_folder, f"epoch_{epoch+1}_inference_{i+1}_{safe_text}.png")
        #             composite.save(composite_save_path)
        #             logger.info(f"Saved composite inference example for epoch {epoch+1}, sample {i+1} with text prompt: {text_prompt}")

        torch.save(seg_model.state_dict(), "point_segmentation_model_catanddog.pth")
        logger.info(f"Checkpoint model saved for epoch {epoch+1}")

# ------------------------------
# 5. Main: Run the Training
# ------------------------------
if __name__ == '__main__':
    train_point_segmentation()
