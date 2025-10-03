import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import clip


###################### Define Helper Functions #####################

def colorize_mask(mask):
    """Convert a segmentation mask (with 4 classes) to a color image.
       Mapping: 0=Background (black), 1=Cat (128, 0, 0), 2=Dog (0, 128, 0), 3=Boundary (white)."""
    color_map = {
        0: (0, 0, 255),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (255, 255, 0)
    }
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_mask[mask == label] = color
    return Image.fromarray(color_mask)

###################### Define Model Components ######################

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
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (h, w, 2)
        pe = self._pe_encoding(grid)
        return pe.permute(2, 0, 1)  # (C, H, W)

    def forward_with_coords(self, coords: torch.Tensor, image_size: tuple) -> torch.Tensor:
        B, N, _ = coords.shape
        coords = coords.clone()
        coords[..., 0] = coords[..., 0] / image_size[1]
        coords[..., 1] = coords[..., 1] / image_size[0]
        return self._pe_encoding(coords)  # returns (B, N, C)

class PromptEncoder(nn.Module):
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

class TextPromptEncoder(nn.Module):
    """
    Uses a pre-trained CLIP text encoder to get text embeddings then projects them to the prompt embedding dimension.
    """
    def __init__(self, clip_model, prompt_embed_dim: int):
        super().__init__()
        self.clip_model = clip_model.eval()  # evaluation mode
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
        # Encoder.
        self.encoder1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck.
        self.bottleneck = conv_block(256, 512)
        # SAM-style Prompt Encoder (for points and boxes).
        self.prompt_encoder = PromptEncoder(
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
        # Decoder.
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
        
        # Process prompt inputs.
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

###################### Prepare the Visualizer UI ######################
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

local_clip_dir = "/Users/guoyuzhang/University/Y5/CV/CV_Assignment/"  # Update as needed
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, download_root=local_clip_dir)

# Instantiate the trained segmentation model and load checkpoint.
model = PointSegUNet(in_channels=3, out_channels=4,
                     prompt_embed_dim=32,
                     image_embedding_size=(16, 16),
                     input_image_size=(256, 256),
                     clip_model=clip_model).to(device)
checkpoint_path = "/Users/guoyuzhang/University/Y5/CV/CV_Assignment/best_point_segmentation_model_catanddog.pth"

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print("Loaded model checkpoint.")
model.eval()

# Define image transformation for model input.
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

###################### Build the Tkinter GUI ######################
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Point-Based U-Net UI")
        
        # --- Controls Frame ---
        controls_frame = tk.Frame(root)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.load_button = tk.Button(controls_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.prompt_mode = tk.StringVar(value="Point")
        mode_frame = tk.LabelFrame(controls_frame, text="Prompt Mode")
        mode_frame.pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(mode_frame, text="Point", variable=self.prompt_mode, value="Point").pack(anchor="w")
        tk.Radiobutton(mode_frame, text="Bounding Box", variable=self.prompt_mode, value="Box").pack(anchor="w")
        
        tk.Label(controls_frame, text="Text Prompt:").pack(side=tk.LEFT, padx=5)
        self.text_entry = tk.Entry(controls_frame, width=20)
        self.text_entry.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(controls_frame, text="Clear Prompts", command=self.clear_prompts)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.infer_button = tk.Button(controls_frame, text="Run Inference", command=self.run_inference)
        self.infer_button.pack(side=tk.LEFT, padx=5)
        
        # --- Main Display Frame for Images ---
        display_frame = tk.Frame(root)
        display_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Canvas for the input image with prompts.
        self.canvas = tk.Canvas(display_frame, bg='gray')
        self.canvas.pack(side=tk.LEFT, padx=5)
        
        # Label for showing the predicted mask.
        self.result_label = tk.Label(display_frame)
        self.result_label.pack(side=tk.LEFT, padx=5)
        
        # --- Legend Frame (always visible on the side) ---
        legend_frame = tk.Frame(root, bd=2, relief=tk.RIDGE)
        legend_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        tk.Label(legend_frame, text="Legend", font=("Arial", 12, "bold")).pack(pady=5)
        legend_items = [
            ("Background", (0, 0, 255)),
            ("Cat", (255, 0, 0)),
            ("Dog", (0, 255, 0)),
            ("Boundary", (255, 255, 0))
        ]
        for text, color in legend_items:
            item_frame = tk.Frame(legend_frame)
            item_frame.pack(fill=tk.X, pady=2, padx=5)
            color_box = tk.Canvas(item_frame, width=20, height=20)
            color_box.pack(side=tk.LEFT)
            color_hex = "#%02x%02x%02x" % color
            color_box.create_rectangle(0, 0, 20, 20, fill=color_hex)
            tk.Label(item_frame, text=text, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        # Store image and prompt variables.
        self.image = None         # Original PIL image in true dimensions.
        self.tk_input = None      # For displaying the input image.
        self.point = None         # (x, y) in original coordinates.
        self.box = None           # (x_min, y_min, x_max, y_max) in original coordinates.
        self.rect_id = None       # Canvas rectangle id.
        self.start_x = None       # For bounding box drawing.
        self.start_y = None

        # Bind mouse events for drawing on the canvas.
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def load_image(self):
        file_path = filedialog.askopenfilename(initialdir="/Users/guoyuzhang/University/Y5/CV/CV_Assignment/Dataset_filtered/Test/color", 
                                               filetypes=[("Image files", "*.jpg *.png *.jpeg"), ("All files", "*.*")])
        if file_path:
            # Load image in its true dimensions.
            self.image = Image.open(file_path).convert("RGB")
            self.tk_input = ImageTk.PhotoImage(self.image)
            # Configure canvas to match image dimensions.
            self.canvas.config(width=self.image.width, height=self.image.height)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_input)
            self.clear_prompts()

    def clear_prompts(self):
        # Reset prompt data.
        self.point = None
        self.box = None
        self.canvas.delete("marker")
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        # Clear the text prompt box.
        self.text_entry.delete(0, tk.END)
        # Clear the predicted mask display.
        self.result_label.configure(image="")
        self.result_label.image = None
        # Reset the canvas to show the original input image (if loaded).
        if self.image is not None:
            self.tk_input = ImageTk.PhotoImage(self.image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_input)

    def on_mouse_down(self, event):
        self.start_x, self.start_y = event.x, event.y

    def on_mouse_move(self, event):
        mode = self.prompt_mode.get()
        if mode == "Box":
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="cyan", width=2, tag="marker")
        elif mode == "Point":
            self.canvas.delete("marker")
            radius = 3
            self.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, fill="yellow", tag="marker")

    def on_mouse_up(self, event):
        mode = self.prompt_mode.get()
        if mode == "Point":
            self.point = (event.x, event.y)
            self.canvas.delete("marker")
            radius = 5
            self.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, fill="yellow", tag="marker")
        elif mode == "Box":
            x_min, y_min = min(self.start_x, event.x), min(self.start_y, event.y)
            x_max, y_max = max(self.start_x, event.x), max(self.start_y, event.y)
            self.box = (x_min, y_min, x_max, y_max)
        self.start_x, self.start_y = None, None

    def run_inference(self):
        if self.image is None:
            print("No image loaded!")
            return
        
        # Get the true image dimensions.
        true_w, true_h = self.image.width, self.image.height
        # Preprocess the image for the model. transform_img resizes to 256x256.
        image_tensor = transform_img(self.image).unsqueeze(0).to(device)
        
        # Compute scaling factors from original dimensions to 256x256.
        scale_x = 256 / true_w
        scale_y = 256 / true_h
        
        # Prepare prompt inputs with scaled coordinates.
        prompt_points = None
        if self.point is not None:
            scaled_point = (self.point[0] * scale_x, self.point[1] * scale_y)
            pt_tensor = torch.tensor([list(scaled_point)], device=device).unsqueeze(0)
            pt_label = torch.tensor([[1]], dtype=torch.long, device=device)
            prompt_points = (pt_tensor, pt_label)
        prompt_boxes = None
        if self.box is not None:
            scaled_box = (self.box[0] * scale_x, self.box[1] * scale_y,
                          self.box[2] * scale_x, self.box[3] * scale_y)
            box_tensor = torch.tensor([list(scaled_box)], device=device)
            prompt_boxes = box_tensor
        text_prompt = self.text_entry.get().strip()
        if text_prompt == "":
            text_prompt = None
        
        # Run model inference.
        with torch.no_grad():
            logits = model(image_tensor, prompt_points=prompt_points, prompt_boxes=prompt_boxes, text_prompt=text_prompt)
            if logits.shape[-2:] != (256, 256):
                logits = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Colorize the segmentation mask and then resize it back to true dimensions.
        color_mask = colorize_mask(pred_mask)
        color_mask = color_mask.resize((true_w, true_h), Image.NEAREST)
        
        # Update the input image with drawn prompts (annotations) in true coordinates.
        input_with_prompts = self.image.copy()
        draw = ImageDraw.Draw(input_with_prompts)
        if self.point is not None:
            r = 5
            draw.ellipse((self.point[0]-r, self.point[1]-r, self.point[0]+r, self.point[1]+r), fill=(255, 255, 0))
        if self.box is not None:
            draw.rectangle(self.box, outline=(0, 255, 255), width=2)
        self.tk_input = ImageTk.PhotoImage(input_with_prompts)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_input)
        
        # Update the predicted mask display.
        self.tk_pred = ImageTk.PhotoImage(color_mask)
        self.result_label.configure(image=self.tk_pred)
        self.result_label.image = self.tk_pred

###################### Run the application ######################
root = tk.Tk()
app = App(root)
root.mainloop()
