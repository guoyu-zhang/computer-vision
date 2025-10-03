import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Residual Convolutional Block
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out += residual
        out = self.relu(out)
        return out

# ------------------------------
# Attention Block for Skip Connections
# ------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: channels in gating signal (from decoder)
        F_l: channels in encoder feature map (skip connection)
        F_int: intermediate channel number (usually F_l//2)
        """
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal from decoder (up-sampled feature)
        # x: encoder feature map (skip connection)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ------------------------------
# UpSampling Block with Attention and Residual Block
# ------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.use_attention = use_attention
        if use_attention:
            # Attention: gating signal channels = out_channels, encoder feature channels = out_channels
            self.attention = AttentionBlock(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        # After concatenation, the channel count becomes out_channels*2
        self.res_block = ResidualBlock(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # padding operation for size mismatch cases
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        if self.use_attention:
            skip = self.attention(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        x = self.res_block(x)
        return x

# ------------------------------
# Enhanced UNet with Residual Blocks and Attention
# ------------------------------
class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, features=[64, 128, 256, 512]):
        super(EnhancedUNet, self).__init__()
        # Encoder
        self.encoder1 = ResidualBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ResidualBlock(features[2], features[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck = ResidualBlock(features[3], features[3]*2)
        # Decoder
        self.up4 = UpBlock(in_channels=features[3]*2, out_channels=features[3], use_attention=True)
        self.up3 = UpBlock(in_channels=features[3], out_channels=features[2], use_attention=True)
        self.up2 = UpBlock(in_channels=features[2], out_channels=features[1], use_attention=True)
        self.up1 = UpBlock(in_channels=features[1], out_channels=features[0], use_attention=True)
        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        # Bottleneck
        b = self.bottleneck(p4)
        # Decoder path with skip connections
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = self.final_conv(d1)
        return out