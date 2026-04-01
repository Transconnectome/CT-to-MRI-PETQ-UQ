import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(ch)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x


class ResidualUNet25D(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        self.enc1 = EncoderBlock(in_ch, 16)
        self.enc2 = EncoderBlock(16, 32)
        self.enc3 = EncoderBlock(32, 64)

        self.pre_bottleneck = ConvBlock(64, 128)
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(6)])

        self.dec1 = DecoderBlock(128 + 64, 64)
        self.dec2 = DecoderBlock(64 + 32, 32)
        self.dec3 = DecoderBlock(32 + 16, 16)

        self.final = nn.Conv2d(16, out_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # x: [B, 3, 256, 256]

        s1 = self.enc1(x)              # [B, 16, 128, 128]
        s2 = self.enc2(s1)             # [B, 32,  64,  64]
        s3 = self.enc3(s2)             # [B, 64,  32,  32]

        x = self.pre_bottleneck(s3)    # [B, 128, 32, 32]
        x = self.res_blocks(x)         # [B, 128, 32, 32]

        x = torch.cat([x, s3], dim=1)  # [B, 192, 32, 32]
        x = self.dec1(x)               # [B, 64, 64, 64]

        x = torch.cat([x, s2], dim=1)  # [B, 96, 64, 64]
        x = self.dec2(x)               # [B, 32, 128, 128]

        x = torch.cat([x, s1], dim=1)  # [B, 48, 128, 128]
        x = self.dec3(x)               # [B, 16, 256, 256]

        x = self.final(x)              # [B, 1, 256, 256]
        return x