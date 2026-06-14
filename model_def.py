"""
3D U-Net (UNETR_Lite_v3_DS) with deep supervision.
Same architecture used during training -> must match the checkpoint exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = BasicBlock(in_ch, out_ch)
        self.down = nn.Conv3d(out_ch, out_ch, 2, 2)

    def forward(self, x):
        x = self.conv(x)
        return self.down(x), x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, 2)
        self.conv = BasicBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.pad(x, [
            0, skip.size(4) - x.size(4),
            0, skip.size(3) - x.size(3),
            0, skip.size(2) - x.size(2)
        ])
        return self.conv(torch.cat([skip, x], dim=1))


class UNETR_Lite_v3_DS(nn.Module):
    def __init__(self, in_channels=1, num_classes=5, base=24):
        super().__init__()

        self.enc1 = EncoderBlock(in_channels, base)
        self.enc2 = EncoderBlock(base, base * 2)
        self.enc3 = EncoderBlock(base * 2, base * 4)
        self.enc4 = EncoderBlock(base * 4, base * 8)

        self.bottleneck = BasicBlock(base * 8, base * 16)

        self.dec4 = DecoderBlock(base * 16, base * 8, base * 8)
        self.dec3 = DecoderBlock(base * 8, base * 4, base * 4)
        self.dec2 = DecoderBlock(base * 4, base * 2, base * 2)
        self.dec1 = DecoderBlock(base * 2, base, base)

        self.out_main = nn.Conv3d(base, num_classes, 1)
        self.out_aux = nn.Conv3d(base * 2, num_classes, 1)

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)

        b = self.bottleneck(x4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        main = self.out_main(d1)
        aux = F.interpolate(self.out_aux(d2), size=main.shape[2:], mode="trilinear", align_corners=False)
        return main, aux
