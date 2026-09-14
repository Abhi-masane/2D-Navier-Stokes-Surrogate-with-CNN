import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Keep this attribute name so older checkpoints still load.
        self.b = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.b(x)


class MiniUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=3, base=32):
        super().__init__()

        # Names are kept short here because they are also part of the saved state_dict.
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.b = ConvBlock(base * 2, base * 4)

        self.u2 = nn.Conv2d(base * 4, base * 2, kernel_size=1)
        self.d2 = ConvBlock(base * 4, base * 2)

        self.u1 = nn.Conv2d(base * 2, base, kernel_size=1)
        self.d1 = ConvBlock(base * 2, base)

        self.f = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        enc1 = self.e1(x)
        enc2 = self.e2(F.max_pool2d(enc1, kernel_size=2))
        center = self.b(F.max_pool2d(enc2, kernel_size=2))

        up2 = self.u2(center)
        up2 = F.interpolate(
            up2,
            size=enc2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec2 = self.d2(torch.cat((up2, enc2), dim=1))

        up1 = self.u1(dec2)
        up1 = F.interpolate(
            up1,
            size=enc1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec1 = self.d1(torch.cat((up1, enc1), dim=1))

        return self.f(dec1)
