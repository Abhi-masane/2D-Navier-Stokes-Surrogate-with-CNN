import torch, torch.nn as nn, torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(o, o, 3, padding=1), nn.ReLU())
    def forward(self, x): return self.b(x)

class MiniUNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=3, base=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.b  = ConvBlock(base * 2, base * 4)
        self.u2 = nn.Conv2d(base * 4, base * 2, 1)   # channel reduction only
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.Conv2d(base * 2, base, 1)       # channel reduction only
        self.d1 = ConvBlock(base * 2, base)
        self.f  = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        a = self.e1(x)                          # full res  (65, 129)
        b = self.e2(nn.MaxPool2d(2)(a))         # half res  (32, 64)
        c = self.b(nn.MaxPool2d(2)(b))          # quarter   (16, 32)
        cu = F.interpolate(self.u2(c), size=b.shape[2:],
                           mode="bilinear", align_corners=False)
        d = self.d2(torch.cat([cu, b], 1))
        du = F.interpolate(self.u1(d), size=a.shape[2:],
                           mode="bilinear", align_corners=False)
        e = self.d1(torch.cat([du, a], 1))
        return self.f(e)
