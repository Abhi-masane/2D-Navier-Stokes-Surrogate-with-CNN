import torch, torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SurrogateDataset
from model import MiniUNet

torch.manual_seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev)

tr = DataLoader(SurrogateDataset("../dataset/manifest.json", "train"), 8, shuffle=True)
va = DataLoader(SurrogateDataset("../dataset/manifest.json", "val"), 8)
model = MiniUNet().to(dev)
opt = torch.optim.Adam(model.parameters(), 1e-3)
mse = nn.MSELoss()
best = 1e9
for ep in range(150):
    model.train(); tl = 0
    for x, y, mk in tr:
        x, y, mk = x.to(dev), y.to(dev), mk.to(dev)
        opt.zero_grad(); pr = model(x)
        loss = mse(pr * mk, y * mk); loss.backward(); opt.step(); tl += loss.item()
    model.eval(); vl = 0
    with torch.no_grad():
        for x, y, mk in va:
            x, y, mk = x.to(dev), y.to(dev), mk.to(dev)
            vl += mse(model(x) * mk, y * mk).item()
    tl, vl = tl / len(tr), vl / len(va)
    print(f"ep {ep:3d} train {tl:.5f} val {vl:.5f}")
    if vl < best:
        best = vl; torch.save(model.state_dict(), "../results/best.pt")
print("best val loss:", best)
