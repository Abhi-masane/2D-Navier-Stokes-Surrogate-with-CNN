import torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset import SurrogateDataset
from model import MiniUNet

dev = torch.device("cpu")
ds = SurrogateDataset("../dataset/manifest.json", "test")
model = MiniUNet(); model.load_state_dict(torch.load("../results/best.pt", map_location=dev))
model.eval()

errs_u, errs_p = [], []
with torch.no_grad():
    for i in range(len(ds)):
        x, y, mk = ds[i]
        pr = model(x[None])[0] * mk[0]
        yt = y * mk[0]
        errs_u.append((torch.norm(pr[:2] - yt[:2]) / torch.norm(yt[:2])).item())
        errs_p.append((torch.norm(pr[2] - yt[2]) / torch.norm(yt[2])).item())
        if i == 0: save = (x, y, pr)

print(f"TEST relative L2  velocity: {np.mean(errs_u)*100:.2f}%  pressure: {np.mean(errs_p)*100:.2f}%")

x, y, pr = save
fig, ax = plt.subplots(3, 3, figsize=(14, 8))
for r, ch, t in zip(range(3), range(3), ["ux", "uy", "p"]):
    for c, arr, tt in zip(range(3), [y[ch], pr[ch], (pr[ch] - y[ch]).abs()],
                          ["true", "pred", "abs err"]):
        im = ax[r, c].pcolormesh(arr.numpy(), cmap="coolwarm", shading="auto")
        ax[r, c].set_aspect("equal"); ax[r, c].set_title(f"{t} {tt}")
        fig.colorbar(im, ax=ax[r, c])
plt.tight_layout(); plt.savefig("../results/test_case.png", dpi=120)
print("saved ../results/test_case.png")
