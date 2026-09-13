import json, os
import numpy as np, torch
from torch.utils.data import Dataset

class SurrogateDataset(Dataset):
    def __init__(self, manifest_path, split):
        m = json.load(open(manifest_path))
        self.base = os.path.dirname(os.path.abspath(manifest_path))
        self.items = [c for c in m["cases"] if c["split"] == split]
        self.re_max = m["re_max"]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        c = self.items[i]
        d = np.load(os.path.join(self.base, c["file"]))
        mask = d["mask"][None].astype(np.float32)                 # (1,H,W)
        re_map = np.full_like(mask, np.float32(c["Re"] / self.re_max))
        x = np.concatenate([mask, re_map], 0)                     # (2,H,W)
        y = np.stack([d["ux"], d["uy"], d["p"]]).astype(np.float32)  # (3,H,W)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)
