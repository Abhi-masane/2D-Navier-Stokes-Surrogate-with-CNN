import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SurrogateDataset(Dataset):
    def __init__(self, manifest_path, split):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.data_dir = os.path.dirname(os.path.abspath(manifest_path))
        self.cases = [case for case in manifest["cases"] if case["split"] == split]
        self.re_max = manifest["re_max"]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        case_path = os.path.join(self.data_dir, case["file"])

        with np.load(case_path) as data:
            mask = data["mask"].astype(np.float32)[None, :, :]
            re_value = np.float32(case["Re"] / self.re_max)
            re_map = np.full(mask.shape, re_value, dtype=np.float32)

            inputs = np.concatenate((mask, re_map), axis=0)
            targets = np.stack((data["ux"], data["uy"], data["p"]), axis=0)
            targets = targets.astype(np.float32)

        return (
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(mask),
        )
