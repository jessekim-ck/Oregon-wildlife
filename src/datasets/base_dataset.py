import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):

    def __init__(self, data_path, transform):
        self.img_dir = "data"
        self.paths = list()
        self.classes = list()
        with open(os.path.join(self.img_dir, data_path), "r") as data:
            for line in data:
                self.paths.append(line[:-1])
                self.classes.append(line.split("/")[2])
        self.class_to_index = {cls: idx for idx, cls in enumerate(sorted(set(self.classes)))}
        self.transform = transform

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.process_img(path)
        cls_idx = self.class_to_index[self.classes[idx]]
        return path, img, cls_idx

    def process_img(self, img_path):
        return self.transform(Image.open(os.path.join(self.img_dir, img_path)).convert("RGB"))

    def __len__(self):
        return len(self.paths)
