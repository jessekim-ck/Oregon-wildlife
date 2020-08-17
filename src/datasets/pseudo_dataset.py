import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class PseudoDataset(Dataset):

    def __init__(self, model, dataloader, transform, th=0.9):
        self.img_dir = "data"
        self.paths = list()
        self.cls_ids = list()
        self.cls_ids_gt = list()

        model.eval()
        with torch.no_grad():
            for data in dataloader:
                _, preds = model.get_cost(data)
                classified = (preds["pred_scores"] > th)
                self.paths.extend(list(preds["paths"][classified]))
                self.cls_ids.extend(list(preds["cls_ids_pred"][classified]))
                self.cls_ids_gt.extend(list(preds["cls_ids"][classified]))

        accuracy = np.mean(np.array(self.cls_ids) == np.array(self.cls_ids_gt))
        print(f"Pseudo train dataset - label accuracy: {accuracy:.4f}")
        
        self.transform = transform

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = self.process_img(path)
        cls_idx = self.cls_ids[idx]
        return path, img, cls_idx

    def process_img(self, img_path):
        return self.transform(Image.open(os.path.join(self.img_dir, img_path)).convert("RGB"))

    def __len__(self):
        return len(self.paths)
