import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from .utils import multi_focal_loss
from src.models import BaseModel
from src.backbones import EfficientNet
from src.datasets import BaseDataset
from src.datasets import PseudoDataset


class PseudoNetFocal(BaseModel):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = BaseDataset
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, value="random")
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.feature = EfficientNet.from_name("efficientnet-b0")
        out_channels = self.feature.out_channels

        self.fc = nn.Linear(out_channels, 20)
    
    def forward(self, x):
        x = self.feature(x)
        x = self.fc(x)
        return x
    
    def get_cost(self, data):
        paths, imgs, cls_ids = data
        x = self(imgs.cuda())
        cost = multi_focal_loss(x, cls_ids.cuda())
        with torch.no_grad():
            pred_scores, cls_ids_pred = torch.max(torch.sigmoid(x), dim=1)
            preds = {
                "paths": np.array(paths),
                "cls_ids": cls_ids.numpy(),
                "cls_ids_pred": cls_ids_pred.cpu().numpy(),
                "pred_scores": pred_scores.cpu().numpy()
            }
        return cost, preds

    def get_pseudo_train_dataloader(self):
        dataset = PseudoDataset(
            model=self,
            dataloader=self.get_test_dataloader(),
            transform=self.train_transform,
            th=0.9
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        return dataloader
