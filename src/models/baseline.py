import torch
import torch.nn as nn

from torchvision import transforms

from src.models import BaseModel
from src.backbones import EfficientNet
from src.datasets import BaseDataset


class BaseLine(BaseModel):

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
        cost = nn.CrossEntropyLoss()(x, cls_ids.cuda())
        with torch.no_grad():
            preds = {
                "cls_ids": cls_ids.cpu().numpy(),
                "cls_ids_pred": torch.argmax(x, dim=1).cpu().numpy()
            }
        return cost, preds
