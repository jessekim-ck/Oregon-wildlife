import torch
import torch.nn as nn

from src.models import BaseModel
from src.backbones import EfficientNet
from src.datasets import BaseDataset
from src.datasets import PseudoDataset


class PseudoNet(BaseModel):

    def __init__(self, args):
        super().__init__()
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
        return torch.sigmoid(x)

    def get_cost(self, data):
        paths, imgs, cls_ids = data
        x = self(imgs.cuda())

        cost = self_defined_multi_BCE_loss()(x, cls_ids.cuda())

        with torch.no_grad():
            cls_ids_pred = torch.argmax(x, dim=1).cpu().numpy()
            preds = {
                "paths": paths,
                "cls_ids": cls_ids,
                "cls_ids_pred": cls_ids_pred,
                "pred_scores": torch.max(x, dim=1).cpu().numpy()
            }
        
        return cost, preds

    def get_pseudo_train_dataloader(self):
        dataset = PseudoDataset(
            model=self,
            dataloader=self.get_test_dataloader(),
            transform=self.train_transform
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )

        return dataloader
