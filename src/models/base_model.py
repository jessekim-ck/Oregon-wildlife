import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.args = None
        self.dataset = callable
        self.train_transform = None
        self.test_transform = None

    def forward(self, x):
        raise NotImplementedError

    def get_cost(self, data):
        raise NotImplementedError

    def get_train_dataloader(self):
        dataset = self.dataset(
            "deepest_train.txt",
            self.train_transform,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        return dataloader

    def get_test_dataloader(self):
        dataset = self.dataset(
            "deepest_test.txt",
            self.test_transform,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size * 2,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        return dataloader

    def get_pseudo_train_dataloader(self):
        raise NotImplementedError
