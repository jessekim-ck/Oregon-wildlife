import os
import time
import argparse
from collections import defaultdict

import torch
import torch.nn as nn

import numpy as np

from .utils import draw_cost_curve


class Trainer:

    def __init__(self, model: nn.Module, args: argparse.Namespace, logging=True):
        self.args = args
        self.model = model
        self.model.cuda()
        self.logging = logging

    def train(self):
        self.best_accuracy = 0
        self.model_name = f"{self.model.__class__.__name__}_{str(time.time()).split('.')[0]}"

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 70], gamma=0.1)

        train_dataloader = self.model.get_train_dataloader()
        val_dataloader = self.model.get_test_dataloader()

        self.write_log(f"Training log for {self.model_name}", mode="w")
        self.write_log("")
        for key, val in self.args.__dict__.items():
            self.write_log(f"{key}: {val}")
        self.write_log("")

        train_cost_dict = dict()
        test_cost_dict = dict()
        accuracy_dict = dict()

        for epoch in range(self.args.epochs):
            self.write_log(f"Epoch {epoch + 1:03d}/{self.args.epochs:03d} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            train_cost = self.train_epoch(train_dataloader)
            self.write_log("")
            train_cost_dict[epoch + 1] = train_cost

            if (epoch + 1) % 10 == 0:
                test_cost, accuracy = self.evaluate(val_dataloader)
                test_cost_dict[epoch + 1] = test_cost
                accuracy_dict[epoch + 1] = accuracy
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    torch.save(self.model.state_dict(), f"results/weights/{self.model_name}.pt")
                    self.write_log("Saved best model!")
                self.write_log("")

            draw_cost_curve(train_cost_dict, test_cost_dict, accuracy_dict, self.model_name)

    def train_unsup(self):
        self.best_accuracy = 0
        self.model_name = f"{self.model.__class__.__name__}_{str(time.time()).split('.')[0]}"

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 70], gamma=0.1)

        train_dataloader = self.model.get_train_dataloader()
        val_dataloader = self.model.get_test_dataloader()

        self.write_log(f"Unsupervised Training log for {self.model_name}", mode="w")
        self.write_log("")
        for key, val in self.args.__dict__.items():
            self.write_log(f"{key}: {val}")
        self.write_log("")

        train_cost_dict = dict()
        test_cost_dict = dict()
        accuracy_dict = dict()

        for epoch in range(self.args.epochs):
            self.write_log(f"Epoch {epoch + 1:03d}/{self.args.epochs:03d} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            self.write_log("Train 3 epochs on training set...")
            for _ in range(5):
                self.train_epoch(train_dataloader)
            self.write_log("")

            self.write_log("Train on pseudo-training set...")
            pseudo_train_dataloader = self.model.get_pseudo_train_dataloader()
            for _ in range(15):
                train_cost = self.train_epoch(pseudo_train_dataloader)
            self.write_log("")
            train_cost_dict[epoch + 1] = train_cost

            test_cost, accuracy = self.evaluate(val_dataloader)
            test_cost_dict[epoch + 1] = test_cost
            accuracy_dict[epoch + 1] = accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                torch.save(self.model.state_dict(), f"results/weights/{self.model_name}.pt")
                self.write_log("Saved best model!")
            self.write_log("")

            draw_cost_curve(train_cost_dict, test_cost_dict, accuracy_dict, self.model_name)

    def train_epoch(self, dataloader):
        batch_cost_list = list()
        epoch_cost_list = list()
        
        self.model.train()
        for batch_idx, data in enumerate(dataloader):
            cost, preds = self.model.get_cost(data)
            self.optimizer.zero_grad()
            cost.backward()
            batch_cost_list.append(cost.item())
            epoch_cost_list.append(cost.item())
            self.optimizer.step()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
                self.write_log(f"Batch: {batch_idx + 1:04d}/{len(dataloader):04d} | Cost: {np.mean(batch_cost_list):.4f}")
                batch_cost_list = list()

        # self.scheduler.step()
        return np.mean(epoch_cost_list)

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            cost_list = list()
            cls_ids_list = list()
            cls_ids_pred_list = list()
            for data in dataloader:
                cost, preds = self.model.get_cost(data)
                cls_ids_list.extend(preds["cls_ids"])
                cls_ids_pred_list.extend(preds["cls_ids_pred"])
                cost_list.append(cost.item())
            
            cls_ids_list = np.array(cls_ids_list)
            cls_ids_pred_list = np.array(cls_ids_pred_list)
            accuracy = np.mean(cls_ids_list == cls_ids_pred_list)

        self.write_log(f"Test cost: {np.mean(cost_list):.4f} | Accuracy: {accuracy:.4f}")

        return np.mean(cost_list), accuracy
                
    def write_log(self, msg, mode="a"):
        if self.logging:
            with open(f"results/logs/{self.model_name}.log", mode) as log:
                log.write(f"{msg}\n")
        print(msg)
