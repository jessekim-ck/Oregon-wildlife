import torch
import torch.nn as nn


def multi_focal_loss(scores, labels, gamma=2):
    cost_list = list()
    for i in range(20):
        p = torch.sigmoid(scores[:, i])
        l = (labels == i).float()
        loss = torch.mean((1 - p).pow(gamma)*l*torch.log(p) + 
                p.pow(gamma)*(1 - l)*torch.log(1 - p))
        cost_list.append(loss)
    cost = -torch.sum(torch.stack(cost_list))
    return cost


def multi_BCE_loss(scores, labels):
    criterion = nn.BCELoss()
    cost_list = list()
    for i in range(20):
        loss = criterion(torch.sigmoid(scores[:, i]), (labels == i).float())
        cost_list.append(loss)
    cost = torch.sum(torch.stack(cost_list))
    return cost
