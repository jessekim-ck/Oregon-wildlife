import torch
import torch.nn as nn


def multi_BCE_loss(scores, labels):
    cost_list = list()
    for i in range(20):
        loss = nn.BCELoss()(torch.sigmoid(scores[:, i]), (labels == i).float())
        cost_list.append(loss)
    cost = torch.sum(torch.stack(cost_list))
    return cost
