import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from .losses import Loss


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()


class InfoNCE(Loss):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos_mask
        loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
        return -loss.mean()


class TripletLoss(Loss):
    def __init__(self, tau):
        super(TripletLoss, self).__init__()
        self.tau = tau
    def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
        similarities = _similarity(anchor, sample)
        pos_similarity = torch.max(similarities * pos_mask + (1 - pos_mask) * -1e6, dim=1)[0]
        neg_similarity = torch.min(similarities * neg_mask + (1 - neg_mask) * 1e6, dim=1)[0]
        loss = torch.clamp(pos_similarity - neg_similarity + self.tau, min=0)
        loss = loss.mean()
        return loss


def choose_lossf(loss_method_name,tau):
    if loss_method_name == "InfoNCE":
        return InfoNCE(tau)
    elif loss_method_name == "Triplet":
        return TripletLoss(tau)    