import torch
import torch.nn as nn
import torch.nn.functional as F

class MAE(nn.Module):
    """Mean Absolute Error"""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction.lower()

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        if self.reduction == "none":
            return (prediton - target).abs()
        elif self.reduction == "sum":
            return torch.sum((prediton - target).abs())
        else:
            return torch.mean((prediton - target).abs())


class MAPE(nn.Module):
    """Mean Absolute Percentage Error"""

    def __init__(self, reduction="mean", epsilon: float = 1e-5):
        super().__init__()
        self.reduction = reduction.lower()
        self.epsilon = epsilon

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        target_abs = target.abs()
        target_abs[target_abs < self.epsilon] = self.epsilon
        if self.reduction == "none":
            return (prediton - target).abs() / target_abs
        elif self.reduction == "sum":
            return torch.sum((prediton - target).abs() / target_abs)
        else:
            return torch.mean((prediton - target).abs() / target_abs)


class MSE(nn.Module):
    """Mean Square Error"""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction.lower()

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        if self.reduction == "none":
            return torch.square(prediton - target)
        elif self.reduction == "sum":
            return torch.sum(torch.square(prediton - target))
        else:
            return torch.mean(torch.square(prediton - target))


class ClsLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        #             (N, c, L)/(N, c),       (N, L)/(N)
        return F.cross_entropy(prediton, target)


class AUXAcc2(nn.Module):
    def __int__(self):
        super().__int__()

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        # (N, c)->(N)
        pred = torch.argmax(prediton, dim=-1)
        n_right = torch.sum(pred == target)
        n_all = (target.shape[0])
        acc = n_right / n_all
        return acc, n_right, n_all


class AUXAcc3(nn.Module):
    def __int__(self):
        super().__int__()

    def forward(self, prediton: torch.Tensor, target: torch.Tensor):
        # (N, L, c)->(N, L)
        pred = torch.argmax(prediton, dim=-1)
        n_right = torch.sum(pred == target)
        n_all = target.shape[0] * target.shape[1]
        acc = n_right / n_all
        return acc, n_right, n_all
