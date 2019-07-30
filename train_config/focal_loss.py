import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0, device: torch.device = torch.device('cuda:0')) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: torch.Tensor = torch.tensor(gamma, device=device)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
