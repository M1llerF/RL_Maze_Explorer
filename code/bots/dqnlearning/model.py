from __future__ import annotations

import torch
from torch import nn


class DqnModel(nn.Module):
    def __init__(self, inputDim: int, hiddenSize: int, numActions: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(inputDim), int(hiddenSize)),
            nn.ReLU(),
            nn.Linear(int(hiddenSize), int(hiddenSize)),
            nn.ReLU(),
            nn.Linear(int(hiddenSize), int(numActions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
