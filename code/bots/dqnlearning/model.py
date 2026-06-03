from __future__ import annotations

import torch
from torch import nn

# Number of feature channels written per cell by bot._encodeNeuralMap.
# If _encodeNeuralMap changes its feature set this constant must be updated
# to match — the encoder and model both rely on it.
MAP_CHANNELS = 7


class MapCnnEncoder(nn.Module):
    """
    Convolutional encoder for the agent's local neural map.

    Input:  (B, MAP_CHANNELS, H, W) — channel-first spatial tensor.
    Output: (B, embedDim)           — compact spatial embedding.

    AdaptiveAvgPool2d makes the network resolution-invariant: the same
    weights process a 15×15 window on small mazes and a 31×31 window on
    large ones without retraining, supporting the goal of scaling to bigger
    maps later.

    Architecture rationale:
      - Two conv layers extract local spatial features (walls, corridors,
        visited regions, goal sightings).
      - Global average pool to a fixed 4×4 grid collapses spatial dimensions
        in a translation-tolerant way, then a linear projection gives a
        fixed-length embedding regardless of input resolution.
    """

    def __init__(self, embedDim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Linear(64 * 4 * 4, int(embedDim)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return self.proj(h.flatten(1))


class DqnModel(nn.Module):
    """
    Q-network with an optional CNN head for spatial map processing.

    When mapShape is provided the stored state vector is expected to be laid
    out as:

        [ flat_features (flatDim floats) | map_flat (C*H*W floats, HWC order) ]

    The model splits at flatDim, reshapes the map portion to (B, C, H, W),
    runs it through MapCnnEncoder, then concatenates the embedding with the
    flat features before the MLP Q-value head.

    When mapShape is None the model behaves identically to the original
    flat-MLP implementation — no CNN weights are created.
    """

    def __init__(
        self,
        flatDim: int,
        hiddenSize: int,
        numActions: int = 4,
        mapShape: tuple[int, int, int] | None = None,
        mapEmbedDim: int = 0,
    ) -> None:
        super().__init__()
        self.flatDim = int(flatDim)
        self.mapShape = mapShape

        if mapShape is not None and mapEmbedDim > 0:
            self._mapEncoder: MapCnnEncoder | None = MapCnnEncoder(int(mapEmbedDim))
            mlpInputDim = self.flatDim + int(mapEmbedDim)
        else:
            self._mapEncoder = None
            mlpInputDim = self.flatDim

        self.mlp = nn.Sequential(
            nn.Linear(mlpInputDim, int(hiddenSize)),
            nn.ReLU(),
            nn.Linear(int(hiddenSize), int(hiddenSize)),
            nn.ReLU(),
            nn.Linear(int(hiddenSize), int(numActions)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x[:, : self.flatDim]

        if self._mapEncoder is not None and self.mapShape is not None:
            C, H, W = self.mapShape
            # Map stored as HWC-flattened; reshape to (B, C, H, W) for conv2d.
            map_chw = (
                x[:, self.flatDim :]
                .view(-1, H, W, C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            combined = torch.cat([flat, self._mapEncoder(map_chw)], dim=1)
        else:
            combined = flat

        return self.mlp(combined)
