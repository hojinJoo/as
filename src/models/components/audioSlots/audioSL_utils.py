from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock


class CustomResnet(ResNet):
    def __init__(self, block, layers, **kwargs):
        super(CustomResnet, self).__init__(block, layers, **kwargs)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                               padding=2, bias=False)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.layer4[0].conv1.stride = 1
        self.layer4[0].downsample[0].stride = 1

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)

        return x


class SoftPositionEmbed(nn.Module):
    """Builds the soft position embedding layer with learnable projection.

    Args:
        hid_dim (int): Size of input feature dimension.
        resolution (tuple): Tuple of integers specifying width and height of grid.
    """

    def __init__(
        self,
        hid_dim: int = 64,
        resolution: Tuple[int, int] = (128, 128),
    ):
        super().__init__()
        self.embedding = nn.Linear(4, hid_dim, bias=True)
        self.grid = self.build_grid(resolution)

    def forward(self, inputs):
        self.grid = self.grid
        grid = self.embedding(self.grid)
        return inputs + grid

    def build_grid(self, resolution):
        ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

