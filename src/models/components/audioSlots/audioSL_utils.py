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
        self.grid = self.grid.to(inputs.device)
        grid = self.embedding(self.grid).to(inputs.device)
        return inputs + grid

    def build_grid(self, resolution):
        ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


# TODO:
# renset34 : reduced stride, out size : 32 X 8
# transformer : 4 layers embeddings 1~n, query self-attention first and cross attention later, initail q dim is 4096 and learned
class Encoder(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        hid_dim: int = 64,
        enc_depth: int = 4,
    ):
        super().__init__()
        assert enc_depth > 2, "Depth must be larger than 2."

        self.resnet = self._prepare_backbone()

        convs = nn.ModuleList(
            [nn.Conv2d(3, hid_dim, 5, padding="same"), nn.ReLU()])
        for _ in range(enc_depth - 2):
            convs.extend(
                [nn.Conv2d(hid_dim, hid_dim, 5, padding="same"), nn.ReLU()])
        convs.append(nn.Conv2d(hid_dim, hid_dim, 5, padding="same"))
        self.convs = nn.Sequential(*convs)

        self.encoder_pos = SoftPositionEmbed(hid_dim, (img_size, img_size))
        self.layer_norm = nn.LayerNorm([img_size * img_size, hid_dim])
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim)
        )

    def _prepare_resnet(self):

        resnet34 = CustomResnet(BasicBlock, [3, 4, 6, 3])
        return resnet34

    def _prepare_transformer(self):
        pass

    def forward(self, x):
        x = self.convs(x)  # [B, D, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W ,D]
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


# TODO:
# Broadcast decoder
class Decoder(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        slot_dim: int = 64,
        dec_hid_dim: int = 64,
        dec_init_size: int = 8,
        dec_depth: int = 6,
    ):
        super().__init__()

        self.img_size = img_size
        self.dec_init_size = dec_init_size
        self.decoder_pos = SoftPositionEmbed(
            slot_dim, (dec_init_size, dec_init_size))

        D_slot = slot_dim
        D_hid = dec_hid_dim
        upsample_step = int(np.log2(img_size // dec_init_size))

        deconvs = nn.ModuleList()
        count_layer = 0
        for _ in range(upsample_step):
            deconvs.extend(
                [
                    nn.ConvTranspose2d(
                        D_hid if count_layer > 0 else D_slot,
                        D_hid,
                        5,
                        stride=(2, 2),
                        padding=2,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                ]
            )
            count_layer += 1

        for _ in range(dec_depth - upsample_step - 1):
            deconvs.extend(
                [
                    nn.ConvTranspose2d(
                        D_hid if count_layer > 0 else D_slot, D_hid, 5, stride=(1, 1), padding=2
                    ),
                    nn.ReLU(),
                ]
            )
            count_layer += 1

        deconvs.append(nn.ConvTranspose2d(
            D_hid, 4, 3, stride=(1, 1), padding=1))
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, x):
        """Broadcast slot features to a 2D grid and collapse slot dimension."""
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1).unsqueeze(2)
        x = x.repeat((1, self.dec_init_size, self.dec_init_size, 1))
        x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)
        x = self.deconvs(x)
        x = x[:, :, : self.img_size, : self.img_size]
        x = x.permute(0, 2, 3, 1)
        return x
