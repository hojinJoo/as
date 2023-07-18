from typing import Callable, List, Optional, Type, Union
import torch
from torch import nn as nn
import torch.nn as nn
import torchvision
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck

# sample_wav = torch.randn(8000)
# after_stft = torch.stft(sample_wav, n_fft=512, win_length=512,
#                         hop_length=125, return_complex=True)
# after_stft = torch.abs(after_stft)
# input(after_stft.size())
# sample = after_stft.unsqueeze(0).repeat((64, 1, 1, 1))


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

        return x


def build_resnet() :
    return CustomResnet(BasicBlock, [3, 4, 6, 3])

# resnet34 = CustomResnet(BasicBlock, [3, 4, 6, 3])
# print(resnet34)
# # tmp = torch.rand(64, 256, 8, 8)
# # maxpool = nn.MaxPool2d(kernel_size=3, stride=1,
# #                        padding=1, dilation=1, ceil_mode=False)
# print(resnet34(sample).size())
# conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
#   padding=2, bias=False)
# print(conv1(sample).size())
