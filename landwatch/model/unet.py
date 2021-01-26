import math

import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class DownScalingBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.downscaling_block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels=in_channels,
                      out_channels=out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downscaling_block(x)


class UNet(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bilinear = bilinear

        left_layers = [pow(2, i) for i in range(6, 11)]

        self.left, self.right = [DownScalingBlock(self.in_channels, 64)], []

        self.left.extend([
            *[DownScalingBlock(left_layers[i],
                               left_layers[i + 1]) for i in range(len(left_layers) - 1)]
        ])

        self.right.extend([
            ConvBlock(512 + 256, 256),
            ConvBlock(256 + 128, 128),
            ConvBlock(128 + 64, 64)
        ])

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.output = nn.Sequential(
            nn.Conv2d(64, self.out_channels, kernel_size=1),
            nn.Upsample(scale_factor=2),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.left[0](x)

        conv2 = self.left[1](conv1)

        conv3 = self.left[2](conv2)

        x = self.left[3](conv3)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.right[0](x)
        x = self.upsample(x)

        x = torch.cat([x, conv2], dim=1)

        x = self.right[1](x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.right[2](x)

        x = self.output(x)

        return x
