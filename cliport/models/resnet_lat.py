import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.utils.utils as utils

from cliport.models.resnet import ConvBlock, IdentityBlock

class ResNet45_10s(nn.Module):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super(ResNet45_10s, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_shape[-1]
        self.output_dim = output_dim
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']
        self.preprocess = preprocess

        self._make_layers()

    def _make_layers(self):
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if self.batchnorm else nn.Identity(),
            nn.ReLU(True),
        )

        # fcn
        self.layer1 = nn.Sequential(
            ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(64, [128, 128, 128], kernel_size=3, stride=2, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(128, [256, 256, 256], kernel_size=3, stride=2, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        self.layer4 = nn.Sequential(
            ConvBlock(256, [512, 512, 512], kernel_size=3, stride=2, batchnorm=self.batchnorm),
            IdentityBlock(512, [512, 512, 512], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        self.layer5 = nn.Sequential(
            ConvBlock(512, [1024, 1024, 1024], kernel_size=3, stride=2, batchnorm=self.batchnorm),
            IdentityBlock(1024, [1024, 1024, 1024], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )

        # head
        self.layer6 = nn.Sequential(
            ConvBlock(1024, [512, 512, 512], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(512, [512, 512, 512], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer7 = nn.Sequential(
            ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer8 = nn.Sequential(
            ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer9 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer10 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        # conv2
        self.conv2 = nn.Sequential(
            ConvBlock(32, [16, 16, self.output_dim], kernel_size=3, stride=1,
                      final_relu=False, batchnorm=self.batchnorm),
            IdentityBlock(self.output_dim, [16, 16, self.output_dim], kernel_size=3, stride=1,
                          final_relu=False, batchnorm=self.batchnorm)
        )

    def forward(self, x):
        x = self.preprocess(x, dist='transporter')
        in_shape = x.shape

        # encoder
        for layer in [self.conv1, self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            x = layer(x)

        # decoder
        im = []
        for layer in [self.layer6, self.layer7, self.layer8, self.layer9, self.layer10, self.conv2]:
            im.append(x)
            x = layer(x)

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return x, im