import torch.nn as nn
import torch.nn.functional as F

import cliport.utils.utils as utils
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.clip_lingunet_lat import CLIPLingUNetLat


class CLIPWithoutSkipConnections(CLIPLingUNetLat):
    """ CLIP RN50 with decoders (no skip connections) """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)

    def _build_decoder(self):
        self.layers = nn.Sequential(
            # conv1
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.UpsamplingBilinear2d(scale_factor=2),

            # decoder blocks
            ConvBlock(1024, [512, 512, 512], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(512, [512, 512, 512], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            ConvBlock(512, [512, 512, 512], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(512, [512, 512, 512], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),

            ConvBlock(512, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            ConvBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),

            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),

            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            ConvBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            # conv2
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, self.output_dim, kernel_size=1)
        )

    def forward(self, x):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  # select RGB
        x, _ = self.encode_image(x)
        x = x.to(in_type)

        assert x.shape[1] == self.input_dim
        x = self.layers(x)
        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return x