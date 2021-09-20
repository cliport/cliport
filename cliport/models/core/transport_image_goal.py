import numpy as np
import cliport.models as models
from cliport.utils import utils

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransportImageGoal(nn.Module):
    """Transport module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        """Transport module for placing.
        Args:
          in_shape: shape of input image.
          n_rotations: number of rotations of convolving kernel.
          crop_size: crop size around pick argmax used as convolving kernel.
          preprocess: function to preprocess input images.
        """
        super().__init__()

        self.iters = 0
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        in_shape = np.array(in_shape)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        # Crop before network (default for Transporters CoRL 2020).
        self.kernel_shape = (self.crop_size, self.crop_size, self.in_shape[2])

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        model = models.names[stream_one_fcn]
        self.key_resnet = model(self.in_shape, self.output_dim, self.cfg, self.device)
        self.query_resnet = model(self.in_shape, self.kernel_dim, self.cfg, self.device)
        self.goal_resnet = model(self.in_shape, self.output_dim, self.cfg, self.device)
        print(f"Transport FCN: {stream_one_fcn}")

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:])
        return output

    def forward(self, inp_img, goal_img, p, softmax=True):
        """Forward pass."""

        # Input image.
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data.copy()).to(dtype=torch.float, device=self.device)
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        # Goal image.
        goal_tensor = np.pad(goal_img, self.padding, mode='constant')
        goal_shape = (1,) + goal_tensor.shape
        goal_tensor = goal_tensor.reshape(goal_shape)
        goal_tensor = torch.from_numpy(goal_tensor.copy()).to(dtype=torch.float, device=self.device)
        goal_tensor = goal_tensor.permute(0, 3, 1, 2)

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size
        hcrop = self.pad_size

        # Cropped input features.
        in_crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        in_crop = self.rotator(in_crop, pivot=pv)
        in_crop = torch.cat(in_crop, dim=0)
        in_crop = in_crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        # Cropped goal features.
        goal_crop = goal_tensor.repeat(self.n_rotations, 1, 1, 1)
        goal_crop = self.rotator(goal_crop, pivot=pv)
        goal_crop = torch.cat(goal_crop, dim=0)
        goal_crop = goal_crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        in_logits = self.key_resnet(in_tensor)
        goal_logits = self.goal_resnet(goal_tensor)
        kernel_crop = self.query_resnet(in_crop)
        goal_crop = self.goal_resnet(goal_crop)

        # Fuse Goal and Transport features
        goal_x_in_logits = goal_logits + in_logits # Mohit: why doesn't multiply work? :(
        goal_x_kernel = goal_crop + kernel_crop

        # TODO(Mohit): Crop after network. Broken for now
        # in_logits = self.key_resnet(in_tensor)
        # kernel_nocrop_logits = self.query_resnet(in_tensor)
        # goal_logits = self.goal_resnet(goal_tensor)

        # goal_x_in_logits = in_logits
        # goal_x_kernel_logits = goal_logits * kernel_nocrop_logits

        # goal_crop = goal_x_kernel_logits.repeat(self.n_rotations, 1, 1, 1)
        # goal_crop = self.rotator(goal_crop, pivot=pv)
        # goal_crop = torch.cat(goal_crop, dim=0)
        # goal_crop = goal_crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        return self.correlate(goal_x_in_logits, goal_x_kernel, softmax)

