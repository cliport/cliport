"""Attention module."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.models as models
from cliport.utils import utils


class Attention(nn.Module):
    """Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__()
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.padding = np.zeros((3, 2), dtype=int)
        max_dim = np.max(in_shape[:2])
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        self.attn_stream = models.names[stream_one_fcn](self.in_shape, 1, self.cfg, self.device)
        print(f"Attn FCN: {stream_one_fcn}")

    def attend(self, x):
        return self.attn_stream(x)

    def forward(self, inp_img, softmax=True):
        """Forward pass."""
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device) # [B W H 6]

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_tens:
            lgts = self.attend(x)
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output