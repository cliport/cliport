import numpy as np
import torch
import torch.nn.functional as F

from cliport.models.core.attention import Attention
import cliport.models as models
import cliport.models.core.fusion as fusion


class TwoStreamAttentionLangFusion(Attention):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device, clip_model):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device, clip_model)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess, self.clip_rn50)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, l):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_tens:
            lgts = self.attend(x, lang_goal)
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


class TwoStreamAttentionLangFusionLat(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device, clip_model):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device, clip_model)

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        return x



class TwoStreamAttentionRotationLangFusionLat(TwoStreamAttentionLangFusion):
    """Attention module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device, clip_model):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device, clip_model)

        self.crop_size = 64
        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

    def _build_nets(self):
        # Initialize fully convolutional Residual Network with 43 layers and
        # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
        stream_one_backbone, stream_two_backbone = self.stream_fcn
        stream_one_model = models.names[stream_one_backbone]
        stream_two_model = models.names[stream_two_backbone]

        in_shape = (64, 64, 6)
        self.attn_stream_one = stream_one_model(in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(in_shape, 1, self.cfg, self.device, self.preprocess,  self.clip_rn50)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn Rotation Stream One: {stream_one_backbone}, Stream Two: {stream_two_backbone}")

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, in_img, p, lang_goal, softmax=True):
        """Forward pass."""
        in_data = np.pad(in_img, self.padding, mode='constant')
        # in_data = self.preprocess(in_data)
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float).cuda()

        # Pivot
        # pv = np.array(in_data.shape[1:3]) // 2
        pv = np.array([p[0], p[1]]) + self.pad_size

        # crop
        hcrop = self.crop_size//2
        in_tens = in_tens[:, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop, :]

        # Rotate input.
        center = (hcrop, hcrop)
        in_tens = in_tens.permute(0, 3, 1, 2)
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=center)

        # Forward pass.
        logits = []
        for x in in_tens:
            logits.append(self.attend(x, lang_goal))
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        # c0 = self.padding[:2, 0]
        # c1 = c0 + in_img.shape[:2]
        # logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # D H W C
        #print("Logits: ", logits.shape)
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output
