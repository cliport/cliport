import cliport.models as models
import cliport.models.core.fusion as fusion
from cliport.models.core.attention import Attention


class TwoStreamAttention(Attention):
    """Two Stream Attention (a.k.a Pick) module"""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)
        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x)
        x = self.fusion(x1, x2)
        return x


class TwoStreamAttentionLat(TwoStreamAttention):
    """Two Stream Attention (a.k.a Pick) module with lateral connections"""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def attend(self, x):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat)
        x = self.fusion(x1, x2)
        return x