import cliport.models as models
from cliport.models.streams.two_stream_transport_lang_fusion import TwoStreamTransportLangFusion


class OneStreamTransportLangFusion(TwoStreamTransportLangFusion):
    """Transport (a.k.a) Place module with language features fused at the bottleneck"""

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)

        print(f"Transport FCN: {stream_one_fcn}")

    def transport(self, in_tensor, crop, l):
        logits = self.key_stream_one(in_tensor, l)
        kernel = self.query_stream_one(crop, l)
        return logits, kernel
