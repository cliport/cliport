import torch
import torch.nn as nn
import torch.nn.functional as F

import cliport.utils.utils as utils
from transformers import DistilBertTokenizer, DistilBertModel
from cliport.models.core import fusion
from cliport.models.resnet import ConvBlock, IdentityBlock


class ResNet43_8s_lang(nn.Module):
    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super(ResNet43_8s_lang, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_shape[-1]
        self.output_dim = output_dim
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']
        self.lang_fusion_type = self.cfg['train']['lang_fusion_type']
        self.preprocess = preprocess

        self._make_layers()

    def _make_layers(self):
        self.conv1 = nn.Sequential(
            # conv1
            nn.Conv2d(self.input_dim, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64) if self.batchnorm else nn.Identity(),
            nn.ReLU(True),

            # fcn
            ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            ConvBlock(64, [128, 128, 128], kernel_size=3, stride=2, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            ConvBlock(128, [256, 256, 256], kernel_size=3, stride=2, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),

            ConvBlock(256, [512, 512, 512], kernel_size=3, stride=2, batchnorm=self.batchnorm),
            IdentityBlock(512, [512, 512, 512], kernel_size=3, stride=1, batchnorm=self.batchnorm),
        )


        # decoders
        self.decoder1 = nn.Sequential(
            ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.decoder2 = nn.Sequential(
            ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.decoder3 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.conv2 = nn.Sequential(
            # conv2
            ConvBlock(64, [16, 16, self.output_dim], kernel_size=3, stride=1,
                      final_relu=False, batchnorm=self.batchnorm),
            IdentityBlock(self.output_dim, [16, 16, self.output_dim], kernel_size=3, stride=1,
                          final_relu=False, batchnorm=self.batchnorm),
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_fc = nn.Linear(768, 1024)

        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 256)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 128)

    def encode_text(self, l):
        with torch.no_grad():
            inputs = self.tokenizer(l, return_tensors='pt')
            input_ids, attention_mask = inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)
            text_embeddings = self.text_encoder(input_ids, attention_mask)
            text_encodings = text_embeddings.last_hidden_state.mean(1)
        text_feat = self.text_fc(text_encodings)
        text_mask = torch.ones_like(input_ids) # [1, max_token_len]
        return text_feat, text_embeddings.last_hidden_state, text_mask

    def forward(self, x, l):
        x = self.preprocess(x, dist='transporter')

        # encode language
        l_enc, l_emb, l_mask = self.encode_text(l)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)

        x = self.conv1(x)

        x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        x = self.decoder1(x)

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        x = self.decoder2(x)

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        x = self.decoder3(x)

        out = self.conv2(x)

        return out