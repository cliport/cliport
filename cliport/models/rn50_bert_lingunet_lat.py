import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import cliport.utils.utils as utils
from transformers import DistilBertTokenizer, DistilBertModel
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.models.core.unet import Up

from cliport.models.core import fusion
from cliport.models.core.fusion import FusionConvLat


class RN50BertLingUNetLat(nn.Module):
    """ ImageNet RN50 & Bert with U-Net skip connections """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super(RN50BertLingUNetLat, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.input_dim = 2048
        self.cfg = cfg
        self.batchnorm = self.cfg['train']['batchnorm']
        self.lang_fusion_type = self.cfg['train']['lang_fusion_type']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.device = device
        self.preprocess = preprocess

        self._load_vision_fcn()
        self._load_lang_enc()
        self._build_decoder()

    def _load_vision_fcn(self):
        resnet50 = models.resnet50(pretrained=True)
        modules = list(resnet50.children())[:-2]

        self.stem = nn.Sequential(*modules[:4])
        self.layer1 = modules[4]
        self.layer2 = modules[5]
        self.layer3 = modules[6]
        self.layer4 = modules[7]

    def _load_lang_enc(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_fc = nn.Linear(768, 1024)

        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

    def _build_decoder(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear)
        self.lat_fusion1 = FusionConvLat(input_dim=1024+512, output_dim=512)

        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)
        self.lat_fusion2 = FusionConvLat(input_dim=512+256, output_dim=256)

        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)
        self.lat_fusion3 = FusionConvLat(input_dim=256+128, output_dim=128)

        self.layer1 =  nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion4 = FusionConvLat(input_dim=128+64, output_dim=64)

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion5 = FusionConvLat(input_dim=64+32, output_dim=32)

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.lat_fusion6 = FusionConvLat(input_dim=32+16, output_dim=16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, self.output_dim, kernel_size=1)
        )

    def resnet50(self, x):
        im = []
        for layer in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            im.append(x)
        return x, im

    def encode_image(self, img):
        with torch.no_grad():
            img_encoding, img_im = self.resnet50(img)
        return img_encoding, img_im

    def encode_text(self, x):
        with torch.no_grad():
            inputs = self.tokenizer(x, return_tensors='pt')
            input_ids, attention_mask = inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)
            text_embeddings = self.text_encoder(input_ids, attention_mask)
            text_encodings = text_embeddings.last_hidden_state.mean(1)
        text_feat = self.text_fc(text_encodings)
        text_mask = torch.ones_like(input_ids) # [1, max_token_len]
        return text_feat, text_embeddings.last_hidden_state, text_mask

    def forward(self, x, lat, l):
        x = self.preprocess(x, dist='clip')

        in_type = x.dtype
        in_shape = x.shape
        x = x[:,:3]  # select RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        l_enc, l_emb, l_mask = self.encode_text(l)
        l_input = l_emb if 'word' in self.lang_fusion_type else l_enc
        l_input = l_input.to(dtype=x.dtype)

        assert x.shape[1] == self.input_dim
        x = self.conv1(x)

        x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        x = self.up1(x, im[-2])
        x = self.lat_fusion1(x, lat[-6])

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        x = self.up2(x, im[-3])
        x = self.lat_fusion2(x, lat[-5])

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        x = self.up3(x, im[-4])
        x = self.lat_fusion3(x, lat[-4])

        x = self.layer1(x)
        x = self.lat_fusion4(x, lat[-3])

        x = self.layer2(x)
        x = self.lat_fusion5(x, lat[-2])

        x = self.layer3(x)
        x = self.lat_fusion6(x, lat[-1])

        x = self.conv2(x)
        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode='bilinear')
        return x