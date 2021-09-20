import torch.nn as nn
import torchvision.models as models

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from cliport.models.core import fusion
from cliport.models.rn50_bert_lingunet import RN50BertLingUNet


class UntrainedRN50BertLingUNet(RN50BertLingUNet):
    """ Untrained ImageNet RN50 & Bert with U-Net skip connections """

    def __init__(self, input_shape, output_dim, cfg, device, preprocess):
        super().__init__(input_shape, output_dim, cfg, device, preprocess)

    def _load_vision_fcn(self):
        resnet50 = models.resnet50(pretrained=False)
        modules = list(resnet50.children())[:-2]

        self.stem = nn.Sequential(*modules[:4])
        self.layer1 = modules[4]
        self.layer2 = modules[5]
        self.layer3 = modules[6]
        self.layer4 = modules[7]

    def _load_lang_enc(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # only Tokenizer is pre-trained
        distilbert_config = DistilBertConfig()
        self.text_encoder = DistilBertModel(distilbert_config)

        self.text_fc = nn.Linear(768, 1024)

        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 512 if 'word' in self.lang_fusion_type else 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)
