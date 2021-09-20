from cliport.models.resnet import ResNet43_8s
from cliport.models.clip_wo_skip import CLIPWithoutSkipConnections

from cliport.models.rn50_bert_unet import RN50BertUNet
from cliport.models.rn50_bert_lingunet import RN50BertLingUNet
from cliport.models.rn50_bert_lingunet_lat import RN50BertLingUNetLat
from cliport.models.untrained_rn50_bert_lingunet import UntrainedRN50BertLingUNet

from cliport.models.clip_unet import CLIPUNet
from cliport.models.clip_lingunet import CLIPLingUNet

from cliport.models.resnet_lang import ResNet43_8s_lang

from cliport.models.resnet_lat import ResNet45_10s
from cliport.models.clip_unet_lat import CLIPUNetLat
from cliport.models.clip_lingunet_lat import CLIPLingUNetLat
from cliport.models.clip_film_lingunet_lat import CLIPFilmLingUNet


names = {
    # resnet
    'plain_resnet': ResNet43_8s,
    'plain_resnet_lang': ResNet43_8s_lang,

    # without skip-connections
    'clip_woskip': CLIPWithoutSkipConnections,

    # unet
    'clip_unet': CLIPUNet,
    'rn50_bert_unet': RN50BertUNet,

    # lingunet
    'clip_lingunet': CLIPLingUNet,
    'rn50_bert_lingunet': RN50BertLingUNet,
    'untrained_rn50_bert_lingunet': UntrainedRN50BertLingUNet,

    # lateral connections
    'plain_resnet_lat': ResNet45_10s,
    'clip_unet_lat': CLIPUNetLat,
    'clip_lingunet_lat': CLIPLingUNetLat,
    'clip_film_lingunet_lat': CLIPFilmLingUNet,
    'rn50_bert_lingunet_lat': RN50BertLingUNetLat,
}
