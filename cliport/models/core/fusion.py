import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DotAttn(nn.Module):
    """ Dot-Attention """

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        raw_score = inp.bmm(h.unsqueeze(2))
        score = F.softmax(raw_score, dim=1)
        return score


class ScaledDotAttn(nn.Module):
    """ Scaled Dot-Attention """

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        raw_score = inp.bmm(h.unsqueeze(2)) / np.sqrt(h.shape[-1])
        score = F.softmax(raw_score, dim=1)
        return score


class Fusion(nn.Module):
    """ Base Fusion Class"""

    def __init__(self, input_dim=3):
        super().__init__()
        self.input_dim = input_dim

    def tile_x2(self, x1, x2, x2_proj=None):
        if x2_proj:
            x2 = x2_proj(x2)

        x2 = x2.unsqueeze(-1).unsqueeze(-1)
        x2 = x2.repeat(x1.shape[0], 1, x1.shape[-2], x1.shape[-1])
        return x2

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        raise NotImplementedError()


class FusionAdd(Fusion):
    """ x1 + x2 """

    def __init__(self, input_dim=3):
        super(FusionAdd, self).__init__(input_dim=input_dim)

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        return x1 + x2


class FusionMult(Fusion):
    """ x1 * x2 """

    def __init__(self, input_dim=3):
        super(FusionMult, self).__init__(input_dim=input_dim)

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        return x1 * x2


class FusionMax(Fusion):
    """ max(x1, x2) """

    def __init__(self, input_dim=3):
        super(FusionMax, self).__init__(input_dim=input_dim)

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        return torch.max(x1, x2)


class FusionConcat(Fusion):
    """ [x1; x2] """

    def __init__(self, input_dim=3):
        super(FusionConcat, self).__init__(input_dim=input_dim)

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        return torch.cat([x1, x2], dim=1)


class FusionConv(Fusion):
    """ 1x1 convs after [x1; x2] """

    def __init__(self, input_dim=3):
        super(FusionConv, self).__init__(input_dim=input_dim)
        self.conv = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=1, bias=False)
        )

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        x = torch.cat([x1, x2], dim=1)  # [B, 2C, H, W]
        x = self.conv(x)                # [B, C, H, W]
        return x


class FusionConvLat(Fusion):
    """ 1x1 convs after [x1; x2] for lateral fusion """

    def __init__(self, input_dim=3, output_dim=3):
        super(FusionConvLat, self).__init__(input_dim=input_dim)
        self.conv = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
        )

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        x = torch.cat([x1, x2], dim=1)  # [B, input_dim, H, W]
        x = self.conv(x)                # [B, output_dim, H, W]
        return x


## ------------- NOTE ----------------
## The following are various fusion types I experimented with.
## Most of them didn't work well ¯\_(ツ)_/¯
## But it doesn't mean there isn't a better way of
## doing lateral and multi-modal (language+vision) fusion.


class FusionFiLM(Fusion):
    """ FiLM (Perez et. al, https://arxiv.org/abs/1709.07871).
        Note: This is not used inside a Residual block before ReLU.
        I had a version this in UpBlock with FiLM, which didn't seem to work at all.
    """

    def __init__(self, input_dim=3, output_dim=3):
        super(FusionFiLM, self).__init__(input_dim=input_dim)

    def forward(self, x1, x2, gamma, beta):
        g = self.tile_x2(x1, x2, gamma)
        b = self.tile_x2(x1, x2, beta)
        return x1 * g + b


class FusionDeepConv(Fusion):
    """ Multi-Layer 1x1 convs after [x1; x2] """

    def __init__(self, input_dim=3):
        super(FusionDeepConv, self).__init__(input_dim=input_dim)
        self.conv = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=False),
        )

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        if x1.shape != x2.shape and len(x1.shape) != len(x2.shape):
            x2 = self.tile_x2(x1, x2, x2_proj)
        x = torch.cat([x1, x2], dim=1)    # [B, 2C, H, W]
        x = self.conv(x)                # [B, C, H, W]
        return x


class FusionMultWord(nn.Module):
    """ Product with weighted-sum of words """

    def __init__(self, input_dim=3):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        B, D, H, W = x1.shape
        x2_len = int(x2_mask.count_nonzero())

        weighted_x1 = torch.zeros_like(x1)
        for t in range(x2_len):
            x2_t = x2_proj(x2[:,t]) if x2_proj else x2[:,t]
            x2_t = x2_t.unsqueeze(-1).unsqueeze(-1).repeat(B, 1, H, W)
            weighted_x1 += x1 * x2_t
        weighted_x1 /= x2_len
        return weighted_x1


class FusionWordAttention(nn.Module):
    """ Word Attention """

    def __init__(self, input_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.dot_attn = DotAttn()

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        B, D, H, W = x1.shape
        x1_flat = x1.reshape(B, D, H*W)
        x2_len = int(x2_mask.count_nonzero())

        # TODO: batch this unrolling?
        weight_sum_x1_flat = torch.zeros_like(x1_flat)
        for t in range(x2_len):
            x2_t = x2_proj(x2[:,t]) if x2_proj else x2[:,t]
            x2_t = x2_t.repeat(B, 1)

            _, attn_x1 = self.dot_attn(x1_flat.transpose(1, 2), x2_t)
            weight_sum_x1_flat += x1_flat * attn_x1.transpose(1, 2)

        weight_sum_x1_flat /= x2_len
        x2 = weight_sum_x1_flat.reshape(B, D, H, W)
        return x2


class FusionSentenceAttention(nn.Module):
    """ Sentence Attention """

    def __init__(self, input_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.dot_attn = ScaledDotAttn()

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        B, D, H, W = x1.shape
        x1_flat = x1.reshape(B, D, H*W)

        x2_t = x2_proj(x2) if x2_proj else x2
        x2_t = x2_t.repeat(B, 1)

        _, attn_x1 = self.dot_attn(x1_flat.transpose(1, 2), x2_t)
        weight_sum_x1_flat = x1_flat * attn_x1.transpose(1, 2)

        x2 = weight_sum_x1_flat.reshape(B, D, H, W)
        return x2


class CrossModalAttention2d(nn.Module):
    """ Cross-Modal Attention. Adapted from: https://github.com/openai/CLIP/blob/main/clip/model.py#L56 """

    def __init__(self, spacial_dim=7, embed_dim=1024, num_heads=32,
                 output_dim=1024, lang_dim=512, lang_max_tokens=77):
        super().__init__()
        self.embed_dim = embed_dim
        self.lang_dim = lang_dim
        self.lang_max_tokens = lang_max_tokens
        self.num_heads = num_heads
        self.lang_proj = nn.Linear(self.lang_dim, embed_dim)
        self.vision_positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2, embed_dim) / embed_dim ** 0.5)
        self.lang_positional_embedding = nn.Parameter(torch.randn(lang_max_tokens, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)

    def forward(self, x, l, l_mask):
        # reshape vision features
        x_shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = x + self.vision_positional_embedding[:x.shape[0], None, :].to(x.dtype)  # (HW)NC

        # project language
        l = l.permute(1, 0, 2)
        l_shape = l.shape
        l = l.reshape(-1, self.lang_dim)
        l = self.lang_proj(l)
        l = l.reshape(l_shape[0], l_shape[1], self.embed_dim)
        l = l + self.lang_positional_embedding[:, None, :].to(l.dtype)

        # hard language mask
        l_len = int(l_mask.count_nonzero())
        l = l[:l_len]
        l = l.repeat(1, x.shape[1], 1)

        x, _ = F.multi_head_attention_forward(
            query=x, key=l, value=l,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 2, 0)
        x = x.reshape(x_shape)
        return x


class FusionMultiHeadedWordAttention(nn.Module):
    """ Multi-Headed Word Attention that uses Cross Modal Attention at different scales """

    def __init__(self, input_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.attn1 = CrossModalAttention2d(spacial_dim=7, embed_dim=1024, output_dim=1024)
        self.attn2 = CrossModalAttention2d(spacial_dim=14, embed_dim=512, output_dim=512)
        self.attn3 = CrossModalAttention2d(spacial_dim=28, embed_dim=256, output_dim=256)

        self.multi_headed_attns = {
            1024: self.attn1,
            512: self.attn2,
            256: self.attn3,
        }

    def forward(self, x1, x2, x2_mask=None, x2_proj=None):
        emb_dim = x1.shape[1]
        x = self.multi_headed_attns[emb_dim](x1, x2, x2_mask)
        return x


names = {
    'add': FusionAdd,
    'mult': FusionMult,
    'mult_word': FusionMultWord,
    'film': FusionFiLM,
    'max': FusionMax,
    'concat': FusionConcat,
    'conv': FusionConv,
    'deep_conv': FusionDeepConv,
    'word_attn': FusionWordAttention,
    'sent_attn': FusionSentenceAttention,
    'multi_headed_word_attn': FusionMultiHeadedWordAttention,
}

