# ------------------------------------------------------------------------
# Mostly a modified copy from timm (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
# ------------------------------------------------------------------------

import logging
import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, build_model_with_cfg
from timm.models.helpers import build_model_with_cfg

from .transformer_block import Block
from .linear_block_v2 import LinearBlock
from einops import rearrange

_logger = logging.getLogger(__name__)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    # add
    "lvit_v2_base_patch16_224": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-lvitjx/lvit_v2_base.pth",
        num_classes=21843,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "lvit_v2_base_768_patch16_224": _cfg(
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-lvitjx/lvit_v2_base_768.pth",
            num_classes=21843,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
}

class OverlapPatchEmbed3D(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(
            self, img_size=224, in_chans=3, stride=14,
            patch_size=16, z_block_size=2, embed_dim=768, flatten=True
        ):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim,
            kernel_size=(z_block_size, patch_size, patch_size),
            stride=(1, stride, stride), padding=(0, patch_size // 2, patch_size // 2))
        self.H, self.W = (img_size + patch_size*2 - stride) // stride, (
                    img_size + patch_size*2 - stride) // stride
        # add cls
        self.num_patches = self.H * self.W
        self.flatten = flatten
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x)
        B, C, T, H, W = x.shape
        if self.flatten:
            x = x.flatten(3).transpose(1, 2).transpose(2, 3) ##B, T, D, C
        x = self.norm(x)
        return x, H, W

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = (img_size[0]+patch_size[0]*2-stride)//stride, (img_size[0]+patch_size[0]*2-stride) // stride
        # add cls
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # print(x.shape)
        x = self.proj(x)
        _, _, H, W = x.shape
        # print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class LinearChangeVisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_frames=8,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        use_orpe=False,
        orpe_dim=1,
        has_kv=False,
        stride=12,
        linear_attention=None,
        attention_type='full_time_space',
        use_3d=False
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.img_size = img_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.use_orpe = use_orpe
        self.orpe_dim = orpe_dim
        print(f"use_orpe {self.use_orpe}")
        print(f"orpe_dim {self.orpe_dim}")
        self.use_3d = use_3d
        if self.use_3d:
            self.patch_embed_3d = OverlapPatchEmbed3D(
                img_size=224,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.patch_embed_3d.proj.weight.data = torch.zeros_like(
                self.patch_embed_3d.proj.weight.data)
            num_patches = self.patch_embed_3d.num_patches
        else:
            self.patch_embed = OverlapPatchEmbed(img_size=img_size,
                                                patch_size=patch_size,
                                                stride=stride,
                                                in_chans=in_chans,
                                                embed_dim=embed_dim)

            num_patches = self.patch_embed.num_patches

        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(
        #     torch.zeros(1, num_patches, embed_dim)
        # )
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        control_flags = [True for _ in range(depth)]
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                LinearBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    has_kv=has_kv,
                    attention_type=attention_type,
                    insert_control_point=True
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        #trunc_normal_(self.pos_embed, std=0.02)
        #trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x, num_frames):
        """
        intput shape: (b * f), c, w, h
        """
        # print("===================")
        # print(f"num_frames {num_frames}")
        # print(f"self.num_frames {self.num_frames}")
        # print("===================")
        B = x.shape[0]
        # print(x.shape)
        if self.use_3d:
            x = x.view((-1, num_frames) + x.size()[-3:]) ##b,f,c,h,w
            x = x.transpose(1, 2)  #####b,c,f,h,w
            x, H, W = self.patch_embed_3d(x)
            x_last = x[:, -1, :, :]
            x = torch.cat((x, x_last), dim=1)
            x = x.view((-1,) + x.size()[2:])
        else:
            x, H, W = self.patch_embed(x)
        # print(x.shape)
        # cls_tokens = self.cls_token.expand(
        #     B, -1, -1
        # )  # stole cls_tokens impl from Phil Wang, thanks
        # # Interpolate positinoal embeddings
        # print(x.shape)
        # new_pos_embed = self.pos_embed
        # print(new_pos_embed.shape)

        # # after pos_embedding, dim is (B, (H * W), C)
        # x = x + new_pos_embed
        x = self.pos_drop(x)
        x = (
            x.view(
                x.size(0) // self.num_frames,
                self.num_frames,
                x.size(1),
                x.size(2),
            )
            + self.temp_embed
        )
        x = x.view(-1, x.size(2), x.size(3))
        ###add cls_token
        # x = x.view((-1, num_frames) + x.size()[-2:])
        # x = rearrange(x, 'b t s c -> b (t s) c')
        # cls_tokens = self.cls_token.expand(
        #     x.size(0), -1, -1
        # )
        # x = torch.cat((cls_tokens, x), dim=1)

        for idx, blk in enumerate(self.blocks):
            x = blk(x, H, W, num_frames)
        # x = x[:, 1:, :]
        # x = rearrange(x, 'b (t s) c -> (b t) s c', t=num_frames)
        # (b f) C
        x = self.norm(x)
        x = self.pre_logits(x)

        return x

    def forward(self, x, num_frames):
        x = self.forward_features(x, num_frames)
        # B, N, E
        x = self.head(x)
        # print(x.shape)
        return x.mean(dim=1)


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/lvit_jax/checkpoint.py#L224
    _logger.info(
        "Resized position embedding: %s to %s", posemb.shape, posemb_new.shape
    )
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info("Position embedding grid-size from %s to %s", gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=(gs_new, gs_new), mode="bilinear"
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, gs_new * gs_new, -1
    )
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    default_cfg = default_cfgs[variant]
    default_num_classes = default_cfg["num_classes"]
    default_img_size = default_cfg["input_size"][-1]

    num_classes = kwargs.pop("num_classes", default_num_classes)
    img_size = kwargs.pop("img_size", default_img_size)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    print('default cfg', default_cfg)
    model = build_model_with_cfg(
        LinearChangeVisionTransformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        # pretrained_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=False,
        **kwargs
    )
    model.default_cfg = default_cfg

    return model

#################### add
@register_model
def lvit_v2_base_patch16_224(pretrained=False, **kwargs):
    """ lvit-Base (lvit-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, has_kv=False, stride=14, **kwargs
    )
    model = _create_vision_transformer(
        "lvit_v2_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model

@register_model
def lvit_v2_base_768_patch16_224(pretrained=False, **kwargs):
    """ lvit-Base (lvit-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=8, has_kv=False, stride=14, **kwargs
    )
    model = _create_vision_transformer(
        "lvit_v2_base_768_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model
####################
