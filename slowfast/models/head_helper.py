#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_
from .transformers.transformer_block import Block
import torch.nn.functional as F

class VitHead(nn.Module):
    def __init__(self, embed_dim, cfg):
        super(VitHead, self).__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.temporal_encoder = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=cfg.TEMPORAL_HEAD.NUM_ATTENTION_HEADS,
                    attn_drop=cfg.TEMPORAL_HEAD.ATTENTION_PROBS_DROPOUT_PROB,
                    drop_path=cfg.TEMPORAL_HEAD.HIDDEN_DROPOUT_PROB,
                    drop=cfg.TEMPORAL_HEAD.HIDDEN_DROPOUT_PROB,
                    insert_control_point=False,
                )
                for _ in range(cfg.TEMPORAL_HEAD.NUM_HIDDEN_LAYERS)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.TEMPORAL_HEAD.HIDDEN_DIM),
            nn.Linear(cfg.TEMPORAL_HEAD.HIDDEN_DIM, cfg.TEMPORAL_HEAD.MLP_DIM),
            nn.GELU(),
            nn.Dropout(cfg.MODEL.DROPOUT_RATE),
            nn.Linear(cfg.TEMPORAL_HEAD.MLP_DIM, cfg.MODEL.NUM_CLASSES),
        )

    def forward(self, x, position_ids, num_frames):
        # temporal encoder (Longformer)
        B, D, E = x.shape
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        #x = self.temporal_encoder(x, num_frames)
        for block in self.temporal_encoder:
            x = block(x, num_frames)
        # MLP head
        x = self.mlp_head(x[:, 0])
        return x

class VVTHead(nn.Module):
    def __init__(self, embed_dim, cfg):
        super().__init__()
        self.seq_pool = False
        if self.seq_pool:
            self.attention_pool = nn.Linear(cfg.TEMPORAL_HEAD.HIDDEN_DIM, 1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.TEMPORAL_HEAD.HIDDEN_DIM),
            nn.Linear(cfg.TEMPORAL_HEAD.HIDDEN_DIM, cfg.TEMPORAL_HEAD.MLP_DIM),
            nn.GELU(),
            nn.Dropout(cfg.MODEL.DROPOUT_RATE),
            nn.Linear(cfg.TEMPORAL_HEAD.MLP_DIM, cfg.MODEL.NUM_CLASSES),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x, position_ids, num_frames):
        """
        B, F, E
        """
        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x.mean(dim=1)
        x = self.mlp_head(x)

        return x
