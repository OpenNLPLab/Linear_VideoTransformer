# ------------------------------------------------------------------------
# Mostly a modified copy from timm (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
# ------------------------------------------------------------------------
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch import einsum
from torch import nn as nn
from einops import rearrange

class AfterReconstruction(nn.Identity):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        insert_control_point=False,
        use_cgate=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.insert_control_point = insert_control_point
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.n_segment = 8
        if insert_control_point:
            self.control_point = AfterReconstruction(dim)
            self.control_point_query = AfterReconstruction(dim)
            self.control_point_value = AfterReconstruction(dim)
        self.use_cgate = use_cgate
        if use_cgate:
            # self.q_gate = nn.Linear(head_dim, head_dim)
            # self.k_gate = nn.Linear(head_dim, head_dim)
            self.pgate = nn.Linear(3*head_dim, head_dim)
    def forward(self, x, num_frames, layer_idx=0, filename=None):
        if self.insert_control_point:
            x = self.control_point(x)
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.insert_control_point:
            k = self.control_point_query(k)
            v = self.control_point_value(v)
        if self.use_cgate:
            qk = torch.cat([q, k, v], 3)
            q = F.sigmoid(self.pgate(qk)) * q
            k = F.sigmoid(self.pgate(qk)) * k
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #############################save qk##################################
        save_qk = False

        if save_qk and filename is not None:
            filename = filename[0]
            # save_file = f"layer{layer_idx}_SpaceQk.npy".format(layer_idx=layer_idx)
            # save_dir = f"/mnt/lustre/liuzexiang/cache/Code/XVIT/videocosformer/output/xvit/qk_weights/{filename}"
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = f"{save_dir}/{save_file}"
            #
            # np.save(save_path, attn[0].cpu().detach().numpy())

            # save_file = f"layer{layer_idx}_q.npy".format(layer_idx=layer_idx)
            # save_dir = f"/mnt/lustre/share_data/liuzexiang/Data/ssv2/qk_weights/ssv2_xvit/q/{filename}"
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = f"{save_dir}/{save_file}"
            # np.save(save_path, q[0].cpu().detach().numpy())
            #
            # save_file = f"layer{layer_idx}_k.npy".format(layer_idx=layer_idx)
            # save_dir = f"/mnt/lustre/share_data/liuzexiang/Data/ssv2/qk_weights/ssv2_xvit/k/{filename}"
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = f"{save_dir}/{save_file}"
            # np.save(save_path, k[0].cpu().detach().numpy())
        ###############################################################
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            insert_control_point=False,
            # ADD FOR ORPE
            use_orpe=False,
            core_matrix=1,
            p_matrix=3,
            theta_type="a",
            theta_learned=True,
            householder_learned=False,
            orpe_dim=1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.insert_control_point = insert_control_point
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.n_segment = 8

        self.use_orpe = use_orpe
        if self.use_orpe:
            print("===================")
            print("use_orpe")
            self.orpe = Orpe(core_matrix, p_matrix, embedding_dim=head_dim,
                             theta_type=theta_type, theta_learned=theta_learned,
                             householder_learned=householder_learned, dim=orpe_dim)

    def abs_clamp(self, t):
        min_mag = 1e-4
        max_mag = 10000
        sign = t.sign()
        return t.abs_().clamp_(min_mag, max_mag) * sign

    def forward(self, x, num_frames):
        """
        intput shape: (b * f), (w * h), c
        """
        B, N, C = x.shape
        # qkv = (
        #     self.qkv(x)
        #     .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #     .permute(2, 0, 1, 3, 4)
        # )
        # split to qkv
        qkv = rearrange(self.qkv(x), 'b n (k c) -> k b n c', k=3)
        # split to multi head
        qkv = rearrange(qkv, 'k b n (h e) -> k b n h e', h=self.num_heads)
        # b, n, h, e
        q, k, v = qkv[0], qkv[1], qkv[2]
        # act
        q_ = F.relu(q)
        k_ = F.relu(k)
        # reshape
        q_ = rearrange(q_, '(b f) n h e -> b f n h e', f=num_frames)
        k_ = rearrange(k_, '(b f) n h e -> b f n h e', f=num_frames)
        v_ = rearrange(v, '(b f) n h e -> b f n h e', f=num_frames)
        if self.use_orpe:
            q_ = self.orpe(q_)
            k_ = self.orpe(k_)
        q_ = rearrange(q_, 'b f n h e -> b (f n) h e')
        k_ = rearrange(k_, 'b f n h e -> b (f n) h e')
        v_ = rearrange(v_, 'b f n h e -> b (f n) h e')

        ##### compute
        eps = 1e-4
        # b h n e, b h e n -> b h e e
        kv_ = torch.matmul(rearrange(k_, 'b n h e -> b h e n'),
                           rearrange(v_, 'b n h e -> b h n e'))

        # 分母
        # b n h e -> b 1 h e
        k_sum = torch.sum(k_, axis=1, keepdim=True)
        # b n h e, b 1 h e -> b n h e
        z_ = 1 / (torch.sum(torch.mul(q_, k_sum), axis=-1) + eps)

        attn_output = torch.matmul(q_.transpose(1, 2), kv_).transpose(1, 2)
        attn_output = torch.mul(attn_output, z_.unsqueeze(-1))

        # reshape
        attn_output = rearrange(attn_output, 'b (f n) h e -> b f n h e', f=num_frames)
        attn_output = rearrange(attn_output, 'b f n h e -> (b f) n (h e)')

        # outprojection
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return attn_output


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        insert_control_point=False,
        linear_attention=False
    ):
        super().__init__()
        self.insert_control_point = insert_control_point
        self.norm1 = norm_layer(dim)
        if not linear_attention:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                insert_control_point=insert_control_point,
            )
        else:
            self.attn = LinearAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, num_frames, layer_idx=0, filename=None):
        x = x + self.drop_path(self.attn(self.norm1(x), num_frames, layer_idx=layer_idx, filename=filename))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def get_sinusoid_encoding(n_position, d_hid):
    """ Sinusoid position encoding table """

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
