# ------------------------------------------------------------------------
# Mostly a modified copy from timm (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)
# ------------------------------------------------------------------------
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum
from torch import nn as nn
from einops import rearrange
#from .orpe import Orpe


class AfterReconstruction(nn.Identity):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        # x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class SimpleRMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(SimpleRMSNorm, self).__init__()
        self.eps = eps
        self.d = d

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        return x_normed


# attention based linear complexity
class LinearJointTSAttention(nn.Module):
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

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.layer_norm = nn.LayerNorm(dim)
        # self.layer_norm = nn.LayerNorm(head_dim)
        self.use_orpe = use_orpe
        if self.use_orpe:
            print("===================")
            print("use_orpe")
            self.orpe = Orpe(core_matrix, p_matrix, embedding_dim=head_dim,
                             theta_type=theta_type, theta_learned=theta_learned,
                             householder_learned=householder_learned, dim=orpe_dim)
            print("===================")

    def print_helper(self, x, i):
        print("=============================")
        print(f"location {i}")
        print(x.shape)
        print(torch.max(x))
        print(torch.min(x))
        print("=============================")

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
        e = q.shape[-1]
        # act
        eps = 1e-6
        q_ = F.relu(q) + self.scale
        k_ = F.relu(k) + self.scale
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
        d_min = 1e-4
        d_max = 1e4
        # b h n e, b h e n -> b h e e
        kv_ = torch.matmul(rearrange(k_, 'b n h e -> b h e n'),
                           rearrange(v_, 'b n h e -> b h n e'))
        # add
        # kv_ = torch.clamp(kv_, d_min, d_max)
        # b 1 h e
        k_sum = torch.sum(k_, axis=1, keepdim=True)  # no einsum
        z_ = 1 / (torch.sum(torch.mul(q_, k_sum), axis=-1) + eps)  # no einsum
        # add
        # z_ = torch.clamp(z_, d_min, d_max)

        # self.print_helper(kv_, 1)

        #### 计算qkv会产生较大的值
        # b h n e, b h e e -> b h n e -> b n h e
        attn_output = torch.matmul(q_.transpose(1, 2), kv_).transpose(1, 2)
        attn_output = torch.mul(attn_output, z_.unsqueeze(-1))
        # attn_output = torch.clamp(attn_output, d_min, d_max)
        # self.print_helper(attn_output, 2)

        # reshape

        # reshape
        attn_output = rearrange(attn_output, 'b (f n) h e -> b f n h e', f=num_frames)
        attn_output = rearrange(attn_output, 'b f n h e -> (b f) n (h e)')

        # self.print_helper(attn_output, 3)

        # outprojection
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        # self.print_helper(attn_output, 4)

        return attn_output

class AfterReconstruction(nn.Identity):
    def __init__(self, inplanes):
        super().__init__()
        self.inplanes = inplanes

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
            use_cgate=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.insert_control_point = insert_control_point
        self.scale = qk_scale or head_dim ** -0.5

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        ############gate reweighting########
        self.use_cgate = use_cgate
        if use_cgate:
            self.q_gate = nn.Linear(head_dim, head_dim)
            self.k_gate = nn.Linear(head_dim, head_dim)
        # self.layer_norm = nn.LayerNorm(dim)
        # self.layer_norm = nn.LayerNorm(head_dim)
        self.use_orpe = use_orpe
        if insert_control_point:
            self.control_point = AfterReconstruction(dim)
            self.control_point_query = AfterReconstruction(dim)
            self.control_point_value = AfterReconstruction(dim)
        if self.use_orpe:
            print("===================")
            print("use_orpe")
            self.orpe = Orpe(core_matrix, p_matrix, embedding_dim=head_dim,
                             theta_type=theta_type, theta_learned=theta_learned,
                             householder_learned=householder_learned, dim=orpe_dim)
            print("===================")

    def print_helper(self, x, i):
        print("=============================")
        print(f"location {i}")
        print(x.shape)
        print(torch.max(x))
        print(torch.min(x))
        print("=============================")

    def forward(self, x, execute=False):
        """
        intput shape: (b * f), (w * h), c
        """
        if self.insert_control_point and execute:
            x = self.control_point(x)
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
        if self.insert_control_point and execute:
            k = rearrange(k, 'b n h e -> b h n e')
            v = rearrange(v, 'b n h e -> b h n e')
            k = self.control_point_query(k)
            v = self.control_point_value(v)
            k = rearrange(k, 'b h n e -> b n h e')
            v = rearrange(v, 'b h n e -> b n h e')
        e = q.shape[-1]
        # act
        eps = 1e-6
        q_ = F.relu(q) + self.scale
        k_ = F.relu(k) + self.scale
        if self.use_cgate:
            q_ = F.sigmoid(self.q_gate(q_)) * q_
            k_ = F.sigmoid(self.k_gate(k_)) * k_
        # reshape
        # q_ = rearrange(q_, '(b f) n h e -> b f n h e', f=num_frames)
        # k_ = rearrange(k_, '(b f) n h e -> b f n h e', f=num_frames)
        # v_ = rearrange(v, '(b f) n h e -> b f n h e', f=num_frames)
        if self.use_orpe:
            q_ = self.orpe(q_)
            k_ = self.orpe(k_)
        # q_ = rearrange(q_, 'b f n h e -> b (f n) h e')
        # k_ = rearrange(k_, 'b f n h e -> b (f n) h e')
        # v_ = rearrange(v_, 'b f n h e -> b (f n) h e')

        ##### compute
        d_min = 1e-4
        d_max = 1e4
        # b h n e, b h e n -> b h e e
        kv_ = torch.matmul(rearrange(k_, 'b n h e -> b h e n'),
                           rearrange(v, 'b n h e -> b h n e'))
        # add
        # kv_ = torch.clamp(kv_, d_min, d_max)
        # b 1 h e
        k_sum = torch.sum(k_, axis=1, keepdim=True)  # no einsum
        z_ = 1 / (torch.sum(torch.mul(q_, k_sum), axis=-1) + eps)  # no einsum
        # add
        # z_ = torch.clamp(z_, d_min, d_max)

        # self.print_helper(kv_, 1)

        #### 计算qkv会产生较大的值
        # b h n e, b h e e -> b h n e -> b n h e
        attn_output = torch.matmul(q_.transpose(1, 2), kv_).transpose(1, 2)
        attn_output = torch.mul(attn_output, z_.unsqueeze(-1))
        # attn_output = torch.clamp(attn_output, d_min, d_max)
        # self.print_helper(attn_output, 2)

        # reshape

        # reshape
        # attn_output = rearrange(attn_output, 'b (f n) h e -> b f n h e', f=num_frames)
        # attn_output = rearrange(attn_output, 'b f n h e -> (b f) n (h e)')

        # self.print_helper(attn_output, 3)

        # outprojection
        attn_output = rearrange(attn_output, 'b n h e -> b n (h e)')
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        # self.print_helper(attn_output, 4)

        return attn_output

class LinearBlock(nn.Module):
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
            # ADD FOR ORPE
            use_orpe=False,
            core_matrix=1,
            p_matrix=3,
            theta_type="a",
            theta_learned=True,
            householder_learned=False,
            orpe_dim=1,
            has_kv=False,
            attention_type='full_time_space',
            insert_control_point=False,
            share_ts_params=False,
            use_cgate=False
    ):
        super().__init__()
        self.share_ts_params = share_ts_params
        self.norm1 = norm_layer(dim)
        self.attention_type = attention_type
        if self.attention_type == 'full_time_space':
            self.attn = LinearJointTSAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                # ADD FOR ORPE
                use_orpe=use_orpe,
                core_matrix=core_matrix,
                p_matrix=p_matrix,
                theta_type=theta_type,
                theta_learned=theta_learned,
                householder_learned=householder_learned,
                orpe_dim=orpe_dim,
            )
        elif self.attention_type == 'divided_time_space':
            if self.share_ts_params:
                self.attn = LinearAttention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    # ADD FOR ORPE
                    use_orpe=use_orpe,
                    core_matrix=core_matrix,
                    p_matrix=p_matrix,
                    theta_type=theta_type,
                    theta_learned=theta_learned,
                    householder_learned=householder_learned,
                    orpe_dim=orpe_dim,
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
                    # ADD FOR ORPE
                    use_orpe=use_orpe,
                    core_matrix=core_matrix,
                    p_matrix=p_matrix,
                    theta_type=theta_type,
                    theta_learned=theta_learned,
                    householder_learned=householder_learned,
                    orpe_dim=orpe_dim,
                    insert_control_point=insert_control_point,
                    use_cgate=use_cgate
                )
                self.temporal_norm1 = norm_layer(dim)
                self.temporal_attn = LinearAttention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    # ADD FOR ORPE
                    use_orpe=use_orpe,
                    core_matrix=core_matrix,
                    p_matrix=p_matrix,
                    theta_type=theta_type,
                    theta_learned=theta_learned,
                    householder_learned=householder_learned,
                    orpe_dim=orpe_dim,
                    insert_control_point=insert_control_point,
                    use_cgate=use_cgate
                )
                #self.temporal_fc = nn.Linear(dim, dim)
        elif self.attention_type == 'xvit':
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                # ADD FOR ORPE
                use_orpe=use_orpe,
                core_matrix=core_matrix,
                p_matrix=p_matrix,
                theta_type=theta_type,
                theta_learned=theta_learned,
                householder_learned=householder_learned,
                orpe_dim=orpe_dim,
                insert_control_point=insert_control_point,
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

    def forward(self, x, H, W, num_frames):
        T = num_frames
        if self.attention_type == 'full_time_space':
            x = x + self.drop_path(self.attn(self.norm1(x), num_frames))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            return x
        elif self.attention_type == 'xvit':
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            return x
        elif self.attention_type == 'divided_time_space':
            ############### add cls_token     #####################
            # xt = x[:, 1:, :]
            # xt = rearrange(xt, 'b (t s) m -> (b t) s m', t=T)
            ############################################
            xt = x
            BT, S, E = xt.shape
            B = BT//T
            ## Temporal
            xt_ = rearrange(xt, '(b t) (h w) m -> b t (h w) m', b=B, h=H, w=W, t=T)
            xt_ = xt_.transpose(1, 2)
            xt_ = rearrange(xt_, 'b (h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
            xt = rearrange(xt_, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
            if self.share_ts_params:
                res_temporal = self.drop_path(self.attn(self.norm1(xt)))
            else:
                res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt), execute=True))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m', b=B, h=H, w=W, t=T)
            #res_temporal = self.temporal_fc(res_temporal)
            xt = xt_ + res_temporal

            ## Spatial
            ############### add cls_token     #####################
            # init_cls_token = x[:, 0, :].unsqueeze(1)
            # cls_token = init_cls_token.repeat(1, T, 1)
            # cls_token = rearrange(cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)

            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
            ############### add cls_token     #####################
            # xs = torch.cat((cls_token, xs), 1)

            res_spatial = self.drop_path(self.attn(self.norm1(xs), execute=True))

            ### Taking care of CLS token
            # cls_token = res_spatial[:, 0, :]
            # cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
            # cls_token = torch.mean(cls_token, 1, True)  ## averaging for every frame
            # res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res = res_spatial
            x = xt

            ## Mlp
            # x = rearrange(x, 'b (h w t) m -> b (h w) t m', b=B, h=H, w=W, t=T).transpose(1, 2)
            # x = rearrange(x, 'b t s m -> b (t s) m')
            # res = rearrange(res, 'b (h w t) m -> b (h w) t m', b=B, h=H, w=W, t=T).transpose(1, 2)
            # res = rearrange(res, 'b t s m -> b (t s) m')
            # x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x+res
            x = rearrange(x, 'b (h w t) m -> b (h w) t m', b=B, h=H, w=W, t=T).transpose(1, 2)
            x = rearrange(x, 'b t (h w) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            #x = rearrange(x, '(b t) (h w) m -> b (t h w) m', b=B, h=H, w=W, t=T)
            #x = torch.cat((cls_token, x), 1)
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
