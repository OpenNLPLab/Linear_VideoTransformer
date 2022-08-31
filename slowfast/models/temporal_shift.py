import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .transformers import VisionTransformer

###################


class SegmentConsensus(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SegmentConsensus, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == "avg":
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == "identity":
            output = input_tensor
        else:
            output = None

        return output


class ConsensusModule(torch.nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = (
            consensus_type if consensus_type != "rnn" else "identity"
        )
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim)(input)


##############


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        nt, num_heads, d, c = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, num_heads, d, c)
        fold = c * num_heads // self.fold_div

        x = (
            x.permute(0, 1, 2, 4, 3)
            .contiguous()
            .view(n_batch, self.n_segment, num_heads * c, d)
        )
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # shift right
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]  # not shift

        out = (
            out.view(n_batch, self.n_segment, num_heads, c, d)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )

        return out.view(nt, num_heads, d, c)


def make_temporal_shift(net, n_segment, n_div=8, locations_list=[]):
    n_segment_list = [n_segment] * 20
    assert n_segment_list[-1] > 0

    counter = 0
    for idx, block in enumerate(net.blocks):
        if idx in locations_list:
            net.blocks[idx].attn.control_point_query = TemporalShift(
                net.blocks[idx].attn.control_point_query,
                n_segment=n_segment_list[counter + 2],
                n_div=n_div,
            )
            net.blocks[idx].attn.control_point_value = TemporalShift(
                net.blocks[idx].attn.control_point_value,
                n_segment=n_segment_list[counter + 2],
                n_div=n_div,
            )
            counter += 1

############################################spatial shift#########################################################

class SpatialShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, shift_size=1):
        super(SpatialShift, self).__init__()
        self.net = net
        self.shift_size = shift_size
        self.n_segment = n_segment
        self.fold_div = n_div
        self.eightneiborshift = True

    def forward(self, x):
        ns, num_heads, t, c = x.size()
        n_batch = ns // self.n_segment
        x = x.view(n_batch, self.n_segment, num_heads, t, c)
        if self.eightneiborshift:
            neibornum = (self.shift_size*2+1)**2-1
            self.fold_div = neibornum*2
        fold = c * num_heads // self.fold_div

        x = (
            x.permute(0, 1, 2, 4, 3)
            .contiguous()
            .view(n_batch, self.n_segment, num_heads * c, t)
        )
        H = int(self.n_segment**0.5)
        x = rearrange(x, 'b (h w) m t -> b h w m t', b=n_batch, h=H, w=H)
        out = torch.zeros_like(x)
        ##################h shift#################
        out[:, :-1*self.shift_size, :, :fold] = x[:, self.shift_size:, :, :fold]  # shift left
        out[:, self.shift_size:, :, fold : 2 * fold] = x[:, :-1*self.shift_size, :, fold : 2 * fold]  # shift right
        ##################w shift#################
        out[:, :, :-1*self.shift_size, 2*fold : 3*fold] = x[:, :, self.shift_size:, 2*fold : 3*fold]  # shift left
        out[:, :, self.shift_size:, 3*fold : 4*fold] = x[:, :, :-1*self.shift_size, 3*fold : 4*fold]  # shift right
        out[:, :, :, 4 * fold:] = x[:, :, :, 4 * fold:]  # not shift
        
        if self.eightneiborshift:
            out[:, self.shift_size:, self.shift_size:, 4 * fold:5*fold] = x[:, :-1*self.shift_size, :-1*self.shift_size, 4 * fold:5*fold]
            out[:, :-1*self.shift_size, :-1*self.shift_size, 5 * fold:6*fold] = x[:, self.shift_size:,self.shift_size:, 5 * fold:6*fold]
            out[:, self.shift_size:, :-1 * self.shift_size, 6 * fold:7*fold] = x[:, :-1*self.shift_size,self.shift_size:, 6 * fold:7*fold]
            out[:, :-1*self.shift_size, self.shift_size:, 7 * fold:8*fold] = x[:, self.shift_size:,:-1*self.shift_size, 7 * fold:8*fold]
            if self.shift_size == 2:
                out[:, 2:, 1:, 8 * fold:9 * fold] = x[:, :-2, :-1, 8 * fold:9 * fold]
                out[:, 2:, :-1, 9 * fold:10 * fold] = x[:, :-2, 1:, 9 * fold:10 * fold]
                out[:, 1:, 2:, 10 * fold:11 * fold] = x[:, :-1, :-2, 10 * fold:11 * fold]
                out[:, :-1, 2:, 11 * fold:12 * fold] = x[:, 1:, :-2, 11 * fold:12 * fold]
                out[:, :-2, 1:, 12 * fold:13 * fold] = x[:, 2:, :-1, 12 * fold:13 * fold]
                out[:, :-2, :-1, 13 * fold:14 * fold] = x[:, 2:, 1:, 13 * fold:14 * fold]
                out[:, 1:, :-2, 14 * fold:15 * fold] = x[:, :-1, 2:, 14 * fold:15 * fold]
                out[:, :-1, :-2, 15 * fold:16 * fold] = x[:, 1:, 2:, 15 * fold:16 * fold]

                out[:, :-1, :, 16 * fold:17 * fold] = x[:, 1:, :, 16 * fold:17 * fold]
                out[:, 1:, :, 17 * fold:18 * fold] = x[:, :-1, :, 17 * fold:18 * fold]
                out[:, :, :-1, 18 * fold:19 * fold] = x[:, :, 1:, 18 * fold:19 * fold]
                out[:, :, 1:, 19 * fold:20 * fold] = x[:, :, :-1, 19 * fold:20 * fold]
                out[:, 1:, 1:, 20 * fold:21 * fold] = x[:, :-1, :-1, 20 * fold:21 * fold]
                out[:, :-1, :-1, 21 * fold:22 * fold] = x[:, 1:, 1:, 21 * fold:22 * fold]
                out[:, 1:, :-1, 22 * fold:23 * fold] = x[:, :-1, 1:, 22 * fold:23 * fold]
                out[:, :-1, 1:, 23 * fold:24 * fold] = x[:, 1:, :-1, 23 * fold:24 * fold]


        out = rearrange(out, 'b h w m t -> b (h w) m t', b=n_batch, h=H, w=H)
        out = (
            out.view(n_batch, self.n_segment, num_heads, c, t)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )

        return out.view(ns, num_heads, t, c)


def make_spatial_shift(net, n_segment, n_div=8, locations_list=[], shift_size=1):
    n_segment_list = [n_segment] * 20
    assert n_segment_list[-1] > 0

    counter = 0
    for idx, block in enumerate(net.blocks):
        if idx in locations_list:
            net.blocks[idx].temporal_attn.control_point_query = SpatialShift(
                net.blocks[idx].temporal_attn.control_point_query,
                n_segment=n_segment_list[counter + 2],
                n_div=n_div,
                shift_size=shift_size
            )
            net.blocks[idx].temporal_attn.control_point_value = SpatialShift(
                net.blocks[idx].temporal_attn.control_point_value,
                n_segment=n_segment_list[counter + 2],
                n_div=n_div,
                shift_size=shift_size
            )
            counter += 1
