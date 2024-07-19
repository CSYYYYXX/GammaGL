import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../layer')))
from typing import Optional

import tensorlayerx as tlx
import tensorlayerx.nn as nn

from gammagl.utils import degree


def full_attention_conv(qs, ks, vs, output_attn=False):
    qs = qs / tlx.ops.l2_normalize(qs, axis=-1)
    ks = ks / tlx.ops.l2_normalize(ks, axis=-1)
    N = qs.shape[0]

    kvs = tlx.ops.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = tlx.ops.einsum("nhm,hmd->nhd", qs, kvs)
    attention_num += N * vs

    all_ones = tlx.ones([ks.shape[0]])
    ks_sum = tlx.ops.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = tlx.ops.einsum("nhm,hm->nh", qs, ks_sum)
    attention_normalizer = tlx.expand_dims(attention_normalizer, axis=-1)
    attention_normalizer += tlx.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer

    if output_attn:
        attention = tlx.ops.einsum("nhm,lhm->nlh", qs, ks).mean(axis=-1)
        normalizer = attention_normalizer.squeeze(axis=-1).mean(axis=-1, keepdims=True)
        attention = attention / normalizer
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads) if use_weight else None

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels) if self.use_weight else source_input.reshape(-1, 1, self.out_channels)

        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, output_attn)
        else:
            attention_output = full_attention_conv(query, key, value)

        final_output = attention_output.mean(axis=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1, alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = nn.ReLU
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, data):
        x = data['node_feat']
        edge_index = data['edge_index']
        edge_weight = data['edge_weight'] if 'edge_weight' in data else None
        layer_ = []

        x = self.fcs(x)
        if self.use_bn:
            x = self.bns(x)
        x = self.activation(x)
        x = nn.Dropout(self.dropout)(x)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation()(x)
            x = nn.Dropout(self.dropout)(x)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return tlx.stack(attentions, axis=0)


