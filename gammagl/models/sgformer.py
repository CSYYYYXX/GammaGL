import sys
import os

# Add the model and datasets directories to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../layer')))
import tensorlayerx as tlx
import tensorlayerx.nn as nn
from sgformer_layer import TransConv
from gammagl.layers.conv import gcn_conv
from gammagl.utils import degree


class SGFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1, alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_graph=True, use_act=False, graph_weight=0.8, gnn=None, aggregate='add'):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, num_layers, num_heads, alpha, dropout, use_bn, use_residual, use_weight)
        self.gnn = gnn
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.use_act = use_act
        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type: {aggregate}')

        self.params1 = list(self.trans_conv.trainable_weights)
        self.params2 = list(self.gnn.trainable_weights) if self.gnn is not None else []
        self.params2.extend(self.fc.trainable_weights)

    def forward(self, data):
        x1 = self.trans_conv(data)
        if self.use_graph:
            x2 = self.gnn(data)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = tlx.concat([x1, x2], axis=1)
        else:
            x = x1
        x = self.fc(x)
        return x

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)
        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.gnn.reset_parameters()
