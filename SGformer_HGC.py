import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.data import HeteroData

def full_attention_conv(qs, ks, vs, output_attn=False):
    """
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    return output [N, H, D]
    """
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape)
    )  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(
            dim=-1, keepdims=True
        )  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        # Adjust input channels to handle concatenated features from previous layer
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, edge_weight, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1.0 / d[col]).sqrt()
        d_norm_out = (1.0 / d[row]).sqrt()
        # Multiply the normalization with edge weights
        value = edge_weight * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
        if self.use_weight:
            x = self.W(x)

        return x

class GraphConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_init=False,
        use_act=True,
    ):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # All layers process hidden_channels like in original SGformer.py
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init)
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
            layer_.append(x)
        return x


class TransConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn
            )  # [N, H, D]
        else:
            attention_output = full_attention_conv(query, key, value)  # [N, H, D]

        final_output = attention_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        num_heads=1,
        alpha=0.5,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_weight=True,
        use_act=True,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(
                    hidden_channels,
                    hidden_channels,
                    num_heads=num_heads,
                    use_weight=use_weight,
                )
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        # self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index=None):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x, edge_index)
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
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
            if self.use_residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGFormer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        trans_num_layers=1,
        trans_num_heads=1,
        trans_dropout=0.5,
        gnn_num_layers=1,  # Single parameter for GNN layers
        gnn_dropout=0.5,
        gnn_use_weight=True,
        gnn_use_init=False,
        gnn_use_bn=True,
        gnn_use_residual=True,
        gnn_use_act=True,
        alpha=0.5,
        trans_use_bn=True,
        trans_use_residual=True,
        trans_use_weight=True,
        trans_use_act=True,
        projection_dim=128,  # Dimension for contrastive learning projection
        layer_norm=False,  # New parameter to toggle between BatchNorm and LayerNorm
    ):
        super().__init__()
        
        self.trans_conv = TransConv(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            alpha,
            trans_dropout,
            trans_use_bn,
            trans_use_residual,
            trans_use_weight,
            trans_use_act,
        )
        
        # Replace separate GNNs with HeteroGCN from PyG
            
            
        self.convs = nn.ModuleList()
        for _ in range(gnn_num_layers):
            conv = HeteroConv({
                ('node', '0', 'node'): GCNConv(hidden_channels, hidden_channels),
                ('node', '1', 'node'): GCNConv(hidden_channels, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
            
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(gnn_num_layers):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            
        self.bns = nn.ModuleList()
        for _ in range(gnn_num_layers + 1):
            if layer_norm:
                self.bns.append(nn.LayerNorm(hidden_channels))
            else:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.gnn_dropout = gnn_dropout
        self.gnn_use_bn = gnn_use_bn
        self.gnn_use_residual = gnn_use_residual
        self.gnn_use_act = gnn_use_act
        self.layer_norm = layer_norm

        # Final layer for classification/regression tasks
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_channels, out_channels),
            nn.Tanh()
        )
        
        # Projection head for contrastive learning
        middle_proj_dim = (out_channels+projection_dim)//2
        self.projection = nn.Sequential(
            nn.Linear(out_channels, middle_proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(middle_proj_dim, projection_dim)
        )

    def forward(self, data, return_projection=True):
        x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        
        # Get transformer embeddings
        x_trans = self.trans_conv(x, edge_type)
        
        # Create a HeteroData object
        hetero_data = HeteroData()
        hetero_data['node'].x = self.lins[0](x)
        if self.gnn_use_bn:
            hetero_data['node'].x = self.bns[0](hetero_data['node'].x)
        hetero_data['node'].x = F.relu(hetero_data['node'].x)
        hetero_data['node'].x = F.dropout(hetero_data['node'].x, p=self.gnn_dropout, training=self.training)
        
        # Add edges by type
        for edge_type_val in [0, 1]:
            mask = edge_type == edge_type_val
            hetero_data[('node', str(edge_type_val), 'node')].edge_index = edge_index[:, mask]
            hetero_data[('node', str(edge_type_val), 'node')].edge_attr = edge_weight[mask].unsqueeze(1)
        
        # Apply heterogeneous GNN layers
        x_dict = {'node': hetero_data['node'].x}
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, hetero_data.edge_index_dict, hetero_data.edge_attr_dict)
            if self.gnn_use_residual:
                x_dict_new['node'] = x_dict_new['node'] + x_dict['node']
            
            x_dict_new['node'] = self.lins[i+1](x_dict_new['node'])
            if self.gnn_use_bn:
                x_dict_new['node'] = self.bns[i+1](x_dict_new['node'])
            if self.gnn_use_act:
                x_dict_new['node'] = F.relu(x_dict_new['node'])
            x_dict_new['node'] = F.dropout(x_dict_new['node'], p=self.gnn_dropout, training=self.training)
            
            x_dict = x_dict_new
        
        # Extract node features - single GNN embedding
        x_graph = x_dict['node']
        
        # Concatenate transformer and GNN embeddings
        combined_embedding = torch.cat([x_trans, x_graph], dim=1)
        
        # For standard tasks (classification/regression)
        output = self.fc(combined_embedding)
        output = F.normalize(output, p=2, dim=1)

        # For contrastive learning
        if return_projection:
            projection = self.projection(output)
            # Normalize the projection to unit hypersphere
            projection = F.normalize(projection, p=2, dim=1)
            return projection
        
        return output

    
    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]
        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        for conv in self.convs:
            for conv_layer in conv.convs.values():
                conv_layer.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
        # Reset projection head
        for layer in self.projection:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class debug_model(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        projection_dim=128,
    ):
        super().__init__()
        
        # Input layer (simplified from SGFormer)
        self.input_layer = nn.Linear(in_channels, hidden_channels)
        
        # Output layer (same as in SGFormer)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_channels, out_channels),
            nn.Tanh()
        )
        
        # Projection head for contrastive learning (same as in SGFormer)
        middle_proj_dim = (out_channels+projection_dim)//2
        self.projection = nn.Sequential(
            nn.Linear(out_channels, middle_proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(middle_proj_dim, projection_dim)
        )

    def forward(self, data, return_projection=False):
        x = data.x
        
        # Process through input layer
        x_processed = self.input_layer(x)
        
        # Create a dummy second embedding to match SGFormer's concatenation
        # This duplicates the processed features to maintain the expected dimensions
        combined_embedding = torch.cat([x_processed, x_processed], dim=1)
        
        # Final output
        output = self.fc(combined_embedding)
        
        # For contrastive learning
        if return_projection:
            projection = self.projection(output)
            # Normalize the projection to unit hypersphere
            projection = F.normalize(projection, p=2, dim=1)
            return projection
        
        return output
        
    def reset_parameters(self):
        self.input_layer.reset_parameters()
        for layer in self.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.projection:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()