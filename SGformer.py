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

def full_attention_conv(qs, ks, vs, output_attn=False, use_standard_attention=False):
    """
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    return output [N, H, D]
    """
    if use_standard_attention:
        # Standard full attention mechanism
        # [N, H, M] x [L, H, M].T -> [N, H, L]
        attn_scores = torch.einsum("nhm,lhm->nhl", qs, ks)
        
        # Apply softmax for normalization
        attn_weights = F.softmax(attn_scores / math.sqrt(ks.shape[-1]), dim=-1)
        
        # [N, H, L] x [L, H, D] -> [N, H, D]
        attn_output = torch.einsum("nhl,lhd->nhd", attn_weights, vs)
        
        if output_attn:
            # Average attention weights across heads for visualization
            attention = attn_weights.mean(dim=1)  # [N, L]
            return attn_output, attention
        else:
            return attn_output
    else:
        # Original SGFormer linear attention
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
    def __init__(self, in_channels, out_channels, num_heads, use_weight=True, use_standard_attention=False):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.use_standard_attention = use_standard_attention

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
                query, key, value, output_attn, self.use_standard_attention
            )  # [N, H, D]
        else:
            attention_output = full_attention_conv(
                query, key, value, output_attn=False, use_standard_attention=self.use_standard_attention
            )  # [N, H, D]

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
        use_standard_attention=False,
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
                    use_standard_attention=use_standard_attention,
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
        use_standard_attention=False,  # New parameter for attention type
        layer_norm=False,  # New flag to use LayerNorm instead of BatchNorm
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
            use_standard_attention,  # Pass the attention type parameter
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

    def forward(self, data):
        x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        
        # Remove edges with type==0
        """mask = edge_type != 0
        edge_index = edge_index[:, mask]
        edge_type = edge_type[mask]
        if edge_weight is not None:
            edge_weight = edge_weight[mask]"""
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # Get transformer embeddings
        x_trans = self.trans_conv(x, edge_type)
        #edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        # Create a HeteroData object
        hetero_data = HeteroData()
        hetero_data['node'].x = self.lins[0](x)
        if self.gnn_use_bn:
            hetero_data['node'].x = self.bns[0](hetero_data['node'].x)
        hetero_data['node'].x = F.relu(hetero_data['node'].x)
        hetero_data['node'].x = F.dropout(hetero_data['node'].x, p=self.gnn_dropout, training=self.training)
        
        # Add edges by type
        for edge_type_val in [0,1]:
            mask = edge_type == edge_type_val
            hetero_data[('node', str(edge_type_val), 'node')].edge_index = edge_index[:, mask]
            # Make sure edge_weight is properly handled for GCNConv
            if edge_weight is not None:
                hetero_data[('node', str(edge_type_val), 'node')].edge_attr = edge_weight[mask].unsqueeze(1)
        
        # Apply heterogeneous GNN layers
        x_dict = {'node': hetero_data['node'].x}
        for i, conv in enumerate(self.convs):
            # Store previous features for residual connection
            prev_features = x_dict['node']
            
            # Apply heterogeneous convolution
            x_dict_new = conv(x_dict, hetero_data.edge_index_dict, hetero_data.edge_attr_dict)
            
            # Apply linear transformation
            x_dict_new['node'] = self.lins[i+1](x_dict_new['node'])
            
            # Apply residual connection before normalization
            if self.gnn_use_residual:
                x_dict_new['node'] = x_dict_new['node'] + prev_features
            
            # Apply batch normalization or layer normalization
            if self.gnn_use_bn:
                x_dict_new['node'] = self.bns[i+1](x_dict_new['node'])
            
            # Apply activation and dropout
            if self.gnn_use_act:
                x_dict_new['node'] = F.relu(x_dict_new['node'])
            x_dict_new['node'] = F.dropout(x_dict_new['node'], p=self.gnn_dropout, training=self.training)
            
            x_dict = x_dict_new
        
        # Extract node features - single GNN embedding
        x_graph = x_dict['node']
        
        # Concatenate transformer and GNN embeddings
        combined_embedding = torch.cat([x_trans, x_graph], dim=1)
        
        x = self.fc(combined_embedding)
        
        return x

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        for conv in self.convs:
            for conv_layer in conv.convs.values():
                conv_layer.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        
        # Reset final layer
        for layer in self.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class SGFormerMulti(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_blocks=2,  # Number of SGFormer blocks to stack
        trans_num_layers=1,
        trans_num_heads=1,
        trans_dropout=0.5,
        gnn_num_layers=1,
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
        use_standard_attention=False,
        layer_norm=False,
        residual_between_blocks=True,  # Whether to use residual connections between blocks
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.residual_between_blocks = residual_between_blocks
        
        # Create multiple SGFormer blocks
        self.blocks = nn.ModuleList()
        
        # First block takes original input channels
        self.blocks.append(
            SGFormer(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,  # Intermediate blocks output hidden_channels
                trans_num_layers=trans_num_layers,
                trans_num_heads=trans_num_heads,
                trans_dropout=trans_dropout,
                gnn_num_layers=gnn_num_layers,
                gnn_dropout=gnn_dropout,
                gnn_use_weight=gnn_use_weight,
                gnn_use_init=gnn_use_init,
                gnn_use_bn=gnn_use_bn,
                gnn_use_residual=gnn_use_residual,
                gnn_use_act=gnn_use_act,
                alpha=alpha,
                trans_use_bn=trans_use_bn,
                trans_use_residual=trans_use_residual,
                trans_use_weight=trans_use_weight,
                trans_use_act=trans_use_act,
                use_standard_attention=use_standard_attention,
                layer_norm=layer_norm,
            )
        )
        
        # Subsequent blocks take hidden_channels as input
        for _ in range(num_blocks - 1):
            self.blocks.append(
                SGFormer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,  # Intermediate blocks output hidden_channels
                    trans_num_layers=trans_num_layers,
                    trans_num_heads=trans_num_heads,
                    trans_dropout=trans_dropout,
                    gnn_num_layers=gnn_num_layers,
                    gnn_dropout=gnn_dropout,
                    gnn_use_weight=gnn_use_weight,
                    gnn_use_init=gnn_use_init,
                    gnn_use_bn=gnn_use_bn,
                    gnn_use_residual=gnn_use_residual,
                    gnn_use_act=gnn_use_act,
                    alpha=alpha,
                    trans_use_bn=trans_use_bn,
                    trans_use_residual=trans_use_residual,
                    trans_use_weight=trans_use_weight,
                    trans_use_act=trans_use_act,
                    use_standard_attention=use_standard_attention,
                    layer_norm=layer_norm,
                )
            )
        
        # Final projection layer to output dimensions
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.Tanh()
        )
        
    def forward(self, data):
        x = data.x
        
        # Process the first block
        x_prev = self.blocks[0](data)
        
        # Create a new data object for subsequent blocks
        for i in range(1, self.num_blocks):
            # Update the data object with new node features
            new_data = data.clone()
            new_data.x = x_prev
            
            # Process through the current block
            x_curr = self.blocks[i](new_data)
            
            # Apply residual connection between blocks if specified
            if self.residual_between_blocks:
                x_prev = x_curr + x_prev
            else:
                x_prev = x_curr
        
        # Final projection
        output = self.final_proj(x_prev)
        
        return output
    
    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()
        
        for layer in self.final_proj:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

