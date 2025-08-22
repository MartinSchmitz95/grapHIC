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
import torch_scatter


class PairNorm(nn.Module):
    """
    PairNorm: Tackling Oversmoothing in GNNs
    Based on the paper: "PairNorm: Tackling Oversmoothing in GNNs"
    
    Args:
        scale (float): Scaling factor for the normalization
        norm_type (str): Type of normalization - 'pair', 'layer', or 'batch'
    """
    def __init__(self, scale=1.0, norm_type='pair'):
        super(PairNorm, self).__init__()
        self.scale = scale
        self.norm_type = norm_type
        
        if norm_type not in ['pair', 'layer', 'batch']:
            raise ValueError(f"norm_type must be one of ['pair', 'layer', 'batch'], got {norm_type}")
    
    def forward(self, x):
        """
        Apply PairNorm normalization
        
        Args:
            x (torch.Tensor): Input tensor of shape [N, D]
            
        Returns:
            torch.Tensor: Normalized tensor of same shape
        """
        if self.norm_type == 'pair':
            return self._pair_norm(x)
        elif self.norm_type == 'layer':
            return self._layer_norm(x)
        elif self.norm_type == 'batch':
            return self._batch_norm(x)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")
    
    def _pair_norm(self, x):
        """
        PairNorm: Normalize each node's features relative to the mean of all nodes
        """
        # Compute mean across all nodes
        mean = x.mean(dim=0, keepdim=True)  # [1, D]
        
        # Center the features
        x_centered = x - mean  # [N, D]
        
        # Compute the scale factor based on the variance
        var = torch.mean(x_centered ** 2, dim=1, keepdim=True)  # [N, 1]
        scale_factor = torch.sqrt(var + 1e-8)  # [N, 1]
        
        # Normalize and scale
        x_norm = x_centered / scale_factor  # [N, D]
        
        return self.scale * x_norm
    
    def _layer_norm(self, x):
        """
        Layer normalization: Normalize across feature dimensions for each node
        """
        # Compute mean and variance across feature dimensions
        mean = x.mean(dim=1, keepdim=True)  # [N, 1]
        var = x.var(dim=1, keepdim=True, unbiased=False)  # [N, 1]
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + 1e-8)  # [N, D]
        
        return self.scale * x_norm
    
    def _batch_norm(self, x):
        """
        Batch normalization: Normalize across nodes for each feature dimension
        """
        # Compute mean and variance across nodes
        mean = x.mean(dim=0, keepdim=True)  # [1, D]
        var = x.var(dim=0, keepdim=True, unbiased=False)  # [1, D]
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + 1e-8)  # [N, D]
        
        return self.scale * x_norm


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
        norm='batch',  # Normalization type: 'pair', 'layer', or 'batch'
        direct_ftrs=False,  # Whether to concatenate original features to final MLP
        pairnorm_scale=1.0,  # Scaling factor for PairNorm
    ):
        super().__init__()

        self.direct_ftrs = direct_ftrs
        self.norm = norm
        self.pairnorm_scale = pairnorm_scale

        # Validate norm parameter
        if norm not in ['pair', 'layer', 'batch']:
            raise ValueError(f"norm must be one of ['pair', 'layer', 'batch'], got {norm}")

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
            if norm == 'pair':
                self.bns.append(PairNorm(scale=pairnorm_scale, norm_type='pair'))
            elif norm == 'layer':
                self.bns.append(nn.LayerNorm(hidden_channels))
            else:  # norm == 'batch'
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.gnn_dropout = gnn_dropout
        self.gnn_use_bn = gnn_use_bn
        self.gnn_use_residual = gnn_use_residual
        self.gnn_use_act = gnn_use_act

        # Final layer for classification/regression tasks - adjust input size if using direct features
        final_input_dim = 2 * hidden_channels + (in_channels if direct_ftrs else 0)
        self.fc = nn.Sequential(
            nn.Linear(final_input_dim, out_channels),
            nn.Tanh()  # Readded for bounded outputs
        )

        # Initialize the final layer to center outputs around 0
        self._init_final_layer()

    def _init_final_layer(self):
        """Initialize final layer to center outputs around 0"""
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization for weights (centered around 0)
                nn.init.xavier_uniform_(layer.weight)
                # Explicitly set bias to zero
                nn.init.zeros_(layer.bias)

    def forward(self, data):
        x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        
        # Store original features if needed for direct concatenation
        original_x = x if self.direct_ftrs else None
        
        # Remove edges with type==0
        """mask = edge_type != 0
        edge_index = edge_index[:, mask]
        edge_type = edge_type[mask]
        if edge_weight is not None:
            edge_weight = edge_weight[mask]"""
        #edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

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
        
        # Concatenate original features if direct_ftrs is enabled
        if self.direct_ftrs:
            combined_embedding = torch.cat([combined_embedding, original_x], dim=1)
        
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
        norm='batch',  # Normalization type: 'pair', 'layer', or 'batch'
        residual_between_blocks=True,  # Whether to use residual connections between blocks
        direct_ftrs=False,  # Whether to concatenate original features to final output
        pairnorm_scale=1.0,  # Scaling factor for PairNorm
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.residual_between_blocks = residual_between_blocks
        self.direct_ftrs = direct_ftrs
        
        # Validate norm parameter
        if norm not in ['pair', 'layer', 'batch']:
            raise ValueError(f"norm must be one of ['pair', 'layer', 'batch'], got {norm}")
        
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
                norm=norm,
                direct_ftrs=False,  # Don't use direct features in intermediate blocks
                pairnorm_scale=pairnorm_scale,
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
                    norm=norm,
                    direct_ftrs=False,  # Don't use direct features in intermediate blocks
                    pairnorm_scale=pairnorm_scale,
                )
            )
        
        # Final projection layer to output dimensions - adjust input size if using direct features
        final_input_dim = hidden_channels + (in_channels if direct_ftrs else 0)
        self.final_proj = nn.Sequential(
            nn.Linear(final_input_dim, out_channels),
            nn.Tanh()  # Readded for bounded outputs
        )
        
        # Initialize the final layer to center outputs around 0
        self._init_final_layer()
        
    def _init_final_layer(self):
        """Initialize final layer to center outputs around 0"""
        for layer in self.final_proj:
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization for weights (centered around 0)
                nn.init.xavier_uniform_(layer.weight)
                # Explicitly set bias to zero
                nn.init.zeros_(layer.bias)

    def forward(self, data):
        x = data.x
        
        # Store original features if needed for direct concatenation
        original_x = x if self.direct_ftrs else None
        
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
        
        # Concatenate original features if direct_ftrs is enabled
        if self.direct_ftrs:
            final_features = torch.cat([x_prev, original_x], dim=1)
        else:
            final_features = x_prev
        
        # Final projection
        output = self.final_proj(final_features)
        
        return output
    
    def reset_parameters(self):
        for block in self.blocks:
            block.reset_parameters()
        
        for layer in self.final_proj:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class SGFormer_NoGate(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
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
        norm='batch',  # Normalization type: 'pair', 'layer', or 'batch'
        edge_feature_dim=1,  # Dimension of edge features
        direct_ftrs=False,  # Whether to concatenate original features to final MLP
        pairnorm_scale=1.0,  # Scaling factor for PairNorm
    ):
        super().__init__()

        self.direct_ftrs = direct_ftrs
        self.norm = norm
        self.pairnorm_scale = pairnorm_scale

        # Validate norm parameter
        if norm not in ['pair', 'layer', 'batch']:
            raise ValueError(f"norm must be one of ['pair', 'layer', 'batch'], got {norm}")

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
            use_standard_attention,
        )
        
        # Use GCNConv layers and incorporate edge features into edge weights
        from torch_geometric.nn import GCNConv
        
        # Edge feature encoder to convert edge features to scalars
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output between 0 and 1 for edge weight scaling
        )
        
        self.convs = nn.ModuleList()
        for i in range(gnn_num_layers):
            conv = GCNConv(hidden_channels, hidden_channels)
            self.convs.append(conv)
            
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(gnn_num_layers):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            
        self.bns = nn.ModuleList()
        for _ in range(gnn_num_layers + 1):
            if norm == 'pair':
                self.bns.append(PairNorm(scale=pairnorm_scale, norm_type='pair'))
            elif norm == 'layer':
                self.bns.append(nn.LayerNorm(hidden_channels))
            else:  # norm == 'batch'
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.gnn_dropout = gnn_dropout
        self.gnn_use_bn = gnn_use_bn
        self.gnn_use_residual = gnn_use_residual
        self.gnn_use_act = gnn_use_act
        self.edge_feature_dim = edge_feature_dim

        # Final layer for classification/regression tasks - adjust input size if using direct features
        final_input_dim = 2 * hidden_channels + (in_channels if direct_ftrs else 0)
        self.fc = nn.Sequential(
            nn.Linear(final_input_dim, out_channels),
            nn.Tanh()  # Readded for bounded outputs
        )

        # Initialize the final layer to center outputs around 0
        self._init_final_layer()

    def _init_final_layer(self):
        """Initialize final layer to center outputs around 0"""
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization for weights (centered around 0)
                nn.init.xavier_uniform_(layer.weight)
                # Explicitly set bias to zero
                nn.init.zeros_(layer.bias)

    def forward(self, data):
        x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        
        # Store original features if needed for direct concatenation
        original_x = x if self.direct_ftrs else None
        
        # Use edge weights as edge features instead of gating weights
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Prepare edge features - combine edge_type and edge_weight
        edge_features = torch.cat([
            edge_type.float().unsqueeze(1),  # Convert edge type to float feature
            edge_weight.unsqueeze(1) if edge_weight.dim() == 1 else edge_weight
        ], dim=1)
        
        # Ensure edge features have the right dimension
        if edge_features.size(1) < self.edge_feature_dim:
            # Pad with zeros if needed
            padding = torch.zeros(edge_features.size(0), self.edge_feature_dim - edge_features.size(1), 
                                device=edge_features.device)
            edge_features = torch.cat([edge_features, padding], dim=1)
        elif edge_features.size(1) > self.edge_feature_dim:
            # Truncate if too large
            edge_features = edge_features[:, :self.edge_feature_dim]

        # Get transformer embeddings
        x_trans = self.trans_conv(x, edge_type)
        
        # Initialize GNN features
        x_graph = self.lins[0](x)
        if self.gnn_use_bn:
            x_graph = self.bns[0](x_graph)
        x_graph = F.relu(x_graph)
        x_graph = F.dropout(x_graph, p=self.gnn_dropout, training=self.training)
        
        # Apply GCNConv layers with edge features encoded as edge weights
        for i, conv in enumerate(self.convs):
            # Store previous features for residual connection
            prev_features = x_graph
            
            # Encode edge features into edge weights for GCNConv
            encoded_edge_weights = self.edge_encoder(edge_features).squeeze(-1)
            x_graph = conv(x_graph, edge_index, edge_weight=encoded_edge_weights)
            
            # Apply linear transformation
            x_graph = self.lins[i+1](x_graph)
            
            # Apply residual connection before normalization
            if self.gnn_use_residual:
                x_graph = x_graph + prev_features
            
            # Apply batch normalization or layer normalization
            if self.gnn_use_bn:
                x_graph = self.bns[i+1](x_graph)
            
            # Apply activation and dropout
            if self.gnn_use_act:
                x_graph = F.relu(x_graph)
            x_graph = F.dropout(x_graph, p=self.gnn_dropout, training=self.training)
        
        # Concatenate transformer and GNN embeddings
        combined_embedding = torch.cat([x_trans, x_graph], dim=1)
        
        # Concatenate original features if direct_ftrs is enabled
        if self.direct_ftrs:
            combined_embedding = torch.cat([combined_embedding, original_x], dim=1)
        
        x = self.fc(combined_embedding)
        
        return x

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        
        # Reset edge encoder
        for layer in self.edge_encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # Reset final layer
        for layer in self.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class SGFormerEdgeEmbs(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        edge_embedding_dim=16,  # Dimension of edge embeddings
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
        norm='batch',  # Normalization type: 'pair', 'layer', or 'batch'
        direct_ftrs=False,  # Whether to concatenate original features to final MLP
        pairnorm_scale=1.0,  # Scaling factor for PairNorm
    ):
        super().__init__()

        self.direct_ftrs = direct_ftrs
        self.norm = norm
        self.pairnorm_scale = pairnorm_scale
        self.edge_embedding_dim = edge_embedding_dim

        # Validate norm parameter
        if norm not in ['pair', 'layer', 'batch']:
            raise ValueError(f"norm must be one of ['pair', 'layer', 'batch'], got {norm}")

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
            use_standard_attention,
        )
        
        # Edge embedding initialization from edge weights
        self.edge_embedding_init = nn.Sequential(
            nn.Linear(1, edge_embedding_dim // 2),  # Start with edge weight
            nn.ReLU(),
            nn.Linear(edge_embedding_dim // 2, edge_embedding_dim),
            nn.Tanh()  # Bounded embeddings
        )
        
        # Node-to-edge message passing layers
        self.node_to_edge_layers = nn.ModuleList()
        for _ in range(gnn_num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),  # 2 nodes per edge
                nn.ReLU(),
                nn.Linear(hidden_channels, edge_embedding_dim),
                nn.Tanh()
            )
            self.node_to_edge_layers.append(layer)
        
        # Edge-to-node message passing layers
        self.edge_to_node_layers = nn.ModuleList()
        for _ in range(gnn_num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_channels + edge_embedding_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Tanh()
            )
            self.edge_to_node_layers.append(layer)
        
        # Node feature processing layers
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(gnn_num_layers):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            
        self.bns = nn.ModuleList()
        for _ in range(gnn_num_layers + 1):
            if norm == 'pair':
                self.bns.append(PairNorm(scale=pairnorm_scale, norm_type='pair'))
            elif norm == 'layer':
                self.bns.append(nn.LayerNorm(hidden_channels))
            else:  # norm == 'batch'
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.gnn_dropout = gnn_dropout
        self.gnn_use_bn = gnn_use_bn
        self.gnn_use_residual = gnn_use_residual
        self.gnn_use_act = gnn_use_act

        # Final layer for classification/regression tasks - adjust input size if using direct features
        final_input_dim = 2 * hidden_channels + (in_channels if direct_ftrs else 0)
        self.fc = nn.Sequential(
            nn.Linear(final_input_dim, out_channels),
            nn.Tanh()  # Readded for bounded outputs
        )

        # Initialize the final layer to center outputs around 0
        self._init_final_layer()

    def _init_final_layer(self):
        """Initialize final layer to center outputs around 0"""
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization for weights (centered around 0)
                nn.init.xavier_uniform_(layer.weight)
                # Explicitly set bias to zero
                nn.init.zeros_(layer.bias)

    def forward(self, data):
        x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        
        # Store original features if needed for direct concatenation
        original_x = x if self.direct_ftrs else None
        
        # Initialize edge weights if None
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Get transformer embeddings
        x_trans = self.trans_conv(x, edge_type)
        
        # Initialize node features
        x_graph = self.lins[0](x)
        if self.gnn_use_bn:
            x_graph = self.bns[0](x_graph)
        x_graph = F.relu(x_graph)
        x_graph = F.dropout(x_graph, p=self.gnn_dropout, training=self.training)
        
        # Initialize edge embeddings from edge weights
        edge_embeddings = self.edge_embedding_init(edge_weight.unsqueeze(1))
        
        # Apply message passing layers
        for i in range(len(self.node_to_edge_layers)):
            # Store previous features for residual connection
            prev_node_features = x_graph
            prev_edge_embeddings = edge_embeddings
            
            # Node-to-edge message passing
            row, col = edge_index
            node_pairs = torch.cat([x_graph[row], x_graph[col]], dim=1)  # [E, 2*H]
            edge_updates = self.node_to_edge_layers[i](node_pairs)  # [E, edge_embedding_dim]
            
            # Update edge embeddings
            edge_embeddings = edge_embeddings + edge_updates
            edge_embeddings = torch.tanh(edge_embeddings)  # Keep bounded
            
            # Edge-to-node message passing
            # First, aggregate edge embeddings for each node
            node_aggregated_edge_embeddings = torch_scatter.scatter_mean(
                edge_embeddings, row, dim=0, dim_size=x_graph.size(0)
            )  # [N, edge_embedding_dim]
            
            # Now concatenate node features with aggregated edge embeddings
            edge_to_node = torch.cat([x_graph, node_aggregated_edge_embeddings], dim=1)  # [N, H+edge_dim]
            node_updates = self.edge_to_node_layers[i](edge_to_node)  # [N, H]
            
            # Update node features
            x_graph = x_graph + node_updates
            x_graph = torch.tanh(x_graph)  # Keep bounded
            
            # Apply linear transformation
            x_graph = self.lins[i+1](x_graph)
            
            # Apply residual connection before normalization
            if self.gnn_use_residual:
                x_graph = x_graph + prev_node_features
                edge_embeddings = edge_embeddings + prev_edge_embeddings
            
            # Apply batch normalization or layer normalization
            if self.gnn_use_bn:
                x_graph = self.bns[i+1](x_graph)
            
            # Apply activation and dropout
            if self.gnn_use_act:
                x_graph = F.relu(x_graph)
            x_graph = F.dropout(x_graph, p=self.gnn_dropout, training=self.training)
        
        # Concatenate transformer and GNN embeddings
        combined_embedding = torch.cat([x_trans, x_graph], dim=1)
        
        # Concatenate original features if direct_ftrs is enabled
        if self.direct_ftrs:
            combined_embedding = torch.cat([combined_embedding, original_x], dim=1)
        
        x = self.fc(combined_embedding)
        
        return x

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        
        # Reset edge embedding initialization
        for layer in self.edge_embedding_init:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # Reset message passing layers
        for layer in self.node_to_edge_layers:
            for sublayer in layer:
                if hasattr(sublayer, 'reset_parameters'):
                    sublayer.reset_parameters()
        
        for layer in self.edge_to_node_layers:
            for sublayer in layer:
                if hasattr(sublayer, 'reset_parameters'):
                    sublayer.reset_parameters()
        
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        
        # Reset final layer
        for layer in self.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class SGFormerGINEdgeEmbs(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        edge_embedding_dim=16,  # Dimension of edge embeddings
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
        norm='batch',  # Normalization type: 'pair', 'layer', or 'batch'
        direct_ftrs=False,  # Whether to concatenate original features to final MLP
        pairnorm_scale=1.0,  # Scaling factor for PairNorm
        eps=0.0,  # Epsilon for GIN (0.0 for sum aggregation)
    ):
        super().__init__()

        self.direct_ftrs = direct_ftrs
        self.norm = norm
        self.pairnorm_scale = pairnorm_scale
        self.edge_embedding_dim = edge_embedding_dim
        self.eps = eps

        # Validate norm parameter
        if norm not in ['pair', 'layer', 'batch']:
            raise ValueError(f"norm must be one of ['pair', 'layer', 'batch'], got {norm}")

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
            use_standard_attention,
        )
        
        # Edge embedding initialization from edge weights
        self.edge_embedding_init = nn.Sequential(
            nn.Linear(1, edge_embedding_dim // 2),  # Start with edge weight
            nn.ReLU(),
            nn.Linear(edge_embedding_dim // 2, edge_embedding_dim),
            nn.Tanh()  # Bounded embeddings
        )
        
        # GIN-style node MLPs (similar to GINConv but with edge embeddings)
        self.node_mlps = nn.ModuleList()
        self.node_mlps.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(gnn_num_layers):
            self.node_mlps.append(nn.Linear(hidden_channels, hidden_channels))
        
        # Edge MLPs for processing edge embeddings
        self.edge_mlps = nn.ModuleList()
        for _ in range(gnn_num_layers):
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(edge_embedding_dim, edge_embedding_dim),
                nn.ReLU(),
                nn.Linear(edge_embedding_dim, edge_embedding_dim),
                nn.Tanh()
            ))
        
        # Node-to-edge message passing (GIN-style aggregation with edge embeddings)
        self.node_to_edge_layers = nn.ModuleList()
        for _ in range(gnn_num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_channels * 2 + edge_embedding_dim, hidden_channels),  # 2 nodes + edge embedding
                nn.ReLU(),
                nn.Linear(hidden_channels, edge_embedding_dim),
                nn.Tanh()
            )
            self.node_to_edge_layers.append(layer)
        
        # Edge-to-node message passing (GIN-style with edge embeddings)
        self.edge_to_node_layers = nn.ModuleList()
        for _ in range(gnn_num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_channels + edge_embedding_dim, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Tanh()
            )
            self.edge_to_node_layers.append(layer)
            
        self.bns = nn.ModuleList()
        for _ in range(gnn_num_layers + 1):
            if norm == 'pair':
                self.bns.append(PairNorm(scale=pairnorm_scale, norm_type='pair'))
            elif norm == 'layer':
                self.bns.append(nn.LayerNorm(hidden_channels))
            else:  # norm == 'batch'
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.gnn_dropout = gnn_dropout
        self.gnn_use_bn = gnn_use_bn
        self.gnn_use_residual = gnn_use_residual
        self.gnn_use_act = gnn_use_act

        # Final layer for classification/regression tasks - adjust input size if using direct features
        final_input_dim = 2 * hidden_channels + (in_channels if direct_ftrs else 0)
        self.fc = nn.Sequential(
            nn.Linear(final_input_dim, out_channels),
            nn.Tanh()  # Readded for bounded outputs
        )

        # Initialize the final layer to center outputs around 0
        self._init_final_layer()

    def _init_final_layer(self):
        """Initialize final layer to center outputs around 0"""
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization for weights (centered around 0)
                nn.init.xavier_uniform_(layer.weight)
                # Explicitly set bias to zero
                nn.init.zeros_(layer.bias)

    def forward(self, data):
        x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        
        # Store original features if needed for direct concatenation
        original_x = x if self.direct_ftrs else None
        
        # Initialize edge weights if None
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Get transformer embeddings
        x_trans = self.trans_conv(x, edge_type)
        
        # Initialize node features (GIN-style)
        x_graph = self.node_mlps[0](x)
        if self.gnn_use_bn:
            x_graph = self.bns[0](x_graph)
        x_graph = F.relu(x_graph)
        x_graph = F.dropout(x_graph, p=self.gnn_dropout, training=self.training)
        
        # Initialize edge embeddings from edge weights
        edge_embeddings = self.edge_embedding_init(edge_weight.unsqueeze(1))
        
        # Apply GIN-style layers with edge embeddings
        for i in range(len(self.node_mlps) - 1):
            # Store previous features for residual connection
            prev_node_features = x_graph
            prev_edge_embeddings = edge_embeddings
            
            row, col = edge_index
            
            # GIN-style node aggregation with edge embeddings
            # 1. Aggregate neighbor features (GIN-style)
            neighbor_features = x_graph[col]  # [E, H]
            
            # 2. Add edge embeddings to neighbor features
            neighbor_with_edge = torch.cat([neighbor_features, edge_embeddings], dim=1)  # [E, H+edge_dim]
            
            # 3. Process through edge-to-node layer
            neighbor_processed = self.edge_to_node_layers[i](neighbor_with_edge)  # [E, H]
            
            # 4. Aggregate neighbors (GIN-style sum aggregation)
            aggregated_neighbors = torch_scatter.scatter_sum(
                neighbor_processed, row, dim=0, dim_size=x_graph.size(0)
            )  # [N, H]
            
            # 5. GIN-style update: MLP((1 + eps) * x + aggregated_neighbors)
            x_graph = (1 + self.eps) * x_graph + aggregated_neighbors
            x_graph = self.node_mlps[i + 1](x_graph)
            
            # Node-to-edge message passing (update edge embeddings)
            # Concatenate source, target, and current edge embeddings
            node_edge_input = torch.cat([
                x_graph[row],  # source node
                x_graph[col],  # target node  
                edge_embeddings  # current edge embedding
            ], dim=1)  # [E, 2*H + edge_dim]
            
            edge_updates = self.node_to_edge_layers[i](node_edge_input)  # [E, edge_embedding_dim]
            
            # Update edge embeddings
            edge_embeddings = edge_embeddings + edge_updates
            edge_embeddings = self.edge_mlps[i](edge_embeddings)  # Process through edge MLP
            
            # Apply residual connection before normalization
            if self.gnn_use_residual:
                x_graph = x_graph + prev_node_features
                edge_embeddings = edge_embeddings + prev_edge_embeddings
            
            # Apply batch normalization or layer normalization
            if self.gnn_use_bn:
                x_graph = self.bns[i + 1](x_graph)
            
            # Apply activation and dropout
            if self.gnn_use_act:
                x_graph = F.relu(x_graph)
            x_graph = F.dropout(x_graph, p=self.gnn_dropout, training=self.training)
        
        # Concatenate transformer and GNN embeddings
        combined_embedding = torch.cat([x_trans, x_graph], dim=1)
        
        # Concatenate original features if direct_ftrs is enabled
        if self.direct_ftrs:
            combined_embedding = torch.cat([combined_embedding, original_x], dim=1)
        
        x = self.fc(combined_embedding)
        
        return x

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        
        # Reset edge embedding initialization
        for layer in self.edge_embedding_init:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        # Reset GIN-style layers
        for mlp in self.node_mlps:
            mlp.reset_parameters()
        
        for mlp in self.edge_mlps:
            for layer in mlp:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        # Reset message passing layers
        for layer in self.node_to_edge_layers:
            for sublayer in layer:
                if hasattr(sublayer, 'reset_parameters'):
                    sublayer.reset_parameters()
        
        for layer in self.edge_to_node_layers:
            for sublayer in layer:
                if hasattr(sublayer, 'reset_parameters'):
                    sublayer.reset_parameters()
        
        for bn in self.bns:
            bn.reset_parameters()
        
        # Reset final layer
        for layer in self.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class SGFormer_L_Pred(nn.Module):
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
        norm='batch',  # Normalization type: 'pair', 'layer', or 'batch'
        direct_ftrs=False,  # Whether to concatenate original features to final MLP
        pairnorm_scale=1.0,  # Scaling factor for PairNorm
        layer_aggregation='concat',  # How to aggregate layer embeddings: 'concat', 'mean', 'sum', 'attention'
    ):
        super().__init__()

        self.direct_ftrs = direct_ftrs
        self.norm = norm
        self.pairnorm_scale = pairnorm_scale
        self.layer_aggregation = layer_aggregation
        self.gnn_num_layers = gnn_num_layers

        # Validate norm parameter
        if norm not in ['pair', 'layer', 'batch']:
            raise ValueError(f"norm must be one of ['pair', 'layer', 'batch'], got {norm}")
        
        # Validate layer aggregation parameter
        if layer_aggregation not in ['concat', 'mean', 'sum', 'attention']:
            raise ValueError(f"layer_aggregation must be one of ['concat', 'mean', 'sum', 'attention'], got {layer_aggregation}")

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
            if norm == 'pair':
                self.bns.append(PairNorm(scale=pairnorm_scale, norm_type='pair'))
            elif norm == 'layer':
                self.bns.append(nn.LayerNorm(hidden_channels))
            else:  # norm == 'batch'
                self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.gnn_dropout = gnn_dropout
        self.gnn_use_bn = gnn_use_bn
        self.gnn_use_residual = gnn_use_residual
        self.gnn_use_act = gnn_use_act

        # Calculate input dimension for final layer based on layer aggregation method
        if layer_aggregation == 'concat':
            # Concatenate transformer embedding + all GNN layer embeddings
            final_input_dim = hidden_channels + (gnn_num_layers + 1) * hidden_channels + (in_channels if direct_ftrs else 0)
        elif layer_aggregation == 'mean' or layer_aggregation == 'sum':
            # Transformer embedding + aggregated GNN embeddings
            final_input_dim = 2 * hidden_channels + (in_channels if direct_ftrs else 0)
        elif layer_aggregation == 'attention':
            # Transformer embedding + attention-weighted GNN embeddings
            final_input_dim = 2 * hidden_channels + (in_channels if direct_ftrs else 0)
            # Add attention mechanism for layer aggregation
            self.layer_attention = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1),
                nn.Softmax(dim=0)  # Softmax across layers
            )

        self.fc = nn.Sequential(
            nn.Linear(final_input_dim, out_channels),
            nn.Tanh()  # Readded for bounded outputs
        )

        # Initialize the final layer to center outputs around 0
        self._init_final_layer()

    def _init_final_layer(self):
        """Initialize final layer to center outputs around 0"""
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot initialization for weights (centered around 0)
                nn.init.xavier_uniform_(layer.weight)
                # Explicitly set bias to zero
                nn.init.zeros_(layer.bias)

    def forward(self, data):
        x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        
        # Store original features if needed for direct concatenation
        original_x = x if self.direct_ftrs else None
        
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
        for edge_type_val in [0,1]:
            mask = edge_type == edge_type_val
            hetero_data[('node', str(edge_type_val), 'node')].edge_index = edge_index[:, mask]
            # Make sure edge_weight is properly handled for GCNConv
            if edge_weight is not None:
                hetero_data[('node', str(edge_type_val), 'node')].edge_attr = edge_weight[mask].unsqueeze(1)
        
        # Store all layer embeddings
        layer_embeddings = []
        layer_embeddings.append(hetero_data['node'].x)  # Initial embedding
        
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
            
            # Store layer embedding
            layer_embeddings.append(x_dict_new['node'])
            
            x_dict = x_dict_new
        
        # Aggregate layer embeddings based on the specified method
        if self.layer_aggregation == 'concat':
            # Concatenate all layer embeddings
            all_embeddings = torch.cat(layer_embeddings, dim=1)  # [N, (L+1)*H]
            combined_embedding = torch.cat([x_trans, all_embeddings], dim=1)
            
        elif self.layer_aggregation == 'mean':
            # Average all layer embeddings
            stacked_embeddings = torch.stack(layer_embeddings, dim=0)  # [L+1, N, H]
            mean_embeddings = torch.mean(stacked_embeddings, dim=0)  # [N, H]
            combined_embedding = torch.cat([x_trans, mean_embeddings], dim=1)
            
        elif self.layer_aggregation == 'sum':
            # Sum all layer embeddings
            stacked_embeddings = torch.stack(layer_embeddings, dim=0)  # [L+1, N, H]
            sum_embeddings = torch.sum(stacked_embeddings, dim=0)  # [N, H]
            combined_embedding = torch.cat([x_trans, sum_embeddings], dim=1)
            
        elif self.layer_aggregation == 'attention':
            # Attention-weighted aggregation of layer embeddings
            stacked_embeddings = torch.stack(layer_embeddings, dim=0)  # [L+1, N, H]
            
            # Compute attention weights for each layer
            # Reshape for attention: [L+1, N, H] -> [N, L+1, H]
            embeddings_for_attention = stacked_embeddings.transpose(0, 1)  # [N, L+1, H]
            
            # Compute attention weights
            attention_weights = self.layer_attention(embeddings_for_attention)  # [N, L+1, 1]
            attention_weights = attention_weights.transpose(0, 1)  # [L+1, N, 1]
            
            # Apply attention weights
            weighted_embeddings = stacked_embeddings * attention_weights  # [L+1, N, H]
            attended_embeddings = torch.sum(weighted_embeddings, dim=0)  # [N, H]
            
            combined_embedding = torch.cat([x_trans, attended_embeddings], dim=1)
        
        # Concatenate original features if direct_ftrs is enabled
        if self.direct_ftrs:
            combined_embedding = torch.cat([combined_embedding, original_x], dim=1)
        
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
        
        # Reset layer attention if using attention aggregation
        if self.layer_aggregation == 'attention':
            for layer in self.layer_attention:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        # Reset final layer
        for layer in self.fc:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

