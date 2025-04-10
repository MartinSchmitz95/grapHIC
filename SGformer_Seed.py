import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

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


class SeedAttentionLayer(nn.Module):
    """
    Attention layer that propagates information from a seed node to all other nodes
    """
    def __init__(self, hidden_channels, num_heads=1):
        super().__init__()
        self.query_proj = nn.Linear(hidden_channels, hidden_channels)
        self.key_proj = nn.Linear(hidden_channels, hidden_channels)
        self.value_proj = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
    def reset_parameters(self):
        self.query_proj.reset_parameters()
        self.key_proj.reset_parameters()
        self.value_proj.reset_parameters()
        self.out_proj.reset_parameters()
        
    def forward(self, x, seed_node_id, output_attn=False):
        """
        Args:
            x: Node features [num_nodes, hidden_channels]
            seed_node_id: Index of the seed node
            output_attn: Whether to output attention weights
        """
        batch_size, hidden_dim = x.shape
        
        # Get seed node feature
        seed_feature = x[seed_node_id].unsqueeze(0)  # [1, hidden_dim]
        
        # Project queries, keys, and values
        queries = self.query_proj(x)  # [N, hidden_dim]
        keys = self.key_proj(seed_feature)  # [1, hidden_dim]
        values = self.value_proj(seed_feature)  # [1, hidden_dim]
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, self.num_heads, self.head_dim)  # [N, H, D/H]
        keys = keys.view(1, self.num_heads, self.head_dim)  # [1, H, D/H]
        values = values.view(1, self.num_heads, self.head_dim)  # [1, H, D/H]
        
        # Compute attention scores for each head
        # For each head, compute dot product between all queries and the seed key
        attn_scores = torch.zeros(batch_size, self.num_heads, 1, device=x.device)
        for h in range(self.num_heads):
            # [N, D/H] @ [D/H, 1] -> [N, 1]
            head_score = torch.mm(queries[:, h], keys[0, h].unsqueeze(1)) * self.scale
            attn_scores[:, h] = head_score
        
        # Apply softmax over the single seed node dimension (dim=2 is always size 1)
        attn_weights = F.softmax(attn_scores, dim=2)  # [N, H, 1]
        
        # Apply attention weights to values for each head
        attn_output = torch.zeros(batch_size, self.num_heads, self.head_dim, device=x.device)
        for h in range(self.num_heads):
            # [N, 1] @ [1, D/H] -> [N, D/H]
            head_output = attn_weights[:, h] @ values[:, h].unsqueeze(0)
            attn_output[:, h] = head_output.squeeze(1)
        
        # Reshape back to original dimensions
        attn_output = attn_output.reshape(batch_size, hidden_dim)
        
        # Project output
        out = self.out_proj(attn_output)  # [N, hidden_dim]
        
        if output_attn:
            return out, attn_weights.squeeze(-1)  # [N, H]
        else:
            return out


class SeedTransConv(nn.Module):
    """
    Multi-layer transformer with seed-based attention
    """
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
        use_act=True,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        
        for i in range(num_layers):
            self.convs.append(
                SeedAttentionLayer(
                    hidden_channels,
                    num_heads=num_heads
                )
            )
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, seed_node_id):
        # Input projection
        x = self.input_proj(x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Store as residual link
        layer_outputs = [x]

        for i, conv in enumerate(self.convs):
            # Seed attention layer
            x_new = conv(x, seed_node_id)
            if self.use_residual:
                x = self.alpha * x_new + (1 - self.alpha) * x
            else:
                x = x_new
                
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        return x

    def get_attentions(self, x, seed_node_id):
        # Input projection
        x = self.input_proj(x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        
        attentions = []
        for i, conv in enumerate(self.convs):
            x_new, attn = conv(x, seed_node_id, output_attn=True)
            attentions.append(attn)
            
            if self.use_residual:
                x = self.alpha * x_new + (1 - self.alpha) * x
            else:
                x = x_new
                
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
        
        return torch.stack(attentions, dim=0)  # [layer num, N, H]


class SGFormerSeed(nn.Module):
    """
    SGFormer variant that uses seed-based attention
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        seed_trans_num_layers=1,
        seed_trans_num_heads=1,
        seed_trans_dropout=0.5,
        gnn_num_layers_0=1,  # Number of layers for edge type 0
        gnn_num_layers_1=1,  # Number of layers for edge type 1
        gnn_dropout=0.5,
        gnn_use_weight=True,
        gnn_use_init=False,
        gnn_use_bn=True,
        gnn_use_residual=True,
        gnn_use_act=True,
        alpha=0.5,
        seed_trans_use_bn=True,
        seed_trans_use_residual=True,
        seed_trans_use_act=True,
        use_graph=True,
    ):
        super().__init__()
        
        # Seed transformer for seed-based attention
        self.seed_trans_conv = SeedTransConv(
            in_channels,
            hidden_channels,
            seed_trans_num_layers,
            seed_trans_num_heads,
            alpha,
            seed_trans_dropout,
            seed_trans_use_bn,
            seed_trans_use_residual,
            seed_trans_use_act,
        )
        
        # Create two separate GNNs with different numbers of layers
        self.graph_conv_0 = GraphConv(
            in_channels,
            hidden_channels,
            gnn_num_layers_0,  # Different number of layers for type 0
            gnn_dropout,
            gnn_use_bn,
            gnn_use_residual,
            gnn_use_weight,
            gnn_use_init,
            gnn_use_act,
        )
        
        self.graph_conv_1 = GraphConv(
            in_channels,
            hidden_channels,
            gnn_num_layers_1,  # Different number of layers for type 1
            gnn_dropout,
            gnn_use_bn,
            gnn_use_residual,
            gnn_use_weight,
            gnn_use_init,
            gnn_use_act,
        )
        
        self.use_graph = use_graph

        # Final layer now expects 5 * hidden_channels (3 from before + 2 for seed and current node embeddings)
        self.fc = nn.Sequential(
            nn.Linear(5 * hidden_channels, out_channels),
            nn.Tanh()
        )

    def forward(self, data):
        x, edge_index, edge_type, edge_weight, seed_node_id = data.x, data.edge_index, data.edge_type, data.edge_weight, data.seed_node_id
        
        # Process with seed transformer
        x_trans = self.seed_trans_conv(x, seed_node_id)
        
        if self.use_graph:
            # Process edge type 0
            mask_0 = edge_type == 0
            edge_index_0 = edge_index[:, mask_0]
            edge_weight_0 = edge_weight[mask_0]
            x_graph_0 = self.graph_conv_0(x, edge_index_0, edge_weight_0)
            
            # Process edge type 1
            mask_1 = edge_type == 1
            edge_index_1 = edge_index[:, mask_1]
            edge_weight_1 = edge_weight[mask_1]
            x_graph_1 = self.graph_conv_1(x, edge_index_1, edge_weight_1)
            
            # Get seed node embeddings
            seed_embedding = x_graph_0[seed_node_id].unsqueeze(0)  # [1, hidden_dim]
            seed_embedding = seed_embedding.expand(x_graph_0.size(0), -1)  # [N, hidden_dim]
            
            # For each node, include its own embedding and the seed node embedding
            node_embeddings = x_graph_0  # Use embeddings from graph_0 as node features
            
            # Concatenate all embeddings [seed_transformer, graph_type0, graph_type1, seed_node, current_node]
            x = torch.cat([x_trans, x_graph_0, x_graph_1, seed_embedding, node_embeddings], dim=1)
        else:
            # If not using graph, pad with zeros to maintain dimension
            batch_size = x_trans.shape[0]
            hidden_dim = x_trans.shape[1]
            
            # Create seed node embedding (just use the raw features when graph is not used)
            seed_embedding = x[seed_node_id].unsqueeze(0)  # [1, in_channels]
            seed_embedding = self.seed_trans_conv.input_proj(seed_embedding)  # [1, hidden_dim]
            seed_embedding = seed_embedding.expand(batch_size, -1)  # [N, hidden_dim]
            
            # Use transformed input features as node embeddings
            node_embeddings = self.seed_trans_conv.input_proj(x)  # [N, hidden_dim]
            
            # Add padding for the graph embeddings that would normally be there
            padding = torch.zeros(batch_size, 2 * hidden_dim).to(x_trans.device)
            
            x = torch.cat([x_trans, padding, seed_embedding, node_embeddings], dim=1)
        
        x = self.fc(x)
        return x

    def get_attentions(self, data):
        x, seed_node_id = data.x, data.seed_node_id
        attns = self.seed_trans_conv.get_attentions(x, seed_node_id)
        return attns

    def reset_parameters(self):
        self.seed_trans_conv.reset_parameters()
            
        if self.use_graph:
            self.graph_conv_0.reset_parameters()
            self.graph_conv_1.reset_parameters()
            
        for module in self.fc.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()