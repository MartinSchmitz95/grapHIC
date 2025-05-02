import math
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
        if self.use_act:
            x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_weight, layer_[0])
            if self.use_bn:
                x_new = self.bns[i + 1](x_new)
            if self.use_act:
                x_new = self.activation(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x_new + x
            else:
                x = x_new
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


class SeedAttentionBlock(nn.Module):
    """
    Multi-layer seed attention block
    """
    def __init__(
        self,
        hidden_channels,
        num_layers=1,
        num_heads=1,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
        use_act=True,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(SeedAttentionLayer(hidden_channels, num_heads))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
            
    def forward(self, x, seed_node_id):
        """
        Args:
            x: Node features [num_nodes, hidden_channels]
            seed_node_id: Index of the seed node
        """
        attentions = []
        
        for i, layer in enumerate(self.layers):
            x_new = layer(x, seed_node_id)
            
            if self.use_bn:
                x_new = self.bns[i](x_new)
            if self.use_act:
                x_new = self.activation(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            if self.use_residual:
                x = x_new + x
            else:
                x = x_new
                
            # Store attention weights for visualization if needed
            with torch.no_grad():
                _, attn = self.layers[i](x, seed_node_id, output_attn=True)
                attentions.append(attn)
                
        return x, attentions


class SeedGNNT(nn.Module):
    """
    Model that first applies multi-head seed attention to the input features
    and then processes the graph with GNN layers. The final predictor takes
    a concatenation of the seed node and actual node embeddings.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        seed_attn_num_layers=2,
        seed_attn_num_heads=8,
        seed_attn_dropout=0.1,
        seed_attn_use_bn=True,
        seed_attn_use_residual=True,
        seed_attn_use_act=True,
        gnn_num_layers_0=2,  # Number of layers for edge type 0
        gnn_num_layers_1=2,  # Number of layers for edge type 1
        gnn_dropout=0.1,
        gnn_use_weight=True,
        gnn_use_init=False,
        gnn_use_bn=True,
        gnn_use_residual=True,
        gnn_use_act=True,
    ):
        super(SeedGNNT, self).__init__()
        
        # Seed attention block (now comes first)
        self.seed_attention = SeedAttentionBlock(
            in_channels,  # Now takes raw input features
            seed_attn_num_layers,
            seed_attn_num_heads,
            seed_attn_dropout,
            seed_attn_use_bn,
            seed_attn_use_residual,
            seed_attn_use_act,
        )
        
        # GNN layers for different edge types (now come after seed attention)
        self.graph_conv_0 = GraphConv(
            in_channels,  # Still takes original input features
            hidden_channels,
            gnn_num_layers_0,
            gnn_dropout,
            gnn_use_bn,
            gnn_use_residual,
            gnn_use_weight,
            gnn_use_init,
            gnn_use_act,
        )
        
        self.graph_conv_1 = GraphConv(
            in_channels,  # Still takes original input features
            hidden_channels,
            gnn_num_layers_1,
            gnn_dropout,
            gnn_use_bn,
            gnn_use_residual,
            gnn_use_weight,
            gnn_use_init,
            gnn_use_act,
        )
        
        # Final prediction layer takes concatenation of seed node and node embeddings
        # hidden_channels * 2 (from both GNNs) + hidden_channels (seed node)
        self.output = nn.Linear(hidden_channels * 2 + hidden_channels, out_channels)
        
    def reset_parameters(self):
        self.seed_attention.reset_parameters()
        self.graph_conv_0.reset_parameters()
        self.graph_conv_1.reset_parameters()
        self.output.reset_parameters()
        
    def forward(self, data):
        x, edge_index, edge_type, edge_weight, seed_node_id = data.x, data.edge_index, data.edge_type, data.edge_weight, data.seed_node_id
        
        # First apply seed attention to the input features
        x_attn, _ = self.seed_attention(x, seed_node_id)
        
        # Process with GNN for edge type 0 (using original features)
        mask_0 = edge_type == 0
        edge_index_0 = edge_index[:, mask_0]
        edge_weight_0 = edge_weight[mask_0]
        x_graph_0 = self.graph_conv_0(x, edge_index_0, edge_weight_0)
        
        # Process with GNN for edge type 1 (using original features)
        mask_1 = edge_type == 1
        edge_index_1 = edge_index[:, mask_1]
        edge_weight_1 = edge_weight[mask_1]
        x_graph_1 = self.graph_conv_1(x, edge_index_1, edge_weight_1)
        
        # Get the seed node embedding
        seed_node_embedding = x_attn[seed_node_id].expand(x.size(0), -1)
        
        # Concatenate GNN embeddings with seed node embedding
        x_combined = torch.cat([x_graph_0, x_graph_1, seed_node_embedding], dim=1)
        
        # Final prediction
        out = self.output(x_combined)
        
        # Ensure output is squeezed to match target dimensions
        return out.squeeze(-1)  # Change from [N, 1] to [N]
    
    def get_attentions(self, data):
        """
        Get attention weights for visualization
        """
        x, edge_index, edge_type, edge_weight, seed_node_id = data.x, data.edge_index, data.edge_type, data.edge_weight, data.seed_node_id
        
        # Apply seed attention to get attention weights
        _, attentions = self.seed_attention(x, seed_node_id)
        
        return attentions