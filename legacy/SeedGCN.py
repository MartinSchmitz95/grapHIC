import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops


class WeightedGCNConv(GCNConv):
    """
    Extension of GCNConv that explicitly supports edge weights
    """
    def forward(self, x, edge_index, edge_weight=None, size=None):
        # GCNConv already supports edge_weight in its forward method
        # We're just making it more explicit here
        return super().forward(x, edge_index, edge_weight)


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5, use_bn=True, use_residual=True):
        super(GCNLayer, self).__init__()
        
        self.gcn = WeightedGCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.use_residual = use_residual
        
        # Residual connection if dimensions match
        self.residual = nn.Identity() if in_channels == out_channels and use_residual else None
        
    def reset_parameters(self):
        self.gcn.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()
            
    def forward(self, x, edge_index, edge_weight=None):
        # Apply GCN layer
        out = self.gcn(x, edge_index, edge_weight)
        
        # Apply batch normalization if specified
        if self.bn is not None:
            out = self.bn(out)
        
        # Apply residual connection if dimensions match
        if self.residual is not None:
            out = out + self.residual(x)
            
        return out


class GCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        dropout=0.5,
        use_bn=True,
        use_residual=True,
    ):
        super(GCNBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            GCNLayer(
                in_channels, 
                hidden_channels, 
                dropout=dropout,
                use_bn=use_bn,
                use_residual=False  # First layer can't use residual as dimensions don't match
            )
        )
        
        # Remaining layers
        for _ in range(num_layers - 1):
            self.layers.append(
                GCNLayer(
                    hidden_channels, 
                    hidden_channels, 
                    dropout=dropout,
                    use_bn=use_bn,
                    use_residual=use_residual
                )
            )
            
        self.dropout = dropout
        
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
            
    def forward(self, x, edge_index, edge_weight=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            
            # Apply activation and dropout for all but the last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return x


class Predictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(Predictor, self).__init__()
        
        # Three feed-forward layers
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, out_channels)
        
        self.dropout = dropout
        
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        
    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return x


class SeedGCN(nn.Module):
    """
    Model that processes a graph with GCN layers for different edge types.
    The final predictor can either use cross-attention between target node embeddings
    and the seed node embeddings, or simple concatenation of all embeddings.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        gcn_num_layers_0=2,  # Number of layers for edge type 0
        gcn_num_layers_1=2,  # Number of layers for edge type 1
        gcn_dropout=0.5,
        gcn_use_bn=True,
        gcn_use_residual=True,
        predictor_hidden_channels=64,
        predictor_dropout=0.5,
        use_cross_attention=False,  # Whether to use cross-attention or concatenation
    ):
        super(SeedGCN, self).__init__()
        
        self.use_cross_attention = use_cross_attention
        
        # Initial feature transformation layer
        self.feature_transform = nn.Linear(in_channels, hidden_channels)
        
        # GCN blocks for different edge types
        self.gcn_0 = GCNBlock(
            hidden_channels,  # Now takes transformed features as input
            hidden_channels,
            gcn_num_layers_0,
            gcn_dropout,
            gcn_use_bn,
            gcn_use_residual,
        )
        
        self.gcn_1 = GCNBlock(
            hidden_channels,  # Now takes transformed features as input
            hidden_channels,
            gcn_num_layers_1,
            gcn_dropout,
            gcn_use_bn,
            gcn_use_residual,
        )
        
        # Predictor input dimension depends on whether we use cross-attention or concatenation
        predictor_in_channels = 2 * hidden_channels if use_cross_attention else 4 * hidden_channels
        
        # Predictor with three feed-forward layers
        self.predictor = Predictor(
            predictor_in_channels,
            predictor_hidden_channels,
            out_channels,
            predictor_dropout,
        )
        
    def reset_parameters(self):
        self.feature_transform.reset_parameters()
        self.gcn_0.reset_parameters()
        self.gcn_1.reset_parameters()
        self.predictor.reset_parameters()
        
    def forward(self, data):
        x, edge_index, edge_type, edge_weight, seed_node_id = data.x, data.edge_index, data.edge_type, data.edge_weight, data.seed_node_id
        
        # Apply initial feature transformation
        x = F.relu(self.feature_transform(x))
        
        # Process with GCN for edge type 0
        mask_0 = edge_type == 0
        edge_index_0 = edge_index[:, mask_0]
        edge_weight_0 = edge_weight[mask_0] if edge_weight is not None else None
        x_0 = self.gcn_0(x, edge_index_0, edge_weight_0)
        
        # Process with GCN for edge type 1
        mask_1 = edge_type == 1
        edge_index_1 = edge_index[:, mask_1]
        edge_weight_1 = edge_weight[mask_1] if edge_weight is not None else None
        x_1 = self.gcn_1(x, edge_index_1, edge_weight_1)
        
        # Get embeddings for the seed node
        seed_embedding_0 = x_0[seed_node_id]
        seed_embedding_1 = x_1[seed_node_id]
        
        # Expand seed embeddings to match batch size
        seed_embedding_0 = seed_embedding_0.expand(x.size(0), -1)
        seed_embedding_1 = seed_embedding_1.expand(x.size(0), -1)
        
        if self.use_cross_attention:
            # Cross-attention approach
            # Concatenate embeddings for target and seed nodes separately
            target_emb = torch.cat([x_0, x_1], dim=1)            # [N, 2*hidden]
            seed_emb = torch.cat([seed_embedding_0, seed_embedding_1], dim=1)  # [N, 2*hidden]
            
            # Cross-attention computation
            attn_scores = torch.einsum('nd,nd->n', target_emb, seed_emb)  # [N]
            attn_weights = F.softmax(attn_scores, dim=0)                  # [N]
            x_combined = attn_weights.unsqueeze(1) * target_emb           # [N, 2*hidden]
        else:
            # Concatenation approach
            # Concatenate target node embeddings with seed node embeddings
            x_combined = torch.cat([x_0, x_1, seed_embedding_0, seed_embedding_1], dim=1)  # [N, 4*hidden]
        
        # Final prediction
        out = self.predictor(x_combined)
        
        # Ensure output is squeezed to match target dimensions
        return out.squeeze(-1)  # Change from [N, 1] to [N]
        
    def get_embeddings(self, data):
        """
        Get node embeddings for visualization
        """
        x, edge_index, edge_type, edge_weight = data.x, data.edge_index, data.edge_type, data.edge_weight
        
        # Apply initial feature transformation
        x = F.relu(self.feature_transform(x))
        
        # Process with GCN for edge type 0
        mask_0 = edge_type == 0
        edge_index_0 = edge_index[:, mask_0]
        edge_weight_0 = edge_weight[mask_0] if edge_weight is not None else None
        x_0 = self.gcn_0(x, edge_index_0, edge_weight_0)
        
        # Process with GCN for edge type 1
        mask_1 = edge_type == 1
        edge_index_1 = edge_index[:, mask_1]
        edge_weight_1 = edge_weight[mask_1] if edge_weight is not None else None
        x_1 = self.gcn_1(x, edge_index_1, edge_weight_1)
        
        return x_0, x_1