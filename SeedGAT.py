import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops


class WeightedGATConv(GATConv):
    """
    Extension of GATConv that supports edge weights
    """
    def forward(self, x, edge_index, edge_weight=None, size=None, return_attention_weights=None):
        # Handle edge weights in message passing
        if edge_weight is not None:
            # First ensure edge_weight is the right shape
            if edge_weight.dim() == 1:
                edge_weight = edge_weight.view(-1, 1)
                
            # Call parent class with return_attention_weights=True to get attention coefficients
            out, (edge_index, attention_weights) = super().forward(x, edge_index, size=size, return_attention_weights=True)
            
            # Apply edge weights to attention weights - element-wise multiplication
            # Make sure dimensions match
            weighted_attention = attention_weights * edge_weight
            
            # If requested, return attention weights
            if return_attention_weights:
                return out, (edge_index, weighted_attention)
            else:
                return out
        else:
            return super().forward(x, edge_index, size=size, return_attention_weights=return_attention_weights)


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6, use_bn=True, use_residual=True):
        super(GATLayer, self).__init__()
        
        self.gat = WeightedGATConv(
            in_channels, 
            out_channels // heads,  # Divide output channels by number of heads
            heads=heads, 
            dropout=dropout,
            concat=True
        )
        
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.use_residual = use_residual
        
        # Residual connection if dimensions match
        self.residual = nn.Identity() if in_channels == out_channels and use_residual else None
        
    def reset_parameters(self):
        self.gat.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()
            
    def forward(self, x, edge_index, edge_weight=None):
        # Apply GAT layer
        out = self.gat(x, edge_index, edge_weight)
        
        # Apply batch normalization if specified
        if self.bn is not None:
            out = self.bn(out)
        
        # Apply residual connection if dimensions match
        if self.residual is not None:
            out = out + self.residual(x)
            
        return out


class GATBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        heads=8,
        dropout=0.6,
        use_bn=True,
        use_residual=True,
    ):
        super(GATBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            GATLayer(
                in_channels, 
                hidden_channels, 
                heads=heads, 
                dropout=dropout,
                use_bn=use_bn,
                use_residual=False  # First layer can't use residual as dimensions don't match
            )
        )
        
        # Remaining layers
        for _ in range(num_layers - 1):
            self.layers.append(
                GATLayer(
                    hidden_channels, 
                    hidden_channels, 
                    heads=heads, 
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
                x = F.elu(x)
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
        x = F.elu(self.fc1(x))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc2(x))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        
        return x


class SeedGAT(nn.Module):
    """
    Model that processes a graph with GAT layers for different edge types.
    The final predictor takes a concatenation of the target node embeddings
    and the seed node embeddings.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout=0.1, 
                 residual=True, use_seed_emb=True, use_seed_attn=True, seed_dim=None):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.use_seed_emb = use_seed_emb
        self.use_seed_attn = use_seed_attn
        
        # First apply seed attention, then GNN layers
        if self.use_seed_attn:
            self.seed_attn = SeedAttention(in_dim, seed_dim or in_dim)
        
        # GNN layers for different edge types
        self.pos_convs = nn.ModuleList()
        self.neg_convs = nn.ModuleList()
        
        # First layer takes input from seed attention
        first_layer_input_dim = in_dim
        self.pos_convs.append(GATConv(first_layer_input_dim, hidden_dim, num_heads, dropout))
        self.neg_convs.append(GATConv(first_layer_input_dim, hidden_dim, num_heads, dropout))
        
        # Remaining layers
        for _ in range(num_layers - 1):
            self.pos_convs.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads, dropout))
            self.neg_convs.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads, dropout))
        
        # Final predictor takes concatenation of seed embedding and GNN output
        final_input_dim = hidden_dim * num_heads
        if self.use_seed_emb:
            final_input_dim += (seed_dim or in_dim)
        
        self.predictor = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def reset_parameters(self):
        for layer in self.pos_convs:
            layer.reset_parameters()
        for layer in self.neg_convs:
            layer.reset_parameters()
        self.predictor.reset_parameters()
        
    def forward(self, x, pos_edge_index, neg_edge_index, seed_nodes=None, seed_emb=None):
        # Apply seed attention first if enabled
        if self.use_seed_attn and seed_nodes is not None:
            x = self.seed_attn(x, seed_nodes)
        
        # Store the seed embeddings for later use
        seed_features = None
        if self.use_seed_emb and seed_emb is not None:
            seed_features = seed_emb
        
        # Process through GNN layers
        pos_h, neg_h = x, x
        
        for i in range(self.num_layers):
            pos_h_new = self.pos_convs[i](pos_h, pos_edge_index)
            neg_h_new = self.neg_convs[i](neg_h, neg_edge_index)
            
            # Combine positive and negative features
            h = pos_h_new - neg_h_new
            
            # Apply residual connection if enabled
            if self.residual and i > 0:
                h = h + pos_h
            
            # Update for next layer
            pos_h, neg_h = h, h
        
        # Final node representations
        node_features = pos_h
        
        # Concatenate with seed embeddings if enabled
        if self.use_seed_emb and seed_features is not None:
            node_features = torch.cat([node_features, seed_features], dim=1)
        
        # Apply final prediction layer
        return self.predictor(node_features)
    
    def get_attentions(self, x, pos_edge_index, neg_edge_index, seed_nodes=None):
        # Apply seed attention first if enabled
        if self.use_seed_attn and seed_nodes is not None:
            x = self.seed_attn(x, seed_nodes)
        
        # Process through GNN layers and collect attention weights
        pos_h, neg_h = x, x
        pos_attentions = []
        neg_attentions = []
        
        for i in range(self.num_layers):
            pos_h_new, pos_attn = self.pos_convs[i](pos_h, pos_edge_index, return_attention_weights=True)
            neg_h_new, neg_attn = self.neg_convs[i](neg_h, neg_edge_index, return_attention_weights=True)
            
            pos_attentions.append(pos_attn)
            neg_attentions.append(neg_attn)
            
            # Combine positive and negative features
            h = pos_h_new - neg_h_new
            
            # Apply residual connection if enabled
            if self.residual and i > 0:
                h = h + pos_h
            
            # Update for next layer
            pos_h, neg_h = h, h
        
        return pos_attentions, neg_attentions