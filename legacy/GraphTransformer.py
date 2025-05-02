import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GINEConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool


class GraphTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        heads=4,
        beta=True,
        use_bn=True,
        use_residual=True,
    ):
        super().__init__()
        
        # Input projection
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        
        # Transformer layers
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(hidden_channels, hidden_channels//heads, heads=heads, beta=beta))
        
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels, hidden_channels//heads, heads=heads, beta=beta))
        
        # Final layer with concat=False to average multi-head output
        self.convs.append(TransformerConv(hidden_channels, hidden_channels//heads, heads=heads, beta=beta, concat=False))
        
        # Layer normalization after each transformer layer
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1)])
        
        # Output projection
        self.lin_out = nn.Linear(hidden_channels//heads, out_channels)
        
        # Configuration
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.num_layers = num_layers

    def reset_parameters(self):
        """
        Resets the parameters of all layers in the model.
        """
        self.lin_in.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, data):
        """
        Forward pass through the Graph Transformer.
        
        Args:
            data: PyG data object containing node features and edge indices
            
        Returns:
            Node embeddings after transformation
        """
        x, edge_index = data.x, data.edge_index
        
        # Initial feature projection
        x = self.lin_in(x)
        
        # Process through transformer layers
        for i in range(self.num_layers - 1):
            # Store for residual connection
            x_res = x
            
            # Apply transformer convolution
            x = self.convs[i](x, edge_index)
            
            # Apply normalization
            if self.use_bn:
                x = self.norms[i](x)
            
            # Apply activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Add residual connection if configured
            if self.use_residual:
                x = x + x_res
        
        # Final transformer layer
        x = self.convs[-1](x, edge_index)
        
        # Output projection
        x = self.lin_out(x)
        
        return x


class GraphTransformer2(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        heads=4,
        beta=True,
        use_bn=True,
        use_residual=True,
        use_pe=False,
        pe_dim=0,
    ):
        super().__init__()
        
        # Configuration
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.use_pe = use_pe
        
        # Input projection
        self.lin_in = nn.Linear(in_channels + pe_dim if use_pe else in_channels, hidden_channels)
        
        # Positional encoding projection if used
        if use_pe:
            self.pe_lin = nn.Linear(pe_dim, pe_dim)
            self.pe_norm = nn.BatchNorm1d(pe_dim)
        
        # GPS-style layers combining local MPNN and global attention
        self.local_convs = nn.ModuleList()
        self.global_attns = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.norms3 = nn.ModuleList()
        self.mlps = nn.ModuleList()
        
        for _ in range(num_layers):
            # Local message passing
            self.local_convs.append(TransformerConv(
                hidden_channels, 
                hidden_channels//heads, 
                heads=heads, 
                beta=beta
            ))
            
            # Global attention (simplified version)
            self.global_attns.append(nn.MultiheadAttention(
                hidden_channels, 
                num_heads=heads, 
                dropout=dropout
            ))
            
            # Layer norms
            self.norms1.append(nn.LayerNorm(hidden_channels))
            self.norms2.append(nn.LayerNorm(hidden_channels))
            self.norms3.append(nn.LayerNorm(hidden_channels))
            
            # MLP after combining local and global
            self.mlps.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels)
            ))
        
        # Output projection
        self.lin_out = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        """
        Resets the parameters of all layers in the model.
        """
        self.lin_in.reset_parameters()
        if self.use_pe:
            self.pe_lin.reset_parameters()
            self.pe_norm.reset_parameters()
        
        for conv in self.local_convs:
            conv.reset_parameters()
        
        # Reset parameters for global attention, norms, and MLPs
        for i in range(self.num_layers):
            # Reset MultiheadAttention
            self.global_attns[i].reset_parameters()
            
            # Reset LayerNorms
            self.norms1[i].reset_parameters()
            self.norms2[i].reset_parameters()
            self.norms3[i].reset_parameters()
            
            # Reset MLP layers
            for layer in self.mlps[i]:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        self.lin_out.reset_parameters()

    def forward(self, data):
        """
        Forward pass through the Graph Transformer.
        
        Args:
            data: PyG data object containing node features, edge indices, and optionally PE
            
        Returns:
            Node embeddings after transformation
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Process positional encodings if available
        if self.use_pe and hasattr(data, 'pe'):
            pe = self.pe_norm(data.pe)
            pe = self.pe_lin(pe)
            x = torch.cat([x, pe], dim=-1)
        
        # Initial feature projection
        x = self.lin_in(x)
        
        # Process through GPS-style layers
        for i in range(self.num_layers):
            # Store original for residual
            x_orig = x
            
            # 1. Local MPNN
            x_local = self.local_convs[i](x, edge_index)
            if self.use_residual:
                x_local = x_local + x_orig
            x_local = F.dropout(x_local, p=self.dropout, training=self.training)
            if self.use_bn:
                x_local = self.norms1[i](x_local)
            
            # 2. Global attention
            # Convert to dense batch for attention
            if batch is not None:
                x_dense, mask = to_dense_batch(x, batch)
                # Apply attention (B, N, C) format
                x_attn, _ = self.global_attns[i](x_dense, x_dense, x_dense, key_padding_mask=~mask)
                # Convert back to sparse
                x_global = x_attn[mask]
            else:
                # If no batch info, treat as single graph
                x_global, _ = self.global_attns[i](x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
                x_global = x_global.squeeze(0)
            
            if self.use_residual:
                x_global = x_global + x_orig
            x_global = F.dropout(x_global, p=self.dropout, training=self.training)
            if self.use_bn:
                x_global = self.norms2[i](x_global)
            
            # 3. Combine local and global
            x = x_local + x_global
            
            # 4. Apply MLP
            x_mlp = self.mlps[i](x)
            if self.use_residual:
                x = x + x_mlp
            else:
                x = x_mlp
                
            if self.use_bn:
                x = self.norms3[i](x)
        
        # Output projection
        x = self.lin_out(x)
        
        return x

