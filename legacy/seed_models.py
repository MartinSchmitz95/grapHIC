import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.utils import add_self_loops

class SeedModel(nn.Module):
    """
    Base class for seed-based graph neural networks that process heterographs with two edge types.
    """
    def __init__(self, node_features, edge_features, num_layers, dropout, hidden_features):
        super(SeedModel, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_features = hidden_features
        
        # Predictor head for final output
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1)
        )
    
    def forward(self, data):
        # Process each edge type separately
        x = data.x
        edge_index_0 = data.edge_index[:, data.edge_type == 0]  # Edges of type 0
        edge_index_1 = data.edge_index[:, data.edge_type == 1]  # Edges of type 1
        
        edge_weight_0 = data.edge_weight[data.edge_type == 0]
        edge_weight_1 = data.edge_weight[data.edge_type == 1]
        
        # Process with GNNs for each edge type
        x_0 = self.process_edge_type_0(x, edge_index_0, edge_weight_0)
        x_1 = self.process_edge_type_1(x, edge_index_1, edge_weight_1)

        # Concatenate embeddings from both edge types
        x_combined = torch.cat([x_0, x_1], dim=1)
        
        # Apply predictor head
        out = self.predictor(x_combined)
        return out
    
    def process_edge_type_0(self, x, edge_index, edge_weight=None):
        # To be implemented by subclasses
        raise NotImplementedError
    
    def process_edge_type_1(self, x, edge_index, edge_weight=None):
        # To be implemented by subclasses
        raise NotImplementedError


class SeedModel_GCN(SeedModel):
    """
    GCN-based implementation of SeedModel for processing heterographs.
    """
    def __init__(self, node_features, edge_features, num_layers, dropout, hidden_features):
        super(SeedModel_GCN, self).__init__(node_features, edge_features, num_layers, dropout, hidden_features)
        
        # GCN layers for edge type 0
        self.conv0_layers = nn.ModuleList()
        self.conv0_layers.append(GCNConv(node_features, hidden_features))
        for _ in range(num_layers - 1):
            self.conv0_layers.append(GCNConv(hidden_features, hidden_features))
        
        # GCN layers for edge type 1
        self.conv1_layers = nn.ModuleList()
        self.conv1_layers.append(GCNConv(node_features, hidden_features))
        for _ in range(num_layers - 1):
            self.conv1_layers.append(GCNConv(hidden_features, hidden_features))
        
        # Batch normalization layers
        self.batch_norms0 = nn.ModuleList([nn.BatchNorm1d(hidden_features) for _ in range(num_layers)])
        self.batch_norms1 = nn.ModuleList([nn.BatchNorm1d(hidden_features) for _ in range(num_layers)])
    
    def process_edge_type_0(self, x, edge_index, edge_weight=None):
        # Process with GCN for edge type 0
        for i, conv in enumerate(self.conv0_layers):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.batch_norms0[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def process_edge_type_1(self, x, edge_index, edge_weight=None):
        # Process with GCN for edge type 1
        for i, conv in enumerate(self.conv1_layers):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.batch_norms1[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SeedModel_GAT(SeedModel):
    """
    GAT-based implementation of SeedModel for processing heterographs.
    """
    def __init__(self, node_features, edge_features, num_layers, dropout, hidden_features, heads=4):
        super(SeedModel_GAT, self).__init__(node_features, edge_features, num_layers, dropout, hidden_features)
        self.heads = heads
        
        # GAT layers for edge type 0
        self.conv0_layers = nn.ModuleList()
        self.conv0_layers.append(GATConv(node_features, hidden_features // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.conv0_layers.append(GATConv(hidden_features, hidden_features // heads, heads=heads))
        
        # GAT layers for edge type 1
        self.conv1_layers = nn.ModuleList()
        self.conv1_layers.append(GATConv(node_features, hidden_features // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.conv1_layers.append(GATConv(hidden_features, hidden_features // heads, heads=heads))
        
        # Batch normalization layers
        self.batch_norms0 = nn.ModuleList([nn.BatchNorm1d(hidden_features) for _ in range(num_layers)])
        self.batch_norms1 = nn.ModuleList([nn.BatchNorm1d(hidden_features) for _ in range(num_layers)])
    
    def process_edge_type_0(self, x, edge_index, edge_weight=None):
        # Process with GAT for edge type 0
        for i, conv in enumerate(self.conv0_layers):
            # GAT doesn't directly use edge_weight, but we could adapt it if needed
            x = conv(x, edge_index)
            x = self.batch_norms0[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def process_edge_type_1(self, x, edge_index, edge_weight=None):
        # Process with GAT for edge type 1
        for i, conv in enumerate(self.conv1_layers):
            x = conv(x, edge_index)
            x = self.batch_norms1[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class SeedModel_EdgeWeightedGAT(SeedModel):
    """
    GAT-based implementation that explicitly incorporates edge weights.
    """
    def __init__(self, node_features, edge_features, num_layers, dropout, hidden_features, heads=4):
        super(SeedModel_EdgeWeightedGAT, self).__init__(node_features, edge_features, num_layers, dropout, hidden_features)
        self.heads = heads
        
        # Custom edge-weighted GAT layers
        self.conv0_layers = nn.ModuleList()
        self.conv0_layers.append(EdgeWeightedGATConv(node_features, hidden_features // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.conv0_layers.append(EdgeWeightedGATConv(hidden_features, hidden_features // heads, heads=heads))
        
        self.conv1_layers = nn.ModuleList()
        self.conv1_layers.append(EdgeWeightedGATConv(node_features, hidden_features // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.conv1_layers.append(EdgeWeightedGATConv(hidden_features, hidden_features // heads, heads=heads))
        
        # Batch normalization layers
        self.batch_norms0 = nn.ModuleList([nn.BatchNorm1d(hidden_features) for _ in range(num_layers)])
        self.batch_norms1 = nn.ModuleList([nn.BatchNorm1d(hidden_features) for _ in range(num_layers)])
    
    def process_edge_type_0(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.conv0_layers):
            x = conv(x, edge_index, edge_weight)
            x = self.batch_norms0[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def process_edge_type_1(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.conv1_layers):
            x = conv(x, edge_index, edge_weight)
            x = self.batch_norms1[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EdgeWeightedGATConv(MessagePassing):
    """
    Custom GAT convolution that explicitly incorporates edge weights.
    """
    def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0.0):
        super(EdgeWeightedGATConv, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops to edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # If edge_weight is provided, add weights for self-loops
        if edge_weight is not None:
            # Add weight=1.0 for self-loops
            self_loop_weight = torch.ones(x.size(0), device=edge_weight.device)
            edge_weight = torch.cat([edge_weight, self_loop_weight])
        
        # Transform node features
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, edge_index_i, edge_index_j, x_i, x_j, edge_weight=None):
        # Compute attention coefficients
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i)
        
        # Apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # If edge weights are provided, incorporate them into the attention
        if edge_weight is not None:
            alpha = alpha * edge_weight.view(-1, 1)
        
        # Return weighted message
        return x_j * alpha.unsqueeze(-1)
    
    def update(self, aggr_out):
        # Return aggregated node features
        return aggr_out.mean(dim=1)


def softmax(src, index, num_nodes=None):
    """
    Compute softmax of elements in src with respect to index.
    """
    if num_nodes is None:
        num_nodes = index.max().item() + 1
    
    out = src.clone()
    
    # Compute max for numerical stability
    max_value_per_index = torch.zeros(num_nodes, dtype=src.dtype, device=src.device)
    max_value_per_index.scatter_(0, index, src, reduce='max')
    max_value_per_index = max_value_per_index.index_select(0, index)
    out = out - max_value_per_index
    
    # Compute exponential and sum
    out = out.exp()
    sum_per_index = torch.zeros(num_nodes, dtype=out.dtype, device=out.device)
    sum_per_index.scatter_(0, index, out, reduce='sum')
    sum_per_index = sum_per_index.index_select(0, index)
    
    # Divide by sum
    return out / (sum_per_index + 1e-16)
