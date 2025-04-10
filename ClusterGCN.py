import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, HeteroConv
from torch_geometric.utils import remove_self_loops, add_self_loops


class HeteroGCNConv(MessagePassing):
    """
    A GCN convolution layer that handles two edge types separately.
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(HeteroGCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels, out_channels)
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_type):
        # Process edges of type 0
        edge_index_0 = edge_index[:, edge_type == 0]
        
        # Add self-loops to edge_index_0
        edge_index_0, _ = remove_self_loops(edge_index_0)
        edge_index_0, _ = add_self_loops(edge_index_0, num_nodes=x.size(0))
        
        # Apply first convolution with edge type 0
        x_0 = self.propagate(edge_index_0, x=x, edge_type=0)
        x_0 = self.lin(x_0)
        x_0 = F.relu(x_0)
        x_0 = F.dropout(x_0, p=self.dropout, training=self.training)
        
        # Process edges of type 1
        edge_index_1 = edge_index[:, edge_type == 1]
        
        # Add self-loops to edge_index_1
        edge_index_1, _ = remove_self_loops(edge_index_1)
        edge_index_1, _ = add_self_loops(edge_index_1, num_nodes=x.size(0))
        
        # Apply second convolution with edge type 1
        x_1 = self.propagate(edge_index_1, x=x_0, edge_type=1)
        x_1 = self.lin_update(x_1)
        x_1 = F.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)
        
        return x_1
    
    def message(self, x_j, edge_type):
        # We can customize message based on edge_type if needed
        return x_j
    
    def update(self, aggr_out):
        return aggr_out


class ClusterGCN(nn.Module):
    """
    A GCN model that processes a graph with two edge types for homophilic clustering.
    Uses dual embeddings and projection heads for contrastive learning.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.1, projection_dim=None):
        super(ClusterGCN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = dropout
        
        # Input layer
        self.convs.append(
            HeteroConv({
                ('node', 'type0', 'node'): GCNConv(in_channels, hidden_channels),
                ('node', 'type1', 'node'): GCNConv(in_channels, hidden_channels)
            })
        )
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                HeteroConv({
                    ('node', 'type0', 'node'): GCNConv(hidden_channels, hidden_channels),
                    ('node', 'type1', 'node'): GCNConv(hidden_channels, hidden_channels)
                })
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                HeteroConv({
                    ('node', 'type0', 'node'): GCNConv(hidden_channels, out_channels),
                    ('node', 'type1', 'node'): GCNConv(hidden_channels, out_channels)
                })
            )
            self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        # Create two separate embedding layers for dual representation
        self.embedding1 = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Tanh()
        )
        
        self.embedding2 = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.Tanh()
        )
        
        # Create projection heads for contrastive learning
        if projection_dim is None:
            projection_dim = out_channels
            
        # Projection heads for both embeddings
        self.projection_head1 = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, projection_dim)
        )
        
        self.projection_head2 = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, projection_dim)
        )
    
    def forward(self, g):
        x = g.x
        edge_index = g.edge_index
        edge_type = g.edge_type
        
        # Create heterogeneous data structures
        x_dict = {'node': x}
        edge_index_dict = {}
        
        # Split edges by type
        edge_index_0 = edge_index[:, edge_type == 0]
        edge_index_1 = edge_index[:, edge_type == 1]
        
        # Add to dictionary with proper edge type keys
        edge_index_dict[('node', 'type0', 'node')] = edge_index_0
        edge_index_dict[('node', 'type1', 'node')] = edge_index_1
        
        # Apply GCN layers
        for i in range(self.num_layers):
            # Apply heterogeneous convolution
            x_dict = self.convs[i](x_dict, edge_index_dict)
            
            # Extract node features (HeteroConv returns a dict)
            x = x_dict['node']
            
            # Apply batch normalization
            x = self.batch_norms[i](x)
            
            if i < self.num_layers - 1:  # No ReLU after final layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Update the node features dictionary for next layer
            x_dict = {'node': x}
        
        # Final node features
        x = x_dict['node']
        
        # Generate two different embeddings
        embeddings1 = self.embedding1(x)
        embeddings2 = self.embedding2(x)
        
        # Apply L2 normalization to embeddings
        #embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        # Generate projections for contrastive learning
        projections1 = self.projection_head1(embeddings1)
        projections2 = self.projection_head2(embeddings2)
        
        # Apply L2 normalization to projections
        projections1 = F.normalize(projections1, p=2, dim=1)
        projections2 = F.normalize(projections2, p=2, dim=1)
        
        return embeddings1, embeddings2, projections1, projections2


# Example usage
if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data
    
    # Create a sample graph
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
                              [1, 2, 0, 3, 1, 0, 2, 3]], dtype=torch.long)
    
    # Define edge types (0 and 1)
    edge_type = torch.tensor([0, 0, 1, 0, 1, 1, 0, 1], dtype=torch.long)
    
    # Create node features
    x = torch.randn(4, 10)  # 4 nodes, 10-dimensional features
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index)
    data.edge_type = edge_type
    
    # Create model
    model = ClusterGCN(in_channels=10, hidden_channels=16, out_channels=8, num_layers=2)
    
    # Forward pass
    (embeddings1, embeddings2), (projections1, projections2) = model(data)
    
    print("Node embeddings1 shape:", embeddings1.shape)
    print("Node embeddings2 shape:", embeddings2.shape)
    print("Node projections1 shape:", projections1.shape)
    print("Node projections2 shape:", projections2.shape)
