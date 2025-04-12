import os
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import degree
import argparse


def convert_to_single_stranded(nx_graph):
    """
    Convert a multi-stranded graph to a single-stranded graph.
    This is the full complexity version from dataset_object.py.
    
    Returns:
        tuple: (single_stranded_graph, new_to_old_mapping)
    """
    # Assert that nx_graph is a multigraph
    if not isinstance(nx_graph, nx.MultiGraph) and not isinstance(nx_graph, nx.MultiDiGraph):
        raise TypeError("Input graph must be a NetworkX MultiGraph or MultiDiGraph")
    
    # Create a new MultiGraph
    single_stranded = nx.MultiGraph()
    
    # Create a mapping from old node IDs to new node IDs
    old_to_new = {}
    new_to_old = {}
    new_id = 0
    
    # First pass: create mapping for even nodes only
    for node in sorted(nx_graph.nodes()):
        if node % 2 == 0:  # Keep even nodes
            old_to_new[node] = new_id
            new_to_old[new_id] = node
            new_id += 1
    
    # Copy node attributes for even nodes with new IDs
    for old_node, new_node in old_to_new.items():
        single_stranded.add_node(new_node)
        single_stranded.nodes[new_node].update(nx_graph.nodes[old_node])
    
    # Process edges
    edge_weights = {}  # Dictionary to store accumulated weights
    final_edge_types = {}
    duplicate_counts = {'hic': 0, 'overlap': 0}
    
    for u, v, key, data in nx_graph.edges(data=True, keys=True):
        edge_type = data['type']
        if isinstance(edge_type, list):
            edge_type = edge_type[0]
        weight = data.get('weight', 1.0)
        
        # Convert nodes to their even complements if they're odd
        u_even = u if u % 2 == 0 else u - 1
        v_even = v if v % 2 == 0 else v - 1
        
        # Get new IDs for the even nodes
        u_new = old_to_new[u_even]
        v_new = old_to_new[v_even]
        
        # Sort the node IDs to ensure consistent ordering
        sorted_nodes = tuple(sorted((u_new, v_new)))
        # Create unique edge identifier as a tuple
        edge_id = sorted_nodes + (str(edge_type),)

        if edge_id not in edge_weights:
            edge_weights[edge_id] = weight
            final_edge_types[edge_type] = final_edge_types.get(edge_type, 0) + 1
        else:
            if edge_type == 'hic':  # Sum weights for HiC edges
                edge_weights[edge_id] += weight
            duplicate_counts[edge_type] += 1

    # Add edges with accumulated weights
    for edge_id, weight in edge_weights.items():
        u_new, v_new, edge_type = edge_id[0], edge_id[1], edge_id[2]
        single_stranded.add_edge(u_new, v_new, type=edge_type, weight=weight)

    print(f"Converted graph: {single_stranded.number_of_nodes()} nodes, {single_stranded.number_of_edges()} edges")
    print(f"Final edge type distribution: {final_edge_types}")
    print(f"Duplicate edges processed: {duplicate_counts}")
    
    return single_stranded, new_to_old


def add_hic_neighbor_weights(nx_graph):
    """
    Add a node feature that sums the weights of all HiC edges connected to each node.
    This is the full complexity version from dataset_object.py.
    """
    hic_weights_sum = {}
    
    # Iterate through all nodes
    for node in nx_graph.nodes():
        total_weight = 0
        # Get all edges connected to this node
        for _, neighbor, key, data in nx_graph.edges(node, data=True, keys=True):
            edge_type = data['type']
            # Ensure edge_type is a string
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            # Sum weights only for HiC edges
            if edge_type == 'hic':
                total_weight += data['weight']
        hic_weights_sum[node] = total_weight
    
    # Add the feature to the graph
    nx.set_node_attributes(nx_graph, hic_weights_sum, 'hic_neighbor_weight')
    
    print(f"Added HiC neighbor weights feature to {len(hic_weights_sum)} nodes")
    
    return nx_graph


def save_to_pyg(nx_graph, output_file, depth):
    """
    Convert a NetworkX graph to a PyTorch Geometric graph and save it.
    This is a simplified version of the save_to_dgl_and_pyg function from dataset_object.py.
    """
    print(f"Total nodes in graph: {nx_graph.number_of_nodes()}")
    # Assert that nx_graph is a multigraph
    assert isinstance(nx_graph, nx.MultiGraph) or isinstance(nx_graph, nx.MultiDiGraph), "Graph must be a NetworkX MultiGraph or MultiDiGraph"
    
    # Get number of nodes
    num_nodes = nx_graph.number_of_nodes()
    
    # Initialize lists for edges, types and weights
    edge_list = []
    edge_types = []
    edge_weights = []
    
    # Process each edge type
    edge_type_map = {'overlap': 0, 'hic': 1}
    
    # Convert edges for each type
    for (u, v, key, data) in nx_graph.edges(data=True, keys=True):
        edge_type = data.get('type')  # Get edge type
        # Ensure edge_type is a string
        if isinstance(edge_type, list):
            edge_type = edge_type[0]  # Take first element if it's a list
        elif edge_type is None:
            print(f"Warning: Edge ({u}, {v}) has no type, defaulting to 'overlap'")
            edge_type = 'overlap'
        
        if edge_type not in edge_type_map:
            print(f"Warning: Unknown edge type {edge_type}, defaulting to 'overlap'")
            edge_type = 'overlap'
            
        edge_list.append([u, v])
        edge_types.append(edge_type_map[edge_type])
        edge_weights.append(data.get('weight', 1.0))  # Default weight to 1.0 if not found
        
    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    
    # Z-score normalize edge weights separately for each type
    for type_idx in range(len(edge_type_map)):
        
        # Create mask for current edge type
        type_mask = edge_type == type_idx
        
        if not torch.any(type_mask):
            continue
            
        # Get weights for current type
        type_weights = edge_weight[type_mask]
        
        # Compute mean and std for current type
        type_mean = torch.mean(type_weights)
        type_std = torch.std(type_weights)
        
        # Normalize weights for current type if std is not 0
        if type_std != 0:
            edge_weight[type_mask] = (type_weights - type_mean) / type_std
            
        print(f"\nEdge weight statistics after normalization for type {list(edge_type_map.keys())[type_idx]}:")
        print(f"Mean: {torch.mean(edge_weight[type_mask])}")
        print(f"Std:  {torch.std(edge_weight[type_mask])}")
    
    # Now compute degrees after edge_index is created
    overlap_degrees = degree(edge_index[0][edge_type == 0], num_nodes=num_nodes)  # For overlap edges
    hic_degrees = degree(edge_index[0][edge_type == 1], num_nodes=num_nodes)  # For HiC edges
    
    # Add degree information to the graph
    for node in range(num_nodes):
        nx_graph.nodes[node]['overlap_degree'] = float(overlap_degrees[node])
        nx_graph.nodes[node]['hic_degree'] = float(hic_degrees[node])
    
    node_attrs = ['overlap_degree', 'hic_degree', 'read_length', 'hic_neighbor_weight', 'support']
    
    # Create node features using all features in node_attrs
    features = []
    for node in range(num_nodes):
        node_features = []
        for feat in node_attrs:
            if feat in nx_graph.nodes[node]:
                node_features.append(float(nx_graph.nodes[node][feat]))
            else:
                node_features.append(0.0)  # Default value if attribute is missing

        features.append(node_features)
    x = torch.tensor(features, dtype=torch.float)

    # Extract ground truth haplotype labels if available
    if 'gt_hap' in nx_graph.nodes[0]:
        gt_hap = [nx_graph.nodes[node]['gt_hap'] for node in range(num_nodes)]
        gt_tensor = torch.tensor(gt_hap, dtype=torch.long)
    else:
        gt_tensor = None
    
    # Extract chromosome numbers if available
    if 'read_chr' in nx_graph.nodes[0]:
        chr_numbers = []
        for node in range(num_nodes):
            chr_str = nx_graph.nodes[node]['read_chr']
            # Extract number from string like 'chrX' or 'X'
            try:
                chr_num = int(chr_str.replace('chr', ''))
            except ValueError:
                # Handle non-numeric chromosomes like 'chrX'
                chr_num = 0
            chr_numbers.append(chr_num)
        chr_tensor = torch.tensor(chr_numbers, dtype=torch.long)
    else:
        chr_tensor = None
    
    # Store original node IDs
    # Get a list of all node IDs in the NetworkX graph
    original_node_ids = list(nx_graph.nodes())
    original_id_tensor = torch.tensor(original_node_ids, dtype=torch.long)
    
    # Create PyG Data object
    data_dict = {
        'x': x,
        'edge_index': edge_index,
        'edge_type': edge_type,
        'edge_weight': edge_weight,
        'original_id': original_id_tensor  # Add original node IDs
    }
    
    if gt_tensor is not None:
        data_dict['y'] = gt_tensor
    
    if chr_tensor is not None:
        data_dict['chr'] = chr_tensor
    
    pyg_data = Data(**data_dict)

    # Apply normalization to specific features
    if 'overlap_degree' in node_attrs:
        pyg_data.x[:, node_attrs.index('overlap_degree')] /= 10
    if 'hic_degree' in node_attrs:
        pyg_data.x[:, node_attrs.index('hic_degree')] /= 100
    if 'read_length' in node_attrs:
        pyg_data.x[:, node_attrs.index('read_length')] /= 10000
    if 'hic_neighbor_weight' in node_attrs:
        pyg_data.x[:, node_attrs.index('hic_neighbor_weight')] /= 1000
    if 'support' in node_attrs:
        # Since we don't have depth parameter, use a default value or calculate from graph  # Default to 30 if not specified
        pyg_data.x[:, node_attrs.index('support')] /= depth

    pyg_data = normalize_edge_weights(pyg_data, edge_type=1)
    # Save PyG graph
    torch.save(pyg_data, output_file)
    print(f"Saved PyG graph with {num_nodes} nodes and {len(edge_list)} edges to {output_file}")
    

def normalize_edge_weights(g, edge_type=1, device=None):
    """
    Normalize edge weights of a specific edge type to [0,1] range.
    
    Args:
        g: Graph object
        edge_type: Type of edges to normalize (default: 1 for Hi-C edges)
        device: Device to run computation on
        
    Returns:
        Graph with normalized edge weights
    """
    
    # Get edge weights for specified edge type
    edge_mask = g.edge_type == edge_type
    edge_weights = g.edge_weight[edge_mask]
    
    # Normalize edge weights to [0,1] range
    if len(edge_weights) > 0:  # Only normalize if we have edges of the specified type
        min_weight = edge_weights.min()
        max_weight = edge_weights.max()
        if max_weight > min_weight:  # Avoid division by zero
            normalized_weights = (edge_weights - min_weight) / (max_weight - min_weight)
            g.edge_weight[edge_mask] = normalized_weights.to(device)
            print(f"Normalized edge weights range: [{normalized_weights.min():.4f}, {normalized_weights.max():.4f}]")
        else:
            print("All edge weights are equal, no normalization needed")
    else:
        print(f"No edges of type {edge_type} found")
    
    return g

def nx_to_pyg(nx_graph, output_file, depth):
    """
    Convert a NetworkX graph to a PyTorch Geometric graph.
    This combines the three functions from dataset_main.py.
    
    Returns:
        tuple: (pyg_data, new_to_old_mapping)
    """
    # Step 1: Convert to single stranded
    nx_multi_reduced, new_to_old = convert_to_single_stranded(nx_graph)
    
    # Step 2: Add HiC neighbor weights
    nx_multi_reduced = add_hic_neighbor_weights(nx_multi_reduced)
    
    # Step 3: Save to PyG
    save_to_pyg(nx_multi_reduced, output_file, depth)
    
    return new_to_old

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert NetworkX graph to PyTorch Geometric graph')
    parser.add_argument('--input', type=str, required=True, help='Input NetworkX graph file (.pkl)')
    parser.add_argument('--output', type=str, required=True, help='Output PyG graph file (.pt)')
    parser.add_argument('--mapping', type=str, help='Output file for node ID mapping (.pkl)')
    parser.add_argument('--depth', type=int, default=40, 
                        help='Sequencing depth for normalizing support feature')
    
    args = parser.parse_args()
    
    # Load NetworkX graph
    import pickle
    with open(args.input, 'rb') as f:
        nx_graph = pickle.load(f)
    
    # Convert to PyG
    nx_to_pyg(nx_graph, args.output, args.depth)
    
