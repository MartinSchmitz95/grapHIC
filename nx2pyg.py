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
    
    """# Z-score normalize edge weights separately for each type
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
        print(f"Std:  {torch.std(edge_weight[type_mask])}")"""
    
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
        print(f"Ground truth haplotype labels found for {len(gt_hap)} nodes")
    else:
        print("No ground truth haplotype labels found")
        exit()
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
    #pyg_data.y = gt_tensor
    #pyg_data.chr = chr_tensor

    #pyg_data = normalize_edge_weights(pyg_data, edge_type=1)
    pyg_data = gating_edge_weights(pyg_data, edge_type=1)

    # Save PyG graph
    torch.save(pyg_data, output_file)
    print(f"Saved PyG graph with {num_nodes} nodes and {len(edge_list)} edges to {output_file}")
    

def normalize_edge_weights(g, edge_type=1, device=None):
    """
    Normalize edge weights of a specific edge type by dividing by 10000.
    
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
    
    # Simply divide the edge weights by 10000
    if len(edge_weights) > 0:  # Only normalize if we have edges of the specified type
        # Divide by 10000
        g.edge_weight[edge_mask] = edge_weights / 10000
        print(f"Divided edge weights by 10000. New range: [{g.edge_weight[edge_mask].min():.4f}, {g.edge_weight[edge_mask].max():.4f}]")
    else:
        print(f"No edges of type {edge_type} found")
    
    return g

def gating_edge_weights(g, edge_type=1):
    """
    Normalize edge weights by dividing each edge weight by the sum of all edge weights 
    connected to its source and target nodes, then averaging these two values.
    
    Args:
        g: PyG graph object
        edge_type: Type of edges to normalize (default: 1 for Hi-C edges)
        
    Returns:
        Graph with normalized edge weights
    """
    # Get edge indices and weights for specified edge type
    edge_mask = g.edge_type == edge_type
    if not torch.any(edge_mask):
        print(f"No edges of type {edge_type} found")
        return g
    
    edge_indices = g.edge_index[:, edge_mask]
    edge_weights = g.edge_weight[edge_mask]
    
    # Calculate sum of edge weights for each node
    node_weight_sums = torch.zeros(g.num_nodes, dtype=torch.float)
    
    # For each edge, add its weight to both source and target nodes' sums
    for i in range(edge_indices.size(1)):
        src, dst = edge_indices[0, i], edge_indices[1, i]
        weight = edge_weights[i]
        node_weight_sums[src] += weight
        node_weight_sums[dst] += weight
    
    # Avoid division by zero
    node_weight_sums = torch.clamp(node_weight_sums, min=1e-10)
    
    # Normalize each edge weight by the sum of weights for its source and target nodes
    new_edge_weights = torch.zeros_like(edge_weights)
    for i in range(edge_indices.size(1)):
        src, dst = edge_indices[0, i], edge_indices[1, i]
        weight = edge_weights[i]
        
        # Normalize by source and target node weight sums
        src_norm = weight / node_weight_sums[src]
        dst_norm = weight / node_weight_sums[dst]
        
        # Average the two normalized weights
        new_edge_weights[i] = (src_norm + dst_norm) / 2.0
    
    # Update the edge weights in the graph
    g.edge_weight[edge_mask] = new_edge_weights
    
    # Print statistics
    print(f"\nEdge weight statistics after gating normalization for type {edge_type}:")
    print(f"Min: {new_edge_weights.min().item():.6f}")
    print(f"Max: {new_edge_weights.max().item():.6f}")
    print(f"Mean: {new_edge_weights.mean().item():.6f}")
    print(f"Std: {new_edge_weights.std().item():.6f}")
    
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

def analyze_hic(nx_graph):
    """
    Analyze Hi-C edges within and between chromosomes.
    
    Args:
        nx_graph: NetworkX graph with 'read_chr' node attribute
        
    Returns:
        tuple: (within_chrom_counts, cross_chrom_counts)
            - within_chrom_counts: Dictionary mapping chromosome to count of Hi-C edges within that chromosome
            - cross_chrom_counts: Dictionary mapping chromosome pairs to count of Hi-C edges between them
    """
    # Check if graph has chromosome information
    if 'read_chr' not in nx_graph.nodes[list(nx_graph.nodes())[0]]:
        print("Error: Graph nodes do not have 'read_chr' attribute")
        return {}, {}
    
    # Get chromosome for each node
    node_to_chrom = {}
    chroms = set()
    for node in nx_graph.nodes():
        chrom = nx_graph.nodes[node]['read_chr']
        node_to_chrom[node] = chrom
        chroms.add(chrom)
    
    # Initialize counters
    within_chrom_counts = {chrom: 0 for chrom in chroms}
    cross_chrom_counts = {}
    
    # Count Hi-C edges
    total_hic_edges = 0
    for u, v, key, data in nx_graph.edges(data=True, keys=True):
        edge_type = data.get('type')
        # Ensure edge_type is a string
        if isinstance(edge_type, list):
            edge_type = edge_type[0]
            
        if edge_type == 'hic':
            total_hic_edges += 1
            chrom_u = node_to_chrom[u]
            chrom_v = node_to_chrom[v]
            
            if chrom_u == chrom_v:
                # Within-chromosome edge
                within_chrom_counts[chrom_u] += 1
            else:
                # Cross-chromosome edge
                # Sort chromosomes to ensure consistent counting
                chrom_pair = tuple(sorted([chrom_u, chrom_v]))
                if chrom_pair not in cross_chrom_counts:
                    cross_chrom_counts[chrom_pair] = 0
                cross_chrom_counts[chrom_pair] += 1
    
    # Print summary
    print(f"\nHi-C Edge Analysis:")
    print(f"Total Hi-C edges: {total_hic_edges}")
    
    print("\nWithin-chromosome Hi-C edges:")
    within_total = sum(within_chrom_counts.values())
    for chrom, count in sorted(within_chrom_counts.items()):
        print(f"  {chrom}: {count} edges")
    print(f"Total within-chromosome: {within_total} ({within_total/total_hic_edges*100:.1f}%)")
    
    print("\nCross-chromosome Hi-C edges:")
    cross_total = sum(cross_chrom_counts.values())
    for chrom_pair, count in sorted(cross_chrom_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {chrom_pair[0]} ↔ {chrom_pair[1]}: {count} edges")
    print(f"Total cross-chromosome: {cross_total} ({cross_total/total_hic_edges*100:.1f}%)")
    
    return within_chrom_counts, cross_chrom_counts

def check_bitwise_xor_nodes(nx_graph):
    """
    Check if for each node n in the graph, node n^1 (bitwise XOR) exists.
    Also compares the degrees of node pairs.
    
    Args:
        nx_graph: NetworkX graph
        
    Returns:
        dict: Statistics about node pairs
    """
    nodes = list(nx_graph.nodes())
    total_nodes = len(nodes)
    
    # Count nodes with and without XOR pairs
    has_xor_pair = 0
    missing_xor_pair = 0
    xor_pairs = {}
    
    # Degree comparison statistics
    same_degree = 0
    different_degree = 0
    degree_differences = []
    
    for node in nodes:
        xor_node = node ^ 1  # Bitwise XOR with 1
        if xor_node in nx_graph:
            has_xor_pair += 1
            xor_pairs[node] = xor_node
            
            # Compare degrees
            node_degree = nx_graph.degree(node)
            xor_node_degree = nx_graph.degree(xor_node)
            
            if node_degree == xor_node_degree:
                same_degree += 1
            else:
                different_degree += 1
                degree_differences.append(abs(node_degree - xor_node_degree))
        else:
            missing_xor_pair += 1
    
    # Calculate statistics
    stats = {
        'total_nodes': total_nodes,
        'nodes_with_xor_pair': has_xor_pair,
        'nodes_missing_xor_pair': missing_xor_pair,
        'percentage_with_pair': (has_xor_pair / total_nodes) * 100 if total_nodes > 0 else 0,
        'same_degree_pairs': same_degree,
        'different_degree_pairs': different_degree
    }
    
    # Calculate average degree difference if there are any differences
    if degree_differences:
        stats['avg_degree_difference'] = sum(degree_differences) / len(degree_differences)
        stats['max_degree_difference'] = max(degree_differences)
    
    # Print summary
    print(f"\nBitwise XOR Node Analysis:")
    print(f"Total nodes: {total_nodes}")
    print(f"Nodes with XOR pair (n^1): {has_xor_pair} ({stats['percentage_with_pair']:.1f}%)")
    print(f"Nodes missing XOR pair: {missing_xor_pair}")
    
    # Print degree comparison
    if has_xor_pair > 0:
        print(f"\nDegree Comparison for XOR Pairs:")
        print(f"Pairs with same degree: {same_degree} ({same_degree/has_xor_pair*100:.1f}%)")
        print(f"Pairs with different degree: {different_degree} ({different_degree/has_xor_pair*100:.1f}%)")
        
        if different_degree > 0:
            print(f"Average degree difference: {stats['avg_degree_difference']:.2f}")
            print(f"Maximum degree difference: {stats['max_degree_difference']}")
    
    return stats, xor_pairs

def analyze_hic_edge_weights(nx_graph):
    """
    Analyze the weight distribution of Hi-C edges in the graph.
    
    Args:
        nx_graph: NetworkX graph with 'type' and 'weight' edge attributes
        
    Returns:
        dict: Statistics about Hi-C edge weights
    """
    hic_weights = []
    
    # Collect weights of all Hi-C edges
    for u, v, key, data in nx_graph.edges(data=True, keys=True):
        edge_type = data.get('type')
        # Ensure edge_type is a string
        if isinstance(edge_type, list):
            edge_type = edge_type[0]
            
        if edge_type == 'hic' and 'weight' in data:
            hic_weights.append(data['weight'])
    
    # Calculate statistics
    if hic_weights:
        min_weight = min(hic_weights)
        max_weight = max(hic_weights)
        avg_weight = sum(hic_weights) / len(hic_weights)
        
        print("\nHi-C Edge Weight Statistics:")
        print(f"Total Hi-C edges with weights: {len(hic_weights)}")
        print(f"Minimum weight: {min_weight}")
        print(f"Maximum weight: {max_weight}")
        print(f"Average weight: {avg_weight:.4f}")
        
        # Calculate percentiles
        hic_weights.sort()
        p25 = hic_weights[int(len(hic_weights) * 0.25)]
        p50 = hic_weights[int(len(hic_weights) * 0.5)]
        p75 = hic_weights[int(len(hic_weights) * 0.75)]
        p90 = hic_weights[int(len(hic_weights) * 0.9)]
        p99 = hic_weights[int(len(hic_weights) * 0.99)]
        
        print(f"25th percentile: {p25}")
        print(f"50th percentile (median): {p50}")
        print(f"75th percentile: {p75}")
        print(f"90th percentile: {p90}")
        print(f"99th percentile: {p99}")
        
        return {
            'count': len(hic_weights),
            'min': min_weight,
            'max': max_weight,
            'avg': avg_weight,
            'p25': p25,
            'p50': p50,
            'p75': p75,
            'p90': p90,
            'p99': p99
        }
    else:
        print("\nNo Hi-C edges with weights found in the graph")
        return {'count': 0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert NetworkX graph to PyTorch Geometric graph')
    parser.add_argument('--input', type=str, required=True, help='Input NetworkX graph file (.pkl)')
    parser.add_argument('--output', type=str, required=True, help='Output PyG graph file (.pt)')
    parser.add_argument('--mapping', type=str, help='Output file for node ID mapping (.pkl)')
    parser.add_argument('--depth', type=int, default=40, 
                        help='Sequencing depth for normalizing support feature')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze Hi-C edges within and between chromosomes')
    parser.add_argument('--check-xor', action='store_true',
                        help='Check if for each node n, node n^1 exists')
    parser.add_argument('--check-weights', action='store_true',
                        help='Check the weight distribution of Hi-C edges')
    
    args = parser.parse_args()
    
    # Load NetworkX graph
    import pickle
    with open(args.input, 'rb') as f:
        nx_graph = pickle.load(f)

    # Check bitwise XOR nodes if requested
    if args.check_xor:
        check_bitwise_xor_nodes(nx_graph)
        exit()

    # Check Hi-C edge weights if requested
    if args.check_weights:
        analyze_hic_edge_weights(nx_graph)
        exit()

    # Analyze Hi-C edges if requested
    if args.analyze:
        # Get unique node attribute names
        attr_names = set()
        for node, attrs in nx_graph.nodes(data=True):
            attr_names.update(attrs.keys())
        
        print("\nNode attributes:")
        print("Found these node attributes in the graph:")
        for attr in sorted(attr_names):
            print(f"  {attr}")
        analyze_hic(nx_graph)
        exit()

    # Convert to PyG
    nx_to_pyg(nx_graph, args.output, args.depth)
    
