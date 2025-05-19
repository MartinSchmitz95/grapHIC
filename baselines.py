import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
import numpy as np
import networkx as nx
from networkx.algorithms.community import label_propagation_communities
from networkx.algorithms.community import greedy_modularity_communities


def spectral_clustering(graph, n_clusters=2, balance_clusters=False):
    # Convert PyG graph to adjacency matrix
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    
    # Create empty adjacency matrix
    adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    
    # If edge weights are available in the graph
    if hasattr(graph, 'edge_weight') and graph.edge_weight is not None:
        # Create weighted adjacency matrix
        edge_weight = graph.edge_weight
        adj[edge_index[0], edge_index[1]] = edge_weight
    else:
        # Create binary adjacency matrix
        adj[edge_index[0], edge_index[1]] = 1
        print("No edge weights available, using binary adjacency matrix")
        exit()
    # Convert adjacency matrix to numpy for sklearn
    adj_np = adj.cpu().numpy()
    
    # Apply spectral clustering
    clustering = SpectralClustering(n_clusters=n_clusters, 
                                   affinity='precomputed',
                                   random_state=0)
    
    # Fit the model to the adjacency matrix
    cluster_labels = clustering.fit_predict(adj_np)
    
    # Balance the clusters if requested
    if balance_clusters:
        cluster_labels = _balance_clusters(cluster_labels, n_clusters, adj_np)
    
    return cluster_labels.tolist()

def _balance_clusters(labels, n_clusters, adj_matrix):
    """
    Balance clusters to have approximately equal sizes.
    
    Args:
        labels: Initial cluster labels
        n_clusters: Number of clusters
        adj_matrix: Adjacency matrix of the graph
        
    Returns:
        Balanced cluster labels
    """
    n_nodes = len(labels)
    target_size = n_nodes // n_clusters
    
    # Count nodes in each cluster
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    
    # Create a copy of labels to modify
    new_labels = labels.copy()
    
    # Identify clusters that are too large and too small
    large_clusters = [i for i, size in enumerate(cluster_sizes) if size > target_size]
    small_clusters = [i for i, size in enumerate(cluster_sizes) if size < target_size]
    
    while large_clusters and small_clusters:
        large_cluster = large_clusters[0]
        small_cluster = small_clusters[0]
        
        # Find nodes to move from large to small cluster
        nodes_in_large = np.where(new_labels == large_cluster)[0]
        
        # Calculate "cost" of moving each node (sum of edges to current cluster - sum of edges to target cluster)
        costs = []
        for node in nodes_in_large:
            # Connections to current cluster (excluding self)
            current_connections = sum(adj_matrix[node, j] for j in nodes_in_large if j != node)
            
            # Connections to target cluster
            target_connections = sum(adj_matrix[node, j] for j in np.where(new_labels == small_cluster)[0])
            
            # Cost is the difference (higher means more connected to current cluster)
            costs.append((current_connections - target_connections, node))
        
        # Sort by cost (move nodes with lowest cost first)
        costs.sort()
        
        # Move the node with the lowest cost
        node_to_move = costs[0][1]
        new_labels[node_to_move] = small_cluster
        
        # Update cluster sizes
        cluster_sizes[large_cluster] -= 1
        cluster_sizes[small_cluster] += 1
        
        # Check if clusters are now balanced
        if cluster_sizes[large_cluster] <= target_size:
            large_clusters.pop(0)
        if cluster_sizes[small_cluster] >= target_size:
            small_clusters.pop(0)
    
    return new_labels

def _process_communities(communities, n_clusters, n_nodes, algorithm_name):
    """
    Process communities to ensure exactly n_clusters are returned.
    
    Args:
        communities: List of communities (each community is a set of nodes)
        n_clusters: Number of clusters to form
        n_nodes: Total number of nodes in the graph
        algorithm_name: Name of the algorithm (for warning messages)
        
    Returns:
        List of cluster labels
    """
    num_communities = len(communities)
    
    # If we already have the desired number of clusters, convert to node labels
    if num_communities == n_clusters:
        # Convert communities to node labels
        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_to_community[node] = i
        
        # Create ordered list of labels
        return [node_to_community.get(i, 0) for i in range(n_nodes)]
    
    # If we have fewer communities than requested, split largest communities
    if num_communities < n_clusters:
        print(f"Warning: {algorithm_name} found only {num_communities} communities, fewer than the requested {n_clusters}")
        # Sort communities by size (largest first)
        communities.sort(key=len, reverse=True)
        
        # Split largest communities until we have n_clusters
        while len(communities) < n_clusters:
            largest = list(communities[0])
            # Split the largest community into two roughly equal parts
            split_point = len(largest) // 2
            community1 = set(largest[:split_point])
            community2 = set(largest[split_point:])
            # Replace the largest community with the two new ones
            communities[0] = community1
            communities.append(community2)
            # Re-sort communities
            communities.sort(key=len, reverse=True)
    
    # If we have more communities than requested, merge smallest communities
    if num_communities > n_clusters:
        # Sort communities by size (smallest first)
        communities.sort(key=len)
        
        # Merge smallest communities until we have n_clusters
        while len(communities) > n_clusters:
            # Merge the two smallest communities
            smallest1 = communities[0]
            smallest2 = communities[1]
            merged = smallest1.union(smallest2)
            # Remove the two smallest communities and add the merged one
            communities.pop(0)
            communities.pop(0)
            communities.append(merged)
            # Re-sort communities
            communities.sort(key=len)
    
    # Convert communities to node labels
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i
    
    # Create ordered list of labels
    return [node_to_community.get(i, 0) for i in range(n_nodes)]

def label_propagation(graph, n_clusters=2, max_iter=1000, tol=1e-6, balance_clusters=False):
    # Convert PyG graph to NetworkX graph
    G = to_networkx(graph)
    
    # Get communities using NetworkX's implementation
    communities = list(label_propagation_communities(G))
    
    # Process communities to ensure exactly n_clusters
    labels = _process_communities(communities, n_clusters, graph.num_nodes, "Label propagation")
    
    # Convert to numpy array
    labels_np = np.array(labels)
    
    # Balance the clusters if requested
    if balance_clusters:
        # Create adjacency matrix for balancing
        adj_np = np.zeros((graph.num_nodes, graph.num_nodes))
        edge_index = graph.edge_index.cpu().numpy()
        
        if hasattr(graph, 'edge_weight') and graph.edge_weight is not None:
            edge_weight = graph.edge_weight.cpu().numpy()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                adj_np[src, dst] = edge_weight[i]
                adj_np[dst, src] = edge_weight[i]  # Ensure symmetry
        else:
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                adj_np[src, dst] = 1
                adj_np[dst, src] = 1
        
        # Balance the clusters
        labels_np = _balance_clusters(labels_np, n_clusters, adj_np)
    
    return labels_np.tolist()

def louvain_clustering(graph, resolution=1.0, n_clusters=2, balance_clusters=False):
    # Convert PyG graph to NetworkX graph
    G = to_networkx(graph)
    
    # Apply Louvain algorithm using NetworkX
    communities = list(greedy_modularity_communities(G, weight='weight', resolution=resolution))
    
    # Process communities to ensure exactly n_clusters
    labels = _process_communities(communities, n_clusters, graph.num_nodes, "Louvain")
    
    # Convert to numpy array
    labels_np = np.array(labels)
    
    # Balance the clusters if requested
    if balance_clusters:
        # Create adjacency matrix for balancing
        adj_np = np.zeros((graph.num_nodes, graph.num_nodes))
        edge_index = graph.edge_index.cpu().numpy()
        
        if hasattr(graph, 'edge_weight') and graph.edge_weight is not None:
            edge_weight = graph.edge_weight.cpu().numpy()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                adj_np[src, dst] = edge_weight[i]
                adj_np[dst, src] = edge_weight[i]  # Ensure symmetry
        else:
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                adj_np[src, dst] = 1
                adj_np[dst, src] = 1
        
        # Balance the clusters
        labels_np = _balance_clusters(labels_np, n_clusters, adj_np)
    
    return labels_np.tolist()

def to_networkx(graph):
    """
    Convert a PyTorch Geometric graph to a NetworkX graph.
    
    Args:
        graph: PyTorch Geometric graph
        
    Returns:
        NetworkX graph with edge weights if available
    """
    # Create empty NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(range(graph.num_nodes))
    
    # Get edge indices
    edge_index = graph.edge_index.cpu().numpy()
    
    # Check if edge weights are available
    if hasattr(graph, 'edge_weight') and graph.edge_weight is not None:
        edge_weight = graph.edge_weight.cpu().numpy()
        # Add weighted edges
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, dst, weight=edge_weight[i])
    else:
        print("no edge weights")
        exit()
    
    return G