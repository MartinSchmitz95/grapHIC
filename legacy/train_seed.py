import torch
import torch.nn as nn
import wandb
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
#import dgl
import os
from datetime import datetime
import numpy as np
import train_utils
import argparse
import yaml
#from SymGatedGCN_DGL import SymGatedGCNYakModel
#from YakModel import YakGATModel
from seed_models import SeedModel_GCN
# Disable
#from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T

#from SGformer_HG import SGFormer
from SGformer_Seed import SGFormerSeed
from SGformer_Seed2 import SGFormerSeed2
from SeedGNNT import SeedGNNT
from SeedGAT import SeedGAT
from SeedGCN import SeedGCN
from GPS import GPS
def initialize_seed_node_feature(g, device):
    # Initialize the node feature tensor with zeros
    node_seed = torch.zeros(g.num_nodes, 1).to(device)

    # Select a random node and set its feature to 1
    random_node = random.randint(0, g.num_nodes - 1)
    node_seed[random_node] = 1
    node_seed[random_node ^ 1] = 1

    return node_seed, random_node

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
    # Get device from graph if not specified
    if device is None and hasattr(g, 'x'):
        device = g.x.device
    
    # Get edge weights for specified edge type
    edge_mask = g.edge_type == edge_type
    edge_weights = g.edge_weight[edge_mask]
    
    # Normalize edge weights to [0,1] range
    if len(edge_weights) > 0:  # Only normalize if we have edges of the specified type
        min_weight = edge_weights.min()
        max_weight = edge_weights.max()
        if max_weight > min_weight:  # Avoid division by zero
            normalized_weights = (edge_weights - min_weight) / (max_weight - min_weight)
            g.edge_weight[edge_mask] = normalized_weights
            print(f"Normalized edge weights range: [{normalized_weights.min():.4f}, {normalized_weights.max():.4f}]")
        else:
            print("All edge weights are equal, no normalization needed")
    else:
        print(f"No edges of type {edge_type} found")
    
    return g

def compute_distance_encoding(g, seed_node, device, max_dist=10, edge_type=None):
    """
    Compute positional encoding based on shortest path distance from seed node.
    
    Args:
        g: Graph object
        seed_node: Index of the seed node
        device: Device to run computation on
        max_dist: Maximum distance to consider
        edge_type: If not None, only consider edges of this type for distance calculation
        
    Returns:
        Distance encoding vector for all nodes
    """
    num_nodes = g.num_nodes
    
    # Initialize distance encoding
    distances = torch.full((num_nodes,), float('inf'), device=device)
    distances[seed_node] = 0
    distances[seed_node ^ 1] = 0  # Also set distance 0 for the complementary seed node
    
    queue = [seed_node, seed_node ^ 1]
    visited = set(queue)
    
    current_dist = 1
    while queue and current_dist <= max_dist:  # Limit to max_dist hops for efficiency
        next_queue = []
        for node in queue:
            # Find neighbors, filtering by edge type if specified
            if edge_type is not None and hasattr(g, 'edge_type'):
                # Get all edges from this node
                edge_indices = (g.edge_index[0] == node).nonzero().squeeze(1)
                # Filter by edge type
                edge_indices = edge_indices[g.edge_type[edge_indices] == edge_type]
                neighbors = g.edge_index[1, edge_indices]
            else:
                # Use all edges
                neighbors = g.edge_index[1, g.edge_index[0] == node]
            
            for neighbor in neighbors:
                neighbor = neighbor.item()
                if neighbor not in visited:
                    distances[neighbor] = current_dist
                    next_queue.append(neighbor)
                    visited.add(neighbor)
        queue = next_queue
        current_dist += 1
    
    # Normalize distances to [0, 1] range
    normalized_distances = torch.where(
        distances != float('inf'),
        distances / max_dist,
        torch.ones_like(distances)  # Set unreachable nodes to 1.0
    )
    
    # Print distribution of distances
    edge_type_str = f" (edge type {edge_type})" if edge_type is not None else ""
    print(f"Distance distribution from seed node{edge_type_str}:")
    for d in range(int(max_dist) + 1):
        count = ((distances == d).sum().item())
        print(f"  Distance {d}: {count} nodes ({100 * count / num_nodes:.2f}%)")
    print(f"  Unreachable: {(distances == float('inf')).sum().item()} nodes")
    
    return normalized_distances.unsqueeze(1)  # Return as column vector

def compute_fiedler_encoding(g, device, edge_type=None):
    """
    Compute positional encoding based on the Fiedler vector (second eigenvector of the Laplacian).
    
    Args:
        g: Graph object
        device: Device to run computation on
        edge_type: If not None, only consider edges of this type for Laplacian computation
        
    Returns:
        Fiedler vector encoding for all nodes
    """
    num_nodes = g.num_nodes
    
    # Initialize with zeros in case computation fails
    fiedler_encoding = torch.zeros(num_nodes, 1, device=device)
    
    # Only compute for graphs with enough nodes
    if num_nodes > 10 and hasattr(g, 'edge_index'):
        try:
            # Create adjacency matrix, filtering by edge type if specified
            adj = torch.zeros(num_nodes, num_nodes, device=device)
            
            if edge_type is not None and hasattr(g, 'edge_type'):
                # Filter edges by type
                edge_mask = g.edge_type == edge_type
                filtered_edge_index = g.edge_index[:, edge_mask]
                
                for i in range(filtered_edge_index.size(1)):
                    src, dst = filtered_edge_index[0, i], filtered_edge_index[1, i]
                    adj[src, dst] = 1
                
                edge_type_str = f" (edge type {edge_type})"
            else:
                # Use all edges
                for i in range(g.edge_index.size(1)):
                    src, dst = g.edge_index[0, i], g.edge_index[1, i]
                    adj[src, dst] = 1
                
                edge_type_str = ""
            
            # Compute Laplacian
            degree = adj.sum(dim=1)
            degree_matrix = torch.diag(degree)
            laplacian = degree_matrix - adj
            
            # Check if we have a connected component
            if torch.all(degree > 0):
                print(f"Computing Fiedler vector{edge_type_str} for connected graph")
            else:
                zero_degree = (degree == 0).sum().item()
                print(f"Warning: {zero_degree} nodes have zero degree{edge_type_str}")
            
            # Compute second eigenvector (Fiedler vector)
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
                # Use second eigenvector (first non-zero)
                fiedler_idx = 1
                while fiedler_idx < len(eigenvalues) and eigenvalues[fiedler_idx] < 1e-5:
                    fiedler_idx += 1
                if fiedler_idx < len(eigenvalues):
                    fiedler = eigenvectors[:, fiedler_idx]
                    # Normalize to [0, 1]
                    fiedler = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)
                    fiedler_encoding = fiedler.unsqueeze(1)
                    print(f"Computed Fiedler vector with eigenvalue: {eigenvalues[fiedler_idx]:.6f}")
                else:
                    print("Could not find suitable Fiedler vector, using fallback")
                    # Fallback: use normalized degree
                    fiedler_encoding = (degree / (degree.max() + 1e-8)).unsqueeze(1)
            except Exception as e:
                print(f"Spectral encoding computation failed: {e}, using fallback")
                # Fallback: use normalized degree
                fiedler_encoding = (degree / (degree.max() + 1e-8)).unsqueeze(1)
        except Exception as e:
            print(f"Error computing spectral encoding: {e}, using zeros")
    
    return fiedler_encoding

def add_seed_positional_encoding(g, seed_node, device, max_dist=10, edge_type=None):
    """
    Add positional encoding features relative to the seed node.
    
    Args:
        g: Graph object
        seed_node: Index of the seed node
        device: Device to run computation on
        max_dist: Maximum distance to consider
        edge_type: If not None, only consider edges of this type for positional encoding
        
    Returns:
        Graph with added positional encoding features
    """
    # Compute distance-based encoding
    distance_encoding = compute_distance_encoding(g, seed_node, device, max_dist, edge_type)
    
    # Compute Fiedler vector encoding
    fiedler_encoding = compute_fiedler_encoding(g, device, edge_type)
    
    # Concatenate encodings to node features
    positional_encoding = torch.cat([distance_encoding, fiedler_encoding], dim=1)
    g.x = torch.cat([g.x, positional_encoding], dim=1)
    
    edge_type_str = f" (edge type {edge_type})" if edge_type is not None else ""
    print(f"Added positional encoding{edge_type_str}. New feature size: {g.x.shape}")
    
    return g

def select_seed_ftrs(g, device):
    # Get seed node and add seed feature
    node_seed, random_node = initialize_seed_node_feature(g, device)
    
    # Add positional encoding relative to seed node
    #g = add_seed_positional_encoding(g, random_node, device, edge_type=1)
    #transform = T.AddRandomWalkPE(walk_length=8, attr_name='pe')
    node_attrs = g.x.to(device)
    transform = T.AddLaplacianEigenvectorPE(k=8, attr_name='x')
    g = transform(g)

    # Now concatenate the tensors
    g.x = torch.cat([g.x.to(device), node_attrs.to(device), node_seed.to(device)], dim=1)

    haplotype_seed = g.y[random_node].item()
    print("Haplotype seed:", haplotype_seed)
    # Initialize gt_hap tensor
    gt_hap = torch.zeros_like(g.y).to(device)
    
    # Get the chromosome of the seed node
    seed_chr = g.chr[random_node].item()
    
    if haplotype_seed == 1:  # MATERNAL seed
        # Set 1 for all nodes with gt_hap=1, 0 for others
        gt_hap = (g.y == 1).float()
    elif haplotype_seed == -1:  # PATERNAL seed
        # Set 1 for all nodes with gt_hap=-1, 0 for others
        gt_hap = (g.y == -1).float()
    elif haplotype_seed == 0:  # Homozygous seed
        # Set 1 for all nodes
        gt_hap = torch.ones_like(g.y).float()
    else:
        raise ValueError(f'Unknown haplotype seed: {haplotype_seed}')
    
    # Set gt_hap to 0 for all nodes with different chromosome than the seed node
    gt_hap[g.chr != seed_chr] = 0
    
    gt_hap = gt_hap.to(device)

    #pagerank = compute_pagerank_n_steps(g, random_node, 16)
    """pagerank = gnn_pagerank(g, random_node, 16, edge_type=1)

    g.x = torch.cat([g.x, pagerank.unsqueeze(1)], dim=1)
    print(f"Added PageRank scores as node feature. New feature size: {g.x.shape}")"""

    
    seed_chr = g.chr[random_node].item()
    g.seed_node_id = random_node
    #exit()
    # Print distribution of gt_hap values
    print("\ngt_hap distribution:")
    unique_gt, counts_gt = torch.unique(gt_hap, return_counts=True)
    for val, count in zip(unique_gt.cpu().numpy(), counts_gt.cpu().numpy()):
        percentage = 100 * count / len(gt_hap)
        print(f"Value {val:.1f}: {count:6d} nodes ({percentage:6.2f}%)")
    return g, gt_hap, seed_chr

def train(model, data_path, train_selection, valid_selection, device, config):
    best_valid_loss = 10000
    overfit = not bool(valid_selection)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)  # Add weight decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['decay'], patience=config['patience'], verbose=True)

    time_start = datetime.now()
    with wandb.init(project=wandb_project, config=config, mode=config['wandb_mode'], name=run_name):

        for epoch in range(config['num_epochs']):
            print(f"Epoch {epoch} of {config['num_epochs']}")
            # Training phase
            print('===> TRAINING')
            model.train()
            random.shuffle(train_selection)
            train_metrics = train_epoch(model, train_selection, data_path, device, optimizer, 
                                      config, epoch)

            # Validation phase (if applicable)
            valid_metrics = {}
            if not overfit:
                print('===> VALIDATION')
                model.eval()
                valid_metrics = validate_epoch(model, valid_selection, data_path, device,
                                            config, epoch)
                scheduler.step(valid_metrics['valid_loss'])  # Uncomment to use scheduler
                
                if valid_metrics['valid_loss'] < best_valid_loss:
                    best_valid_loss = valid_metrics['valid_loss']
                    model_path = os.path.join(save_model_path, f'{args.run_name}_best.pt')
                    torch.save(model.state_dict(), model_path)
                    print("Saved model")

            # Log metrics
            log_dict = {**train_metrics}
            if valid_metrics:
                log_dict.update(valid_metrics)
            wandb.log(log_dict)

            if (epoch+1) % 10 == 0:
                model_path = os.path.join(save_model_path, f'{args.run_name}_{epoch+1}.pt')
                torch.save(model.state_dict(), model_path)
                print("Saved model")

def train_epoch(model, train_selection, data_path, device, optimizer, 
                config, epoch):
    train_loss = []
    pred_mean, pred_std = [], []
    compute_metrics = (epoch % config['compute_metrics_every'] == 0)
    train_phasing_acc, train_scaffolding_acc = [], []
    pos_weights = []

    for idx, graph_name in enumerate(train_selection):
        print(f"Training graph {graph_name}, id: {idx} of {len(train_selection)}")
        g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
        
        # Normalize edge weights once when loading the graph
        g = normalize_edge_weights(g, edge_type=1, device=device)

        optimizer.zero_grad()

        g, y, active_chr = select_seed_ftrs(g, device)
        
        # Calculate positive weight based on the percentage of positive samples in this graph
        pos_ratio = y.mean().item()  # Percentage of positive samples (1s)
        if pos_ratio > 0 and pos_ratio < 1:
            # Formula: (1-pos_ratio)/pos_ratio to balance positive and negative samples
            pos_weight_value = (1 - pos_ratio) / pos_ratio
        else:
            # Default fallback if all samples are positive or all negative
            pos_weight_value = 1.0
        
        pos_weight = torch.tensor([pos_weight_value]).to(device)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        pos_weights.append(pos_weight_value)
        
        predictions = model(g).squeeze()
        
        # Remove this clamp - it's preventing proper gradient flow
        # predictions = torch.clamp(predictions, 0.0, 1.0)
        pred_mean.append(predictions.mean().item())
        pred_std.append(predictions.std().item())

        loss = bce_loss(predictions, y)

        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss.append(loss.item())

        # Print loss for debugging
        print(f"  Loss: {loss.item():.4f}, Pred mean: {predictions.mean().item():.4f}, Pred std: {predictions.std().item():.4f}")

        if compute_metrics:
            with torch.no_grad():  # No need to track gradients for metrics
                probabilities = torch.sigmoid(predictions)
                phasing_acc = compute_phasing_accuracy(y, probabilities, g.chr, active_chr, debug=True)
                scaffolding_acc = compute_scaffolding_accuracy(y, probabilities, g.chr, active_chr, debug=True)
                train_phasing_acc.append(phasing_acc)
                train_scaffolding_acc.append(scaffolding_acc)
    metrics = {
        'train_loss': np.mean(train_loss),
    }
    
    if compute_metrics:
        metrics.update({
            'train_phasing_acc': np.mean(train_phasing_acc),
            'train_scaffolding_acc': np.mean(train_scaffolding_acc)
        })
    
    return metrics

def validate_epoch(model, valid_selection, data_path, device, 
                  config, epoch):
    valid_loss = []
    valid_pred_mean, valid_pred_std = [], []
    compute_metrics = (epoch % config['compute_metrics_every'] == 0)
    valid_phasing_acc, valid_scaffolding_acc = [], []
    pos_weights = []

    with torch.no_grad():
        for idx, graph_name in enumerate(valid_selection):
            print(f"Validating graph {graph_name}, id: {idx} of {len(valid_selection)}")
            g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
            
            # Normalize edge weights once when loading the graph
            g = normalize_edge_weights(g, edge_type=1, device=device)

            g, y, active_chr = select_seed_ftrs(g, device)
            
            # Calculate positive weight based on the percentage of positive samples in this graph
            pos_ratio = y.mean().item()  # Percentage of positive samples (1s)
            if pos_ratio > 0 and pos_ratio < 1:
                # Formula: (1-pos_ratio)/pos_ratio to balance positive and negative samples
                pos_weight_value = (1 - pos_ratio) / pos_ratio
            else:
                # Default fallback if all samples are positive or all negative
                pos_weight_value = 1.0
                
            pos_weight = torch.tensor([pos_weight_value]).to(device)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            pos_weights.append(pos_weight_value)
            
            predictions = model(g).squeeze()
            valid_pred_mean.append(predictions.mean().item())
            valid_pred_std.append(predictions.std().item())

            loss = bce_loss(predictions, y)
            valid_loss.append(loss.item())

            if compute_metrics:
                with torch.no_grad():  # No need to track gradients for metrics
                    probabilities = torch.sigmoid(predictions)
                    phasing_acc = compute_phasing_accuracy(y, probabilities, g.chr, active_chr, debug=True)
                    scaffolding_acc = compute_scaffolding_accuracy(y, probabilities, g.chr, active_chr, debug=True)
                    valid_phasing_acc.append(phasing_acc)
                    valid_scaffolding_acc.append(scaffolding_acc)

    metrics = {
        'valid_loss': np.mean(valid_loss),
    }
    
    if compute_metrics:
        metrics.update({
            'valid_phasing_acc': np.mean(valid_phasing_acc),
            'valid_scaffolding_acc': np.mean(valid_scaffolding_acc)
        })
    
    return metrics

def compute_pagerank_n_steps(g, seed_node, n_steps, damping_factor=0.85, device=None):
    """
    Compute PageRank after n steps, starting with the seed node having rank 1.
    Only walks over edges of type 0 (overlap edges).
    
    Args:
        g: Graph object
        seed_node: Index of the seed node
        n_steps: Number of steps to run PageRank
        damping_factor: Damping factor for PageRank (default: 0.85)
        device: Device to run computation on
        
    Returns:
        PageRank scores for all nodes after n steps
    """
    # Get device from graph if not specified
    if device is None:
        device = g.x.device
    
    num_nodes = g.num_nodes
    
    # Initialize PageRank scores: seed node has score 1, others 0
    scores = torch.zeros(num_nodes, device=device)
    scores[seed_node] = 0.5
    scores[seed_node ^ 1] = 0.5  # Split initial probability between the two seed nodes
    
    # Extract only overlap edges (type 0)
    mask = g.edge_type == 0
    overlap_edge_index = g.edge_index[:, mask]
    source_nodes, target_nodes = overlap_edge_index
    
    # Pre-compute out-degrees for all nodes (much faster)
    out_degrees = torch.zeros(num_nodes, device=device)
    index, counts = torch.unique(source_nodes, return_counts=True)
    out_degrees[index] = counts.float()
    
    # Create edge weights for efficient matrix-vector multiplication
    edge_values = torch.ones(source_nodes.size(0), device=device)
    for i in range(source_nodes.size(0)):
        if out_degrees[source_nodes[i]] > 0:
            edge_values[i] = 1.0 / out_degrees[source_nodes[i]]
    
    # Create teleportation vector (only to seed nodes)
    teleport = torch.zeros(num_nodes, device=device)
    teleport[seed_node] = 0.5
    teleport[seed_node ^ 1] = 0.5
    
    # For each step
    for step in range(n_steps):
        # Matrix-vector multiplication: new_scores = A * scores
        new_scores = torch.zeros_like(scores)
        for i in range(source_nodes.size(0)):
            new_scores[target_nodes[i]] += edge_values[i] * scores[source_nodes[i]]
        
        # Apply damping factor and add teleportation
        new_scores = damping_factor * new_scores
        
        # Add teleportation component (only to seed nodes)
        teleport_contribution = (1 - damping_factor) * teleport
        new_scores += teleport_contribution
        
        # Update scores
        scores = new_scores
        
        # Normalize to prevent numerical issues
        total = scores.sum()
        if total > 0:
            scores = scores / total
        
        # Print progress less frequently
        print(f"Step {step}: Sum of scores = {scores.sum().item():.6f}, Max score = {scores.max().item():.6f}")
    
    # Print distribution of PageRank scores
    print(f"PageRank score distribution after {n_steps} steps:")
    ranges = [(0, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1.0)]
    for low, high in ranges:
        count = ((scores >= low) & (scores < high)).sum().item()
        percentage = 100 * count / num_nodes
        print(f"  {low} to {high}: {count} nodes ({percentage:.2f}%)")
    
    # Print top 5 nodes by score
    top_values, top_indices = torch.topk(scores, min(5, len(scores)))
    print("Top 5 nodes by PageRank score:")
    for i, (idx, val) in enumerate(zip(top_indices.cpu().numpy(), top_values.cpu().numpy())):
        print(f"  #{i+1}: Node {idx} with score {val:.6f}")
    
    # Add PageRank scores as a node feature
    return scores

def compute_phasing_accuracy(true_y, pred_y, chr_labels, active_chr, debug=False):
    """
    Compute phasing accuracy by comparing true haplotype labels with predicted probabilities.
    Only considers nodes from the specified active chromosome.
    
    Args:
        true_y: Ground truth binary labels (1 for maternal, 0 for paternal)
        pred_y: Predicted probabilities
        chr_labels: Chromosome labels for each node
        active_chr: The chromosome to consider for accuracy calculation
        debug: Whether to print debug information
        
    Returns:
        Phasing accuracy as a float between 0 and 1
    """
    
    # Create mask for nodes in the active chromosome
    same_chr_mask = (chr_labels == active_chr)
    
    # Skip if no nodes from the active chromosome
    if not torch.any(same_chr_mask):
        if debug:
            print(f"Warning: No nodes found in chromosome {active_chr}")
        return 0.0
    
    # Filter to only include nodes from the active chromosome
    filtered_true_y = true_y[same_chr_mask]
    filtered_pred_y = pred_y[same_chr_mask]
    
    # Convert predictions to binary (0 or 1) using 0.5 as threshold
    binary_preds = (filtered_pred_y >= 0.5).float()
    
    # Calculate accuracy
    correct = (binary_preds == filtered_true_y).sum().item()
    total = filtered_true_y.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    
    if debug:
        print(f"Phasing accuracy: {accuracy:.4f} ({correct}/{total} correct)")
        print(f"Active chromosome: {active_chr}")
        print(f"Nodes in active chromosome: {same_chr_mask.sum().item()}")
        
        # Print distribution of predictions
        pred_dist = {
            "0.0-0.1": ((filtered_pred_y >= 0.0) & (filtered_pred_y < 0.1)).sum().item(),
            "0.1-0.3": ((filtered_pred_y >= 0.1) & (filtered_pred_y < 0.3)).sum().item(),
            "0.3-0.5": ((filtered_pred_y >= 0.3) & (filtered_pred_y < 0.5)).sum().item(),
            "0.5-0.7": ((filtered_pred_y >= 0.5) & (filtered_pred_y < 0.7)).sum().item(),
            "0.7-0.9": ((filtered_pred_y >= 0.7) & (filtered_pred_y < 0.9)).sum().item(),
            "0.9-1.0": ((filtered_pred_y >= 0.9) & (filtered_pred_y <= 1.0)).sum().item()
        }
        print("Prediction distribution:", pred_dist)
    
    return accuracy

def compute_scaffolding_accuracy(true_y, pred_y, chr_labels, active_chr, debug=False):
    """
    Compute scaffolding accuracy by comparing true haplotype labels with predicted probabilities.
    Considers all nodes except homozygous nodes (y=0) in the active chromosome.
    
    Args:
        true_y: Ground truth binary labels (1 for maternal, -1 for paternal, 0 for homozygous)
        pred_y: Predicted probabilities
        chr_labels: Chromosome labels for each node
        active_chr: The chromosome to exclude other haplotype nodes from
        debug: Whether to print debug information
        
    Returns:
        Scaffolding accuracy as a float between 0 and 1
    """
    # Create mask for nodes to include in scaffolding accuracy:
    # 1. All nodes from chromosomes other than active_chr
    # 2. Non-homozygous nodes (y != 0) from active_chr
    other_chr_mask = (chr_labels != active_chr)
    active_chr_active_hap_mask = (chr_labels == active_chr) & (true_y == 1)
    
    # Combine masks to get all nodes to consider
    scaffolding_mask = other_chr_mask | active_chr_active_hap_mask
    
    # Skip if no nodes to evaluate
    if not torch.any(scaffolding_mask):
        if debug:
            print("Warning: No nodes found for scaffolding accuracy calculation")
        return 0.0
    
    # Filter to only include relevant nodes
    filtered_true_y = true_y[scaffolding_mask]
    filtered_pred_y = pred_y[scaffolding_mask]
    
    # Convert predictions to binary (0 or 1) using 0.5 as threshold
    binary_preds = (filtered_pred_y >= 0.5).float()
    
    # For scaffolding, we need to convert true_y to binary format:
    # 1 for maternal (true_y == 1), 0 for paternal (true_y == -1)
    binary_true_y = (filtered_true_y > 0).float()
    
    # Calculate accuracy
    correct = (binary_preds == binary_true_y).sum().item()
    total = binary_true_y.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    
    if debug:
        print(f"Scaffolding accuracy: {accuracy:.4f} ({correct}/{total} correct)")
        print(f"Active chromosome: {active_chr}")
        print(f"Nodes in other chromosomes: {other_chr_mask.sum().item()}")
        print(f"Non-homozygous nodes in active chromosome: {active_chr_active_hap_mask.sum().item()}")
        print(f"Total nodes considered: {scaffolding_mask.sum().item()}")
        
        # Print distribution of predictions
        pred_dist = {
            "0.0-0.1": ((filtered_pred_y >= 0.0) & (filtered_pred_y < 0.1)).sum().item(),
            "0.1-0.3": ((filtered_pred_y >= 0.1) & (filtered_pred_y < 0.3)).sum().item(),
            "0.3-0.5": ((filtered_pred_y >= 0.3) & (filtered_pred_y < 0.5)).sum().item(),
            "0.5-0.7": ((filtered_pred_y >= 0.5) & (filtered_pred_y < 0.7)).sum().item(),
            "0.7-0.9": ((filtered_pred_y >= 0.7) & (filtered_pred_y < 0.9)).sum().item(),
            "0.9-1.0": ((filtered_pred_y >= 0.9) & (filtered_pred_y <= 1.0)).sum().item()
        }
        print("Prediction distribution:", pred_dist)
    
    return accuracy

def gnn_pagerank(g, seed_node, n_steps, edge_type=0, damping_factor=0.85, device=None):
    """
    Compute PageRank using graph neural networks, starting with the seed node having rank 1.
    Uses PyTorch Geometric operations for efficient computation.
    
    Args:
        g: Graph object (PyTorch Geometric)
        seed_node: Index of the seed node
        n_steps: Number of steps to run PageRank
        edge_type: Type of edges to use for PageRank computation (default: 0 for overlap edges)
        damping_factor: Damping factor for PageRank (default: 0.85)
        device: Device to run computation on
        
    Returns:
        PageRank scores for all nodes after n steps
    """
    # Get device from graph if not specified
    if device is None:
        device = g.x.device
    
    num_nodes = g.num_nodes
    
    # Print seed node information
    print(f"Seed node: {seed_node}, Complement: {seed_node ^ 1}")
    print(f"Using edges of type {edge_type} for PageRank computation")
    
    # Initialize PageRank scores: seed nodes have score 0.5 each, others 0
    scores = torch.zeros(num_nodes, device=device)
    scores[seed_node] = 0.5
    scores[seed_node ^ 1] = 0.5  # Split initial probability between the two seed nodes
    
    # Extract only edges of the specified type
    mask = g.edge_type == edge_type
    edge_index = g.edge_index[:, mask]
    
    # Check if we have any edges of the specified type
    if edge_index.size(1) == 0:
        print(f"Warning: No edges of type {edge_type} found in the graph!")
        print(f"Available edge types: {torch.unique(g.edge_type).tolist()}")
        return scores  # Return initial scores if no edges of the specified type
    
    edge_weight = g.edge_weight[mask]

    # Create teleportation vector (only to seed nodes)
    teleport = torch.zeros(num_nodes, device=device)
    teleport[seed_node] = 0.5
    teleport[seed_node ^ 1] = 0.5
    
    # Compute out-degrees for normalization
    row, col = edge_index
    
    # Normalize edge weights by out-degree
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, row, edge_weight)  # Sum weights for each source node
    
    # Normalize edge weights by out-degree
    normalized_edge_weight = edge_weight.clone()
    for i in range(row.size(0)):
        if deg[row[i]] > 0:
            normalized_edge_weight[i] = edge_weight[i] / deg[row[i]]
    
    # For each step
    for step in range(n_steps):
        # Create a message for each edge: source_score * edge_weight
        messages = scores[row] * normalized_edge_weight
        
        # Aggregate messages at target nodes
        new_scores = torch.zeros_like(scores)
        new_scores.scatter_add_(0, col, messages)
        
        # Apply damping factor and add teleportation
        new_scores = damping_factor * new_scores + (1 - damping_factor) * teleport
        
        # Normalize to prevent numerical issues
        total = new_scores.sum()
        if total > 0:
            new_scores = new_scores / total
        
        # Update scores
        scores = new_scores
        
        # Print progress and seed node scores
        print(f"Step {step}: Sum of scores = {scores.sum().item():.6f}, Max score = {scores.max().item():.6f}")
        print(f"  Seed node {seed_node} score: {scores[seed_node].item():.6f}")
        print(f"  Complement {seed_node ^ 1} score: {scores[seed_node ^ 1].item():.6f}")

    # Print distribution of PageRank scores
    print(f"PageRank score distribution after {n_steps} steps:")
    ranges = [(0, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1.0)]
    for low, high in ranges:
        count = ((scores >= low) & (scores < high)).sum().item()
        percentage = 100 * count / num_nodes
        print(f"  {low} to {high}: {count} nodes ({percentage:.2f}%)")
    
    # Print top 5 nodes by score
    top_values, top_indices = torch.topk(scores, min(5, len(scores)))
    print("Top 5 nodes by PageRank score:")
    for i, (idx, val) in enumerate(zip(top_indices.cpu().numpy(), top_values.cpu().numpy())):
        print(f"  #{i+1}: Node {idx} with score {val:.6f}")
        
    # Check if seed nodes are in top 10
    top10_indices = torch.topk(scores, 10)[1]
    if seed_node in top10_indices or (seed_node ^ 1) in top10_indices:
        print("✓ At least one seed node is in the top 10")
    else:
        print("⚠ Neither seed node is in the top 10!")
        seed_rank = (scores >= scores[seed_node]).sum().item()
        comp_rank = (scores >= scores[seed_node ^ 1]).sum().item()
        print(f"Seed node {seed_node} rank: {seed_rank} out of {num_nodes}")
        print(f"Complement {seed_node ^ 1} rank: {comp_rank} out of {num_nodes}")
    
    return scores

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="experiment eval script")
    parser.add_argument("--load_checkpoint", type=str, default='', help="dataset path")
    parser.add_argument("--save_model_path", type=str, default='/mnt/sod2-project/csb4/wgs/martin/trained_models_march', help="dataset path")
    parser.add_argument("--data_path", type=str, default='data', help="dataset path")
    parser.add_argument("--run_name", type=str, default='test', help="dataset path")
    parser.add_argument("--device", type=str, default='cpu', help="dataset path")
    parser.add_argument("--data_config", type=str, default='dataset_config.yml', help="dataset path")
    parser.add_argument("--hyper_config", type=str, default='train_config.yml', help="dataset path")
    parser.add_argument('--diploid', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument("--wandb", type=str, default='debug', help="dataset path")
    parser.add_argument('--symmetry', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--aux', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--bce', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--hap_switch', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--quick', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument("--seed", type=int, default=0, help="dataset path")

    args = parser.parse_args()

    wandb_project = args.wandb
    run_name = args.run_name
    load_checkpoint = args.load_checkpoint
    save_model_path = args.save_model_path
    data_path = args.data_path
    device = args.device
    data_config = args.data_config
    diploid = args.diploid
    symmetry = args.symmetry
    aux = args.aux
    hyper_config = args.hyper_config
    quick = args.quick
    hap_switch = args.hap_switch
    full_dataset, valid_selection, train_selection = train_utils.create_dataset_dicts(data_config=data_config)
    train_data = train_utils.get_numbered_graphs(train_selection)
    valid_data = train_utils.get_numbered_graphs(valid_selection, starting_counts=train_selection)

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    with open(hyper_config) as file:
        config = yaml.safe_load(file)['training']

    train_utils.set_seed(args.seed)
   
    model = SGFormerSeed2(in_channels=config['node_features'], hidden_channels=config['hidden_features'], out_channels=1, trans_num_layers=config['num_trans_layers'], gnn_num_layers_0=config['num_gnn_layers_overlap'], gnn_num_layers_1=config['num_gnn_layers_hic'], gnn_dropout=config['gnn_dropout'], seed_attention=True).to(device)
    """model = SGFormerSeed(
        in_channels=config['node_features'],
        hidden_channels=config['hidden_features'],
        out_channels=1,
        seed_trans_num_layers=config['num_trans_layers'],
        seed_trans_num_heads= 8, #config['num_heads'],
        seed_trans_dropout= 0, #config['dropout'],
        gnn_num_layers_0=config['num_gnn_layers_overlap'],
        gnn_num_layers_1=config['num_gnn_layers_hic'],
        gnn_dropout= 0, #config['gnn_dropout'],
    ).to(device)"""
    #model = SeedModel_GCN(node_features=config['node_features'], edge_features=1, num_layers=config['num_gnn_layers_overlap'], dropout=config['gnn_dropout'], hidden_features=config['hidden_features']).to(device)
    #model = MultiSGFormer(num_sgformers=3, in_channels=2, hidden_channels=128, out_channels=1, trans_num_layers=2, gnn_num_layers=4, gnn_dropout=0.0).to(device)
    
    # Create model
    """model = SeedGNNT(
        in_channels=config['node_features'],
        hidden_channels=config['hidden_features'],
        out_channels=1,
        gnn_num_layers_0=config['num_gnn_layers_overlap'],
        gnn_num_layers_1=config['num_gnn_layers_hic'],
        gnn_dropout= 0, #config['gnn_dropout'],
        seed_attn_num_layers=config['num_trans_layers'],
        seed_attn_num_heads= 8,#config['seed_attn_num_heads'],
        seed_attn_dropout= 0,#config['seed_attn_dropout'],
    ).to(device)"""

    """model = SeedGAT(
        in_channels=config['node_features'],
        hidden_channels=config['hidden_features'],
        out_channels=1,
        gat_num_layers_0=config['num_gnn_layers_overlap'],
        gat_num_layers_1=config['num_gnn_layers_hic'],
        gat_dropout= 0, #config['gnn_dropout'],
        gat_heads= 8,#config['seed_attn_num_heads'],
        gat_use_bn=True,
        gat_use_residual=True,
        predictor_hidden_channels=64,
        predictor_dropout=0.0,
    ).to(device)"""

    model = SeedGCN(
        in_channels=config['node_features'],
        hidden_channels=config['hidden_features'],
        out_channels=1,
        gcn_num_layers_0=config['num_gnn_layers_overlap'],
        gcn_num_layers_1=config['num_gnn_layers_hic'],
        gcn_dropout= 0, #config['gnn_dropout'],
        #use_cross_attention=True,
    ).to(device)
    
    """model = GPS(
        in_channels=13,
        hidden_channels=128,
        out_channels=1,
        num_layers=6,
        heads=8,
        dropout=0.0,
        attn_type='performer',
    ).to(device)"""
    

    #latest change: half hidden channels, reduce gnn_layers, remove dropout
    to_undirected = False
    model.to(device)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint, map_location=device))
        print(f"Loaded model from {load_checkpoint}")

    train(model, data_path, train_data, valid_data, device, config)

    #python train_yak2.py --save_model_path /mnt/sod2-project/csb4/wgs/martin/rl_dgl_datasets/trained_models/ --data_path /mnt/sod2-project/csb4/wgs/martin/diploid_datasets/diploid_dataset_hg002_cent/pyg_graphs/ --data_config dataset_diploid_cent.yml --hyper_config config_yak.yml --wandb pred_yak --run_name yak2_SGformer_base --device cuda:4 --diploid
