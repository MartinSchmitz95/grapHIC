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
# Disable
#from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from SGformer_C2 import SGFormerC2
from contrastive_loss import MultiLabelConLoss, ConLoss, SupervisedContrastiveLoss, MultiDimHammingLoss, adaptive_clustering_loss
import copy
from ClusterGCN import ClusterGCN

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

def preprocess_graph(g, device):
    # Get seed node and add seed feature
    
    # Add positional encoding relative to seed node
    #g = add_seed_positional_encoding(g, random_node, device, edge_type=1)
    #transform = T.AddRandomWalkPE(walk_length=8, attr_name='pe')
    """node_attrs = g.x.to(device)
    transform = T.AddLaplacianEigenvectorPE(k=8, attr_name='x')
    g = transform(g)
    # Now concatenate the tensors
    g.x = torch.cat([g.x.to(device), node_attrs.to(device)], dim=1)"""

    # Create a new tensor that combines chromosome and haplotype information
    chr_tensor = g.chr.to(device)
    y_tensor = g.y.to(device)
    
    # First pass: collect all unique chromosomes
    unique_chrs = torch.unique(chr_tensor).cpu().numpy()
    num_chrs = len(unique_chrs)
    print(f"Found {num_chrs} unique chromosomes")
    
    # Create multi-label encoding with 2 entries per chromosome
    # Each node will have 1s in positions corresponding to its chromosome and haplotype
    # Size: [num_nodes, num_chromosomes * 2]
    multi_labels = torch.zeros((len(chr_tensor), num_chrs * 2), dtype=torch.float, device=device)
    
    # Create mapping from chromosome value to index in our label vector
    chr_to_idx = {chr_val.item(): idx for idx, chr_val in enumerate(unique_chrs)}
    
    # Create a 2-dimensional multi-label vector
    multi_labels = torch.zeros((len(y_tensor), 2), dtype=torch.float, device=device)
    
    # First entry is 1 if y is 1 or 0
    multi_labels[:, 0] = (y_tensor == 1) | (y_tensor == 0)
    
    # Second entry is 1 if y is -1 or 0
    multi_labels[:, 1] = (y_tensor == -1) | (y_tensor == 0)
    g.hap_gt = multi_labels
    # For debugging: count haplotypes per chromosome
    chr_to_haps = {}
    for node_idx in range(len(chr_tensor)):
        chr_val = chr_tensor[node_idx].item()
        hap_val = y_tensor[node_idx].item()
        
        if chr_val not in chr_to_haps:
            chr_to_haps[chr_val] = set()
        chr_to_haps[chr_val].add(hap_val)
    
    for chr_val, haps in chr_to_haps.items():
        print(f"Chromosome {chr_val} has {len(haps)} haplotypes: {sorted(haps)}")
    
    print(f"Created multi-label encoding with {multi_labels.shape[1]} dimensions ({num_chrs} chromosomes × 2 haplotypes)")
    return g, multi_labels

def add_laplacian_pe_by_edge_type(g, edge_type, k=8, attr_name=None):
    """
    Add Laplacian eigenvector positional encoding for a specific edge type.
    
    Args:
        g: Graph object
        edge_type: Type of edges to use for Laplacian computation (0 or 1)
        k: Number of eigenvectors to compute (default: 8)
        attr_name: Name of attribute to store PE (default: 'pe_type{edge_type}')
        
    Returns:
        Graph with added positional encoding attribute
    """
    if attr_name is None:
        attr_name = f'pe_{edge_type}'
    
    # Get edge indices for the specified edge type
    edge_mask = g.edge_type == edge_type
    edge_index_filtered = g.edge_index[:, edge_mask]
    
    # Also filter the edge weights if they exist
    edge_weight_filtered = None
    if hasattr(g, 'edge_weight'):
        edge_weight_filtered = g.edge_weight[edge_mask]
    
    # Store original edge structure
    original_edge_index = g.edge_index.clone()
    original_edge_type = g.edge_type.clone()
    original_edge_weight = g.edge_weight.clone() if hasattr(g, 'edge_weight') else None
    
    # Temporarily replace edge_index with only edges of the specified type
    g.edge_index = edge_index_filtered
    if hasattr(g, 'edge_weight'):
        g.edge_weight = edge_weight_filtered
    
    # Create a temporary graph object with just the filtered edges
    temp_g = copy.copy(g)
    
    # Apply Laplacian eigenvector PE transform
    transform = T.AddLaplacianEigenvectorPE(k=k, attr_name=attr_name)
    temp_g = transform(temp_g)
    
    # Copy the computed positional encoding to the original graph
    setattr(g, attr_name, getattr(temp_g, attr_name))
    
    # Restore original edge structure
    g.edge_index = original_edge_index
    g.edge_type = original_edge_type
    if original_edge_weight is not None:
        g.edge_weight = original_edge_weight
    
    print(f"Added positional encoding for edge type {edge_type}: {attr_name} shape: {getattr(g, attr_name).shape}")
    
    return g

def train(model, data_path, train_selection, valid_selection, device, config):
    best_valid_loss = 10000
    overfit = not bool(valid_selection)
    if overfit:
        valid_selection = train_selection

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])  # Add weight decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['decay'], patience=config['patience'], verbose=True)

    # Track best total loss
    best_total_loss = float('inf')
    
    # Initialize separation metrics to avoid NaN
    chr_separation = 0
    hap_separation = 0

    phasing_loss = MultiLabelConLoss(temperature=0.1, device=device)
    scaffolding_loss = ConLoss(temperature=0.1, device=device)
    #scaffolding_loss = SupervisedContrastiveLoss()
    #scaffolding_loss = MultiDimHammingLoss()

    time_start = datetime.now()
    with wandb.init(project=wandb_project, config=config, mode=config['wandb_mode'], name=run_name):
        for epoch in range(config['num_epochs']):
            print(f"Epoch {epoch} of {config['num_epochs']}")
            
            # Compute metrics in epoch 0 or at regular intervals if evaluation is enabled
            if config['evaluate'] and (epoch == 0 or epoch % config['compute_metrics_every'] == 0):
                print('===> EVALUATING EMBEDDINGS')
                eval_metrics = evaluate_embeddings(model, data_path, valid_selection, device)
                # Update separation metrics
                chr_separation = eval_metrics['chromosome_separation']
                hap_separation = eval_metrics['haplotype_separation']
                
                # Optionally visualize embeddings (skip for epoch 0)
                if config['visualize_during_training'] and epoch > 0:
                    for graph_name in valid_selection[:1]:  # Visualize just the first validation graph
                        output_dir = os.path.join('embedding_plots', f'epoch_{epoch}')
                        visualize_embeddings(model, data_path, graph_name, device, output_dir)
        
            # Training phase
            print('===> TRAINING')
            model.train()
            random.shuffle(train_selection)
            train_metrics = train_epoch(model, train_selection, data_path, device, optimizer, 
                                      config, phasing_loss, scaffolding_loss)

            # Validation phase (if applicable)
            valid_metrics = {}
            if not overfit:
                print('===> VALIDATION')
                model.eval()
                valid_metrics = validate_epoch(model, valid_selection, data_path, device,
                                            config, phasing_loss, scaffolding_loss)
                
                scheduler.step(valid_metrics['valid_loss'])  # Use total loss for scheduler
                
                # Save model with lowest validation loss
                if valid_metrics['valid_loss'] < best_valid_loss:
                    best_valid_loss = valid_metrics['valid_loss']
                    model_path = os.path.join(save_model_path, f'{args.run_name}_best.pt')
                    torch.save(model.state_dict(), model_path)
                    print("Saved model")

            
            # Only include separation metrics if evaluation is enabled
            if config['evaluate']:
                train_metrics['chr_separation'] = chr_separation
                train_metrics['hap_separation'] = hap_separation

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
                config, phasing_loss, scaffolding_loss):
    train_loss = []
    train_phasing_loss = []
    train_scaffolding_loss = []
    compute_metrics = False #(epoch % config['compute_metrics_every'] == 0)
    train_phasing_acc, train_scaffolding_acc = [], []

    batch_size = config['batch_size']  # Set batch size for sampling nodes

    for idx, graph_name in enumerate(train_selection):
        print(f"Training graph {graph_name}, id: {idx} of {len(train_selection)}")
        g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
        # Normalize edge weights once when loading the graph
        #g = normalize_edge_weights(g, edge_type=1, device=device)
        #g, labels = preprocess_graph(g, device)
        #transform = T.AddLaplacianEigenvectorPE(k=8, attr_name='pe')
        #g = add_laplacian_pe_by_edge_type(g, edge_type=0, k=8)
        #g = add_laplacian_pe_by_edge_type(g, edge_type=1, k=8)
        #print(f"save graph {graph_name}")
        #torch.save(g, os.path.join(data_path, graph_name + '.pt'))
        # Now concatenate the tensors
        
        #g.x = torch.cat([g.x.to(device), g.pe_0.to(device), g.pe_1.to(device)], dim=1)
        g.x = torch.abs(g.y).float().unsqueeze(1).to(device) #torch.cat([torch.abs(g.y).float(), torch.abs(g.y).float()], dim=1)
        #torch.save(g, os.path.join(data_path, graph_name + '.pt'))
        optimizer.zero_grad()
        chr_labels = g.chr.int().to(device)
        # Get embeddings for all nodes
        _, _, phasing_projections, scaffolding_projections = model(g)
        
        # Compute scaffolding loss if not only_phase
        scaffolding_batch_loss = 0
        if not config['only_phase']:
            # Randomly sample nodes if we have more than batch_size
            num_nodes = scaffolding_projections.size(0)
            if num_nodes > batch_size:
                # Generate random indices for sampling
                indices = torch.randperm(num_nodes, device=device)[:batch_size]
                # Sample embeddings and labels
                sampled_scaffolding_projections = scaffolding_projections[indices]
                sampled_labels = chr_labels[indices]
                
                # Compute loss on the sampled batch
                #scaffolding_batch_loss = scaffolding_loss(sampled_scaffolding_embs, sampled_labels)
                #scaffolding_batch_loss = scaffolding_loss(scaffolding_embs, chr_labels, g.edge_index)
                scaffolding_batch_loss = adaptive_clustering_loss(sampled_scaffolding_projections, sampled_labels, g.edge_index)
            else:
                # If we have fewer nodes than batch_size, use all nodes
                #scaffolding_batch_loss = scaffolding_loss(scaffolding_embs, chr_labels)
                #scaffolding_batch_loss = scaffolding_loss(scaffolding_embs, chr_labels, g.edge_index)
                scaffolding_batch_loss = adaptive_clustering_loss(scaffolding_projections, chr_labels, g.edge_index)

            train_scaffolding_loss.append(scaffolding_batch_loss.item())
                
        # Compute phasing loss if not only_scaffold
        phasing_batch_loss = 0
        if not config['only_scaffold']:
            # Get unique chromosomes
            unique_chrs = torch.unique(chr_labels)
            # Randomly select a chromosome
            random_chr_idx = torch.randint(0, len(unique_chrs), (1,)).item()
            random_chr = unique_chrs[random_chr_idx]
            
            # Get nodes from the selected chromosome
            chr_mask = chr_labels == random_chr
            chr_node_indices = torch.where(chr_mask)[0]
            
            # Sample nodes if we have more than batch_size
            if len(chr_node_indices) > batch_size:
                # Randomly sample batch_size nodes from this chromosome
                sampled_indices = chr_node_indices[torch.randperm(len(chr_node_indices))[:batch_size]]
            else:
                sampled_indices = chr_node_indices
            
            # Get phasing embeddings and haplotype labels for sampled nodes
            sampled_phasing_projections = phasing_projections[sampled_indices]
            sampled_hap_labels = g.hap_gt[sampled_indices].to(device)
            
            # Compute phasing loss
            phasing_batch_loss = phasing_loss(sampled_phasing_projections, sampled_hap_labels)
            
            train_phasing_loss.append(phasing_batch_loss.item())
        
        # Combine losses based on flags
        if config['only_scaffold']:
            loss = scaffolding_batch_loss
        elif config['only_phase']:
            loss = phasing_batch_loss
        else:
            loss = scaffolding_batch_loss + phasing_batch_loss

        loss.backward()
        optimizer.step()
        
        # Track total loss
        train_loss.append(loss.item())

        # Print loss for debugging
        if config['only_scaffold']:
            print(f"Total Loss: {loss.item():.4f} (Scaffolding only)")
        elif config['only_phase']:
            print(f"Total Loss: {loss.item():.4f} (Phasing only)")
        else:
            print(f"Total Loss: {loss.item():.4f} (Scaffolding: {scaffolding_batch_loss.item():.4f}, Phasing: {phasing_batch_loss.item():.4f})")
        
    metrics = {
        'train_loss': np.mean(train_loss),
    }
    
    # Add individual loss components to metrics if they were computed
    if not config['only_phase'] and train_scaffolding_loss:
        metrics['train_scaffolding_loss'] = np.mean(train_scaffolding_loss)
    if not config['only_scaffold'] and train_phasing_loss:
        metrics['train_phasing_loss'] = np.mean(train_phasing_loss)
        
    return metrics

def validate_epoch(model, valid_selection, data_path, device, 
                  config, phasing_loss, scaffolding_loss):
    """Validate the model on the validation set using contrastive learning."""
    valid_loss = []
    valid_phasing_loss = []
    valid_scaffolding_loss = []
    batch_size = config['batch_size']  # Same batch size as in training
    
    with torch.no_grad():
        for idx, graph_name in enumerate(valid_selection):
            print(f"Validating graph {graph_name}, id: {idx+1} of {len(valid_selection)}")
            g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
            
            """# Normalize edge weights
            g = normalize_edge_weights(g, edge_type=1, device=device)
            
            # Preprocess graph
            g, labels = preprocess_graph(g, device)"""
            
            # Get embeddings for all nodes
            _, _, phasing_projections, scaffolding_projections = model(g)
            
            # Compute scaffolding loss if not only_phase
            scaffolding_batch_loss = 0
            if not config['only_phase']:
                # Randomly sample nodes if we have more than batch_size
                num_nodes = scaffolding_projections.size(0)
                if num_nodes > batch_size:
                    # Generate random indices for sampling
                    indices = torch.randperm(num_nodes, device=device)[:batch_size]
                    # Sample embeddings and labels
                    sampled_scaffolding_projections = scaffolding_projections[indices]
                    sampled_labels = g.chr[indices].to(device)
                    
                    # Compute loss on the sampled batch
                    scaffolding_batch_loss = scaffolding_loss(sampled_scaffolding_projections, sampled_labels)
                else:
                    # If we have fewer nodes than batch_size, use all nodes
                    scaffolding_batch_loss = scaffolding_loss(scaffolding_projections, g.chr.to(device))
                
                valid_scaffolding_loss.append(scaffolding_batch_loss.item())
            
            # Compute phasing loss if not only_scaffold
            phasing_batch_loss = 0
            if not config['only_scaffold']:
                # Get unique chromosomes
                unique_chrs = torch.unique(g.chr)
                # Randomly select a chromosome
                random_chr_idx = torch.randint(0, len(unique_chrs), (1,)).item()
                random_chr = unique_chrs[random_chr_idx]
                
                # Get nodes from the selected chromosome
                chr_mask = g.chr == random_chr
                chr_node_indices = torch.where(chr_mask)[0]
                
                # Sample nodes if we have more than batch_size
                if len(chr_node_indices) > batch_size:
                    # Randomly sample batch_size nodes from this chromosome
                    sampled_indices = chr_node_indices[torch.randperm(len(chr_node_indices))[:batch_size]]
                else:
                    sampled_indices = chr_node_indices
                
                # Get phasing embeddings and haplotype labels for sampled nodes
                sampled_phasing_projections = phasing_projections[sampled_indices]
                sampled_hap_labels = g.hap_gt[sampled_indices].to(device)
                
                # Compute phasing loss
                phasing_batch_loss = phasing_loss(sampled_phasing_projections, sampled_hap_labels)
                
                valid_phasing_loss.append(phasing_batch_loss.item())
            
            # Combine losses based on flags
            if config['only_scaffold']:
                loss = scaffolding_batch_loss
            elif config['only_phase']:
                loss = phasing_batch_loss
            else:
                loss = scaffolding_batch_loss + phasing_batch_loss
            
            valid_loss.append(loss.item())
            
            # Print loss for debugging
            if config['only_scaffold']:
                print(f"Validation Loss: {loss.item():.4f} (Scaffolding only)")
            elif config['only_phase']:
                print(f"Validation Loss: {loss.item():.4f} (Phasing only)")
            else:
                print(f"Validation Loss: {loss.item():.4f} (Scaffolding: {scaffolding_batch_loss.item():.4f}, Phasing: {phasing_batch_loss.item():.4f})")
    
    # Calculate average metrics
    metrics = {
        'valid_loss': np.mean(valid_loss) if valid_loss else 0.0,
    }
    
    # Add individual loss components to metrics if they were computed
    if not config['only_phase'] and valid_scaffolding_loss:
        metrics['valid_scaffolding_loss'] = np.mean(valid_scaffolding_loss)
    if not config['only_scaffold'] and valid_phasing_loss:
        metrics['valid_phasing_loss'] = np.mean(valid_phasing_loss)
    
    print(f"Average Validation Loss: {metrics['valid_loss']:.4f}")
    
    return metrics

def evaluate_embeddings(model, data_path, eval_selection, device):
    """
    Evaluate how well the embeddings separate chromosomes and haplotypes.
    
    Args:
        model: Trained model
        data_path: Path to data
        eval_selection: List of graph names to evaluate
        device: Device to run computation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Metrics to track
    chr_separation_scores = []
    haplotype_separation_scores = []
    
    with torch.no_grad():
        for idx, graph_name in enumerate(eval_selection):
            print(f"Evaluating graph {graph_name}, id: {idx+1} of {len(eval_selection)}")
            g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
            
            # Normalize edge weights
            """g = normalize_edge_weights(g, edge_type=1, device=device)
            
            # Preprocess graph
            g, _ = preprocess_graph(g, device)"""
            #g.x = torch.cat([g.x.to(device), g.pe_0.to(device), g.pe_1.to(device)], dim=1)    
            g.x = torch.abs(g.y).float().unsqueeze(1).to(device) #torch.cat([torch.abs(g.y).float(), torch.abs(g.y).float()], dim=1)

            # Get embeddings
            phasing_embs, scaffolding_embs, _, _ = model(g)  # This returns a tuple
            
            # Get chromosome and haplotype labels
            chr_labels = g.chr
            hap_labels = g.y
            
            # Calculate chromosome separation score using scaffolding embeddings
            print(f"Scaffolding embeddings shape: {scaffolding_embs.shape}")
            chr_score = calculate_silhouette_score_torch(scaffolding_embs, chr_labels)
            chr_separation_scores.append(chr_score)
            
            # Calculate haplotype separation score within each chromosome using phasing embeddings
            hap_score = calculate_haplotype_separation_torch(phasing_embs, chr_labels, hap_labels)
            haplotype_separation_scores.append(hap_score)
            
            print(f"Chromosome separation score: {chr_score:.4f}")
            print(f"Haplotype separation score: {hap_score:.4f}")
    
    # Calculate average scores
    avg_chr_score = np.mean(chr_separation_scores)
    avg_hap_score = np.mean(haplotype_separation_scores)
    
    return {
        'chromosome_separation': avg_chr_score,
        'haplotype_separation': avg_hap_score,
        'chr_scores': chr_separation_scores,
        'hap_scores': haplotype_separation_scores
    }

def calculate_silhouette_score_torch(embeddings, labels):
    """
    Calculate silhouette score using PyTorch.
    
    Args:
        embeddings: Node embeddings tensor [n_samples, n_features]
        labels: Cluster labels tensor [n_samples]
        
    Returns:
        Silhouette score (-1 to 1, higher is better)
    """
    # Get unique labels
    unique_labels = torch.unique(labels)
    
    # Need at least 2 clusters for silhouette score
    if len(unique_labels) < 2:
        print("Only one cluster found, skipping silhouette score")
        return 0.0
    
    n_samples = embeddings.size(0)
    
    # Calculate pairwise distances between all samples
    # Using squared Euclidean distance for efficiency
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    # Initialize arrays for a (average distance to points in same cluster)
    # and b (average distance to points in nearest different cluster)
    a_values = torch.zeros(n_samples, device=embeddings.device)
    b_values = torch.full((n_samples,), float('inf'), device=embeddings.device)
    
    # Calculate a and b values for each sample
    for i, label in enumerate(unique_labels):
        # Mask for samples in this cluster
        mask = labels == label
        cluster_size = mask.sum().item()
        
        if cluster_size <= 1:
            # Skip clusters with only one sample
            continue
        
        # Calculate a values (mean distance to other points in same cluster)
        for idx in torch.where(mask)[0]:
            # Get distances to all points in same cluster
            cluster_distances = dist_matrix[idx][mask]
            # Exclude distance to self (which is 0)
            cluster_distances = cluster_distances[cluster_distances > 0]
            if len(cluster_distances) > 0:
                a_values[idx] = torch.mean(cluster_distances)
        
        # Calculate b values (mean distance to points in nearest different cluster)
        for other_label in unique_labels:
            if label == other_label:
                continue
                
            other_mask = labels == other_label
            other_cluster_size = other_mask.sum().item()
            
            if other_cluster_size == 0:
                continue
                
            # Calculate mean distance to points in other cluster
            for idx in torch.where(mask)[0]:
                other_distances = dist_matrix[idx][other_mask]
                mean_dist = torch.mean(other_distances)
                # Update b if this is the nearest different cluster
                b_values[idx] = torch.min(b_values[idx], mean_dist)
    
    # Calculate silhouette score for each sample
    s_values = torch.zeros(n_samples, device=embeddings.device)
    valid_mask = (a_values > 0) & (b_values < float('inf'))
    
    if valid_mask.sum() == 0:
        return 0.0
        
    # For valid samples, calculate (b - a) / max(a, b)
    s_values[valid_mask] = (b_values[valid_mask] - a_values[valid_mask]) / torch.max(a_values[valid_mask], b_values[valid_mask])
    
    # Return mean silhouette score
    return torch.mean(s_values).item()

def calculate_haplotype_separation_torch(embeddings, chr_labels, hap_labels):
    """
    Calculate how well embeddings separate haplotypes within the same chromosome.
    
    Args:
        embeddings: Node embeddings tensor
        chr_labels: Chromosome labels tensor
        hap_labels: Haplotype labels tensor
        
    Returns:
        Average silhouette score for haplotype separation within chromosomes
    """
    # Get unique chromosome labels
    unique_chrs = torch.unique(chr_labels)
    
    chr_hap_scores = []
    
    for chr_val in unique_chrs:
        # Get indices for this chromosome
        chr_mask = chr_labels == chr_val
        
        # Skip if too few nodes for this chromosome
        if chr_mask.sum() < 10:
            continue
        
        # Get embeddings and haplotype labels for this chromosome
        chr_embeddings = embeddings[chr_mask]
        chr_hap_labels = hap_labels[chr_mask]
        
        # Get unique haplotype labels for this chromosome
        unique_haps = torch.unique(chr_hap_labels)
        
        # Need at least 2 haplotypes for silhouette score
        if len(unique_haps) < 2:
            continue
            
        # Calculate silhouette score for this chromosome
        score = calculate_silhouette_score_torch(chr_embeddings, chr_hap_labels)
        chr_hap_scores.append(score)
    
    # Return average score across all chromosomes
    if chr_hap_scores:
        return np.mean(chr_hap_scores)
    else:
        return 0.0

def visualize_embeddings(model, data_path, graph_name, device, output_dir='embedding_plots'):
    """
    Visualize embeddings using t-SNE or UMAP.
    
    Args:
        model: Trained model
        data_path: Path to data
        graph_name: Name of graph to visualize
        device: Device to run computation on
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        # Load graph
        g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
        
        # Normalize edge weights
        """g = normalize_edge_weights(g, edge_type=1, device=device)
        
        # Preprocess graph
        g, _ = preprocess_graph(g, device)"""
        g.x = torch.cat([g.x.to(device), g.pe_0.to(device), g.pe_1.to(device)], dim=1)
        
        # Get embeddings
        phasing_embs, scaffolding_embs, phasing_projections, scaffolding_projections = model(g)  # This returns a tuple
        
        # Get chromosome and haplotype labels
        chr_labels = g.chr.cpu().numpy()
        hap_labels = g.y.cpu().numpy()
        
        # Create two plots - one for scaffolding (chromosome) embeddings and one for phasing embeddings
        
        # 1. Visualize scaffolding embeddings (for chromosomes)
        embeddings_np = scaffolding_embs.cpu().numpy()
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        
        # Plot by chromosome
        plt.figure(figsize=(12, 10))
        unique_chrs = np.unique(chr_labels)
        for chr_val in unique_chrs:
            mask = chr_labels == chr_val
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Chr {chr_val}', alpha=0.7)
        
        plt.title(f'Scaffolding Embeddings by Chromosome - {graph_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{graph_name}_chr_embeddings.png'))
        plt.close()
        
        # 2. Visualize phasing embeddings (for haplotypes within chromosomes)
        phasing_np = phasing_embs.cpu().numpy()
        
        # Plot by haplotype for each chromosome
        for chr_val in unique_chrs:
            plt.figure(figsize=(12, 10))
            chr_mask = chr_labels == chr_val
            
            # Skip chromosomes with too few nodes
            if np.sum(chr_mask) < 10:
                plt.close()
                continue
                
            # Get phasing embeddings for this chromosome
            chr_embeddings = phasing_np[chr_mask]
            
            # Apply t-SNE for this chromosome's embeddings
            if len(chr_embeddings) > 5:  # Need at least a few points for t-SNE
                tsne_chr = TSNE(n_components=2, random_state=42)
                chr_embeddings_2d = tsne_chr.fit_transform(chr_embeddings)
                
                chr_hap_labels = hap_labels[chr_mask]
                
                # Plot each haplotype
                for hap_val in np.unique(chr_hap_labels):
                    hap_mask = chr_hap_labels == hap_val
                    plt.scatter(chr_embeddings_2d[hap_mask, 0], chr_embeddings_2d[hap_mask, 1], 
                               label=f'Haplotype {hap_val}', alpha=0.7)
                
                plt.title(f'Phasing Embeddings for Chromosome {chr_val} - {graph_name}')
                plt.legend()
                plt.savefig(os.path.join(output_dir, f'{graph_name}_chr{chr_val}_hap_embeddings.png'))
            plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="experiment eval script")
    parser.add_argument("--load_checkpoint", type=str, default='', help="dataset path")
    parser.add_argument("--save_model_path", type=str, default='/mnt/sod2-project/csb4/wgs/martin/trained_models_march', help="dataset path")
    parser.add_argument("--data_path", type=str, default='data', help="dataset path")
    parser.add_argument("--run_name", type=str, default='test', help="dataset path")
    parser.add_argument("--device", type=str, default='cpu', help="dataset path")
    parser.add_argument("--data_config", type=str, default='dataset_config.yml', help="dataset path")
    parser.add_argument("--hyper_config", type=str, default='train_config.yml', help="dataset path")
    parser.add_argument("--wandb", type=str, default='debug', help="dataset path")

    args = parser.parse_args()

    wandb_project = args.wandb
    run_name = args.run_name
    load_checkpoint = args.load_checkpoint
    save_model_path = args.save_model_path
    data_path = args.data_path
    device = args.device
    data_config = args.data_config
    hyper_config = args.hyper_config
    full_dataset, valid_selection, train_selection = train_utils.create_dataset_dicts(data_config=data_config)
    train_data = train_utils.get_numbered_graphs(train_selection)
    valid_data = train_utils.get_numbered_graphs(valid_selection, starting_counts=train_selection)

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    with open(hyper_config) as file:
        config = yaml.safe_load(file)['training']
    
    train_utils.set_seed(config['seed'])
   
    """model = SGFormerC2(
        in_channels=config['node_features'],
        hidden_channels=config['hidden_features'],
        out_channels= config['emb_dim'],
        trans_num_layers=config['num_trans_layers'],
        trans_dropout= 0, #config['dropout'],
        gnn_num_layers=config['num_gnn_layers_overlap'],
        gnn_dropout= 0, #config['gnn_dropout'],
    ).to(device)"""

    model = ClusterGCN(in_channels=config['node_features'], hidden_channels=config['hidden_features'], out_channels=config['emb_dim'], num_layers=config['num_gnn_layers_overlap']).to(device)
    to_undirected = False
    model.to(device)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint, map_location=device))
        print(f"Loaded model from {load_checkpoint}")
    
    train(model, data_path, train_data, valid_data, device, config)

    # After training or if only evaluating
    if args.evaluate:
        print("Evaluating model performance...")
        eval_metrics = evaluate_embeddings(model, data_path, valid_data, device)
        print(f"Chromosome separation score: {eval_metrics['chromosome_separation']:.4f}")
        print(f"Haplotype separation score: {eval_metrics['haplotype_separation']:.4f}")
        
        # Visualize embeddings for a few graphs
        if args.visualize:
            for graph_name in valid_data[:3]:  # Visualize first 3 validation graphs
                visualize_embeddings(model, data_path, graph_name, device)