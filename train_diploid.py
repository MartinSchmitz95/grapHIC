import torch
import wandb
import random
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import numpy as np
import train_utils
import argparse
import yaml
#from SGformer_HGC import SGFormer as SGFormer_HGC
from GIN import HeteroGIN
from SGformer import SGFormer, SGFormerMulti, SGFormer_NoGate, SGFormerEdgeEmbs, SGFormerGINEdgeEmbs, SGFormer_L_Pred
from losses_diploid import GlobalPairLossConsise, LocalPairLossConsise, TripletLoss, ContrastiveEmbeddingLoss, SharedToZeroLoss, SimpleBCEWithAbsLoss, MeanZeroLoss

# Limit PyTorch to use only 16 CPU cores
torch.set_num_threads(16)

emb_losses = ["subcon", "info_nce", "full_cont"]
pred_losses = ["pairloss_global", "pairloss_local", "triplet_loss"]

def sample_subgraph_by_chromosomes(g, max_nodes, device):
    """
    Sample a subgraph by randomly selecting chromosomes until max_nodes is reached.
    Optimized version for better performance.
    
    Args:
        g: PyTorch Geometric graph
        max_nodes: Maximum number of nodes allowed
        device: Device to place tensors on
    
    Returns:
        g_sampled: Sampled subgraph with nodes from selected chromosomes
    """
    if g.x.shape[0] <= max_nodes:
        return g
    
    # Get unique chromosomes and shuffle them efficiently
    unique_chr = torch.unique(g.chr)
    chr_order = unique_chr[torch.randperm(len(unique_chr), device=device)]
    
    # Pre-allocate tensors for efficiency
    selected_nodes = []
    total_nodes = 0
    
    for chr_id in chr_order:
        # Get nodes for current chromosome
        chr_mask = (g.chr == chr_id)
        chr_nodes = torch.where(chr_mask)[0]
        chr_node_count = len(chr_nodes)
        
        if total_nodes + chr_node_count <= max_nodes:
            # Add all nodes from this chromosome
            selected_nodes.append(chr_nodes)
            total_nodes += chr_node_count
        else:
            # Add partial nodes from this chromosome to reach max_nodes
            remaining_nodes = max_nodes - total_nodes
            if remaining_nodes > 0:
                # Randomly sample remaining_nodes from this chromosome
                perm = torch.randperm(chr_node_count, device=device)
                selected_nodes.append(chr_nodes[perm[:remaining_nodes]])
                total_nodes += remaining_nodes
            break

    # Combine all selected nodes efficiently
    node_indices = torch.cat(selected_nodes)
    
    # Create subgraph efficiently
    g_sampled = g.clone()
    g_sampled.x = g.x[node_indices]
    g_sampled.y = g.y[node_indices]
    g_sampled.chr = g.chr[node_indices]
    
    # Create node mapping for efficient edge remapping
    node_mapping = torch.zeros(g.x.shape[0], dtype=torch.long, device=device)
    node_mapping[node_indices] = torch.arange(len(node_indices), device=device)
    
    # Filter edges efficiently
    edge_mask = torch.isin(g.edge_index[0], node_indices) & torch.isin(g.edge_index[1], node_indices)
    g_sampled.edge_index = g.edge_index[:, edge_mask]
    
    # Update edge attributes if they exist
    if hasattr(g, 'edge_weight'):
        g_sampled.edge_weight = g.edge_weight[edge_mask]
    if hasattr(g, 'edge_type'):
        g_sampled.edge_type = g.edge_type[edge_mask]
    
    # Remap node indices efficiently using the pre-computed mapping
    g_sampled.edge_index[0] = node_mapping[g_sampled.edge_index[0]]
    g_sampled.edge_index[1] = node_mapping[g_sampled.edge_index[1]]

    return g_sampled

def global_phasing_quotient(y_true, y_pred, chr, debug=False):
    device = y_true.device
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    
    def calculate_fractions(label, mask):
        combined_mask = (y_true == label) & mask
        preds = y_pred[combined_mask]
        if len(preds) > 0:
            sum_smaller_zero = (preds < 0).float().sum().item()
            sum_larger_zero = (preds >= 0).float().sum().item()
        else:
            sum_smaller_zero = 0.0
            sum_larger_zero = 0.0
        return sum_smaller_zero, sum_larger_zero

    # Get unique chromosomes
    unique_chr = torch.unique(chr)
    total_correct = 0
    total_samples = 0

    for c in unique_chr:
        # Create mask for current chromosome
        chr_mask = (chr == c)
        
        # Skip chromosomes with no valid labels
        if not ((y_true[chr_mask] == 1).any() or (y_true[chr_mask] == -1).any()):
            continue

        # Calculate fractions for -1 and 1 for this chromosome
        sum_neg1_smaller_zero, sum_neg1_larger_zero = calculate_fractions(-1, chr_mask)
        sum_pos1_smaller_zero, sum_pos1_larger_zero = calculate_fractions(1, chr_mask)
        
        # Evaluate both combinations
        combination1 = (sum_neg1_smaller_zero + sum_pos1_larger_zero)
        combination2 = (sum_neg1_larger_zero + sum_pos1_smaller_zero)
        chr_total = combination1 + combination2
        
        if chr_total > 0:  # Only include chromosomes with valid predictions
            chr_correct = max(combination1, combination2)
            total_correct += chr_correct
            total_samples += chr_total
            if debug:
                print(f"Chr {c} correct/total: {chr_correct}/{chr_total} = {chr_correct/chr_total:.3f}")

    # Return metric based on all samples
    if total_samples > 0:
        final_metric = total_correct / total_samples
        print(f"Global Phasing Quotient: {total_correct}/{total_samples} = {final_metric:.3f}")
        return final_metric
    else:
        print("No valid samples found")
        return 0.0
    
def local_phasing_quotient(y_true, y_pred, chr, edge_index, edge_type, debug=False):
    if debug:
        return local_phasing_quotient_debug(y_true, y_pred, chr, edge_index, edge_type)
    else:
        return local_phasing_quotient_no_debug(y_true, y_pred, chr, edge_index, edge_type)

def local_phasing_quotient_debug(y_true, y_pred, chr, edge_index, edge_type):
    device = y_true.device
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    
    # Get source and destination nodes
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # Get unique chromosomes
    unique_chr = torch.unique(chr)
    total_correct = 0
    total_edges = 0
    
    for c in unique_chr:
        # Create combined mask for current chromosome
        chr_mask = (
            (edge_type == 0) &                           # overlap edges
            (chr[src_nodes] == c) &                      # source node in current chromosome
            (chr[dst_nodes] == c) &                      # destination node in current chromosome
            (y_true[src_nodes] != 0) &                   # non-zero source labels
            (y_true[dst_nodes] != 0)                     # non-zero destination labels
        )
        
        if chr_mask.sum() == 0:
            continue
            
        # Apply the mask for current chromosome
        chr_src_nodes = src_nodes[chr_mask]
        chr_dst_nodes = dst_nodes[chr_mask]
        
        # Get true labels and predictions for current chromosome
        chr_src_true = y_true[chr_src_nodes]
        chr_dst_true = y_true[chr_dst_nodes]
        chr_src_pred = y_pred[chr_src_nodes]
        chr_dst_pred = y_pred[chr_dst_nodes]
        
        # Calculate if predictions have same/different signs
        chr_pred_same_phase = (chr_src_pred * chr_dst_pred) >= 0
        chr_true_same_phase = (chr_src_true * chr_dst_true) >= 0
        
        # Count correct phasings for current chromosome
        chr_correct = (chr_pred_same_phase == chr_true_same_phase).sum().item()
        chr_total = chr_mask.sum().item()
        
        print(f"Chr {c} correct/total: {chr_correct}/{chr_total} = {chr_correct/chr_total:.3f}")
            
        total_correct += chr_correct
        total_edges += chr_total
    
    if total_edges == 0:
        print("No valid edges found")
        return 0.0
        
    final_metric = total_correct / total_edges
    print(f"Local Phasing Quotient: {total_correct}/{total_edges} = {final_metric:.3f}")
    return final_metric

def local_phasing_quotient_no_debug(y_true, y_pred, chr, edge_index, edge_type):
    device = y_true.device
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    
    # Get source and destination nodes
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    # Create combined mask for all conditions at once
    combined_mask = (
        (edge_type == 0) &                           # overlap edges
        (chr[src_nodes] == chr[dst_nodes]) &        # same chromosome
        (y_true[src_nodes] != 0) &                  # non-zero source labels
        (y_true[dst_nodes] != 0)                    # non-zero destination labels
    )
    
    # Apply the combined mask once
    src_nodes = src_nodes[combined_mask]
    dst_nodes = dst_nodes[combined_mask]
    
    # Get true labels and predictions
    src_true = y_true[src_nodes]
    dst_true = y_true[dst_nodes]
    src_pred = y_pred[src_nodes]
    dst_pred = y_pred[dst_nodes]
    
    total_edges = combined_mask.sum().item()
    if total_edges == 0:
        print("No valid edges found")
        return 0.0
        
    # Calculate if predictions have same/different signs
    pred_same_phase = (src_pred * dst_pred) >= 0
    true_same_phase = (src_true * dst_true) >= 0
    
    # Count correct phasings
    total_correct = (pred_same_phase == true_same_phase).sum().item()
    
    final_metric = total_correct / total_edges
    print(f"Local Phasing Quotient: {total_correct}/{total_edges} = {final_metric:.3f}")
    return final_metric

def homo_separation(y_true, y_pred, debug=False):
    """
    Measures how well homozygous nodes (y=0) are separated from heterozygous nodes (y=-1, y=1).
    Tries different epsilon thresholds and returns the best F1 score.
    Nodes with |prediction| > epsilon are classified as heterozygous.
    """
    device = y_true.device
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    
    # Create binary labels: 0 for homozygous (y=0), 1 for heterozygous (y!=0)
    homo_mask = (y_true == 0)
    hetero_mask = (y_true != 0)
    
    homo_count = homo_mask.sum().item()
    hetero_count = hetero_mask.sum().item()
    
    if homo_count == 0 or hetero_count == 0:
        if debug:
            print(f"Homo separation: insufficient data (homo: {homo_count}, hetero: {hetero_count})")
        return 0.0, 0.0
    
    # Create binary ground truth labels
    binary_true = hetero_mask.float()  # 1 for heterozygous, 0 for homozygous
    
    # Get absolute predictions
    abs_pred = torch.abs(y_pred)
    
    # Try different epsilon thresholds
    # Use fixed thresholds from 0.01 to 1.00 with 0.01 increments
    epsilons = torch.arange(0.01, 1.01, 0.01, device=device)
    
    best_f1 = 0.0
    best_epsilon = 0.0
    
    for epsilon in epsilons:
        # Predict heterozygous if |prediction| > epsilon
        binary_pred = (abs_pred > epsilon).float()
        
        # Calculate true positives, false positives, and false negatives
        true_positives = ((binary_pred == 1) & (binary_true == 1)).sum().item()
        false_positives = ((binary_pred == 1) & (binary_true == 0)).sum().item()
        false_negatives = ((binary_pred == 0) & (binary_true == 1)).sum().item()
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon.item()
    
    if debug:
        # Additional debug info
        homo_mean_abs = abs_pred[homo_mask].mean().item() if homo_count > 0 else 0
        hetero_mean_abs = abs_pred[hetero_mask].mean().item() if hetero_count > 0 else 0
        print(f"Homo separation: best_f1={best_f1:.4f}, best_epsilon={best_epsilon:.4f}")
        print(f"  homo_mean_abs={homo_mean_abs:.4f}, hetero_mean_abs={hetero_mean_abs:.4f}")
        print(f"  homo_count={homo_count}, hetero_count={hetero_count}")
    
    return best_f1, best_epsilon

def process_batch(model, data_selection, data_path, device, optimizer, objective, aux_loss, config, epoch, mode, is_training=True, project_embeddings=False):
    losses = []
    pred_mean, pred_std = [], []
    compute_metrics = (epoch % config['compute_metrics_every'] == 0)
    #if not project_embeddings:
    #    compute_metrics = False
    global_phasing_metrics, local_phasing_metrics = [], []
    homo_separation_metrics = []
    homo_separation_epsilons = []
    
    phase = "Training" if is_training else "Validating"
    
    for idx, graph_name in enumerate(data_selection):
        print(f"{phase} graph {graph_name}, id: {idx} of {len(data_selection)}")
        g = torch.load(os.path.join(data_path, graph_name + '.pt'), map_location=device)
        
        #x_addon = torch.abs(g.y).float().unsqueeze(1).to(device)
        #g.x = torch.cat([g.x, x_addon], dim=1)
        #print(g.x)
        #g.x = torch.zeros_like(torch.abs(g.y).float().unsqueeze(1).to(device))
        if not hasattr(g, 'chr'):
            g.chr = torch.ones(g.x.shape[0], dtype=torch.long, device=device)
        
        # Apply subgraph sampling if max_nodes is specified
        if 'max_nodes' in config and config['max_nodes'] > 0:
            g = sample_subgraph_by_chromosomes(g, config['max_nodes'], device)
        
        if is_training:
            optimizer.zero_grad()
        edge_weight = g.edge_weight
        # Use torch.no_grad for validation
        with torch.set_grad_enabled(is_training):
            predictions = model(g)
            if mode in pred_losses:
                predictions = predictions.squeeze()
                pred_mean.append(predictions.mean().item())
                pred_std.append(predictions.std().item())
            
            # Check if graph has multiple chromosomes
            has_multiple_chr = len(torch.unique(g.chr)) > 1
            print(f"has_multiple_chr: {has_multiple_chr}")

            
            # Check if the loss function needs the graph parameter
            if hasattr(objective, 'multi_chr_forward') and 'g' in objective.multi_chr_forward.__code__.co_varnames:
                # Loss functions that need the graph parameter (LocalPairLoss, LocalPairLossConsise, etc.)
                loss = objective(g.y, predictions, g.edge_index[0], g.edge_index[1], g, g.chr, multi=has_multiple_chr)
            else:
                # Loss functions that don't need the graph parameter (GlobalPairLossConsise, etc.)
                loss = objective(g.y, predictions, g.edge_index[0], g.edge_index[1], g.chr, multi=has_multiple_chr)
            if aux_loss:
                aux_loss_term = aux_loss(g.y, predictions)
                loss += aux_loss_term
            
            if is_training:
                loss.backward()
                optimizer.step()
                
            losses.append(loss.item())
            
            if compute_metrics:
                debug = not is_training  # Only debug during validation
                global_phasing_metrics.append(global_phasing_quotient(g.y, predictions, g.chr, debug=debug))
                local_phasing_metrics.append(local_phasing_quotient(g.y, predictions, g.chr, 
                                                                g.edge_index, g.edge_type, debug=debug))
                homo_accuracy, homo_epsilon = homo_separation(g.y, predictions, debug=debug)
                homo_separation_metrics.append(homo_accuracy)
                homo_separation_epsilons.append(homo_epsilon)
    
    prefix = "valid_" if not is_training else "train_"
    metrics = {
        f'{prefix}loss': np.mean(losses),
        f'{prefix}mean': np.mean(pred_mean) if pred_mean else 0,
        f'{prefix}std': np.mean(pred_std) if pred_std else 0
    }
    
    if compute_metrics:
        metrics.update({
            f'{prefix}global_phasing': np.mean(global_phasing_metrics),
            f'{prefix}local_phasing': np.mean(local_phasing_metrics),
            f'{prefix}homo_separation': np.mean(homo_separation_metrics),
            f'{prefix}homo_epsilon': np.mean(homo_separation_epsilons)
        })
    
    return metrics

def train(model, data_path, train_selection, valid_selection, device, mode, project_embeddings, config):
    best_valid_loss = 10000
    overfit = not bool(valid_selection)

    if mode == 'pairloss_global':
        objective = GlobalPairLossConsise().to(device)
    elif mode == 'pairloss_local':
        objective = LocalPairLossConsise().to(device)
    elif mode == 'triplet_loss':
        objective = TripletLoss().to(device)
    elif mode == 'subcon':
        objective = ContrastiveEmbeddingLoss(mode='n_vs_n', n_samples=16).to(device)
    elif mode == 'info_nce':
        objective = ContrastiveEmbeddingLoss(mode='1_vs_n', n_samples=16).to(device)
    elif mode == 'full_cont':
        objective = ContrastiveEmbeddingLoss(mode='all_vs_all').to(device)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    if mode in pred_losses:
        #aux_loss = SharedToZeroLoss(weight=config['aux_loss_weight']).to(device)
        #aux_loss = SimpleBCEWithAbsLoss(weight=config['aux_loss_weight']).to(device)
        aux_loss = MeanZeroLoss(weight=config['aux_loss_weight']).to(device)
    else:
        aux_loss = None

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['decay'], patience=config['patience'], verbose=True)

    time_start = datetime.now()
    with wandb.init(project=wandb_project, config=config, mode=config['wandb_mode'], name=run_name):

        for epoch in range(config['num_epochs']):
            print(f"Epoch {epoch} of {config['num_epochs']}")
            
            # Training phase
            print('===> TRAINING')
            model.train()
            random.shuffle(train_selection)
            train_metrics = process_batch(model, train_selection, data_path, device, optimizer, 
                                      objective, aux_loss, config, epoch, mode, is_training=True, project_embeddings=project_embeddings)

            # Validation phase (if applicable)
            valid_metrics = {}
            if not overfit:
                print('===> VALIDATION')
                model.eval()
                valid_metrics = process_batch(model, valid_selection, data_path, device, optimizer,
                                            objective, aux_loss, config, epoch, mode, is_training=False, project_embeddings=project_embeddings)
                #scheduler.step(valid_metrics['valid_loss'])
                
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
            
            # Print metrics to console
            print(f"\nEpoch {epoch} Metrics:")
            # Only print loss and phasing metrics
            metrics_to_print = ['train_loss', 'valid_loss', 'train_global_phasing', 'valid_global_phasing', 'train_local_phasing', 'valid_local_phasing', 'train_homo_separation', 'valid_homo_separation', 'train_homo_epsilon', 'valid_homo_epsilon']
            for key in metrics_to_print:
                if key in log_dict:
                    value = log_dict[key]
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            print("-" * 50)

            if (epoch+1) % 100 == 0:
                model_path = os.path.join(save_model_path, f'{args.run_name}_{epoch+1}.pt')
                torch.save(model.state_dict(), model_path)
                print("Saved model")

def get_graph_data(data_path, data_selection, config=None):
    """
    Analyze graph data and display statistics about nodes and edges.
    
    Args:
        data_path: Path to the directory containing graph data
        data_selection: List of graph names to analyze
        config: Configuration dictionary containing max_nodes parameter
    
    Returns:
        dict: Dictionary containing statistics about the graphs
    """
    print("Analyzing graph dataset...")
    
    # Initialize counters
    total_nodes = 0
    total_edges = 0
    total_edges_type0 = 0
    total_edges_type1 = 0
    
    # Lists to store counts for calculating std
    node_counts = []
    edge_counts = []
    edge_type0_counts = []
    edge_type1_counts = []
    
    # Lists to store node features for calculating mean and std
    all_node_features = []
    
    for graph_name in data_selection:
        g = torch.load(os.path.join(data_path, graph_name + '.pt'))
        
        # Apply subgraph sampling if max_nodes is specified in config
        if config and 'max_nodes' in config and config['max_nodes'] > 0:
            device = torch.device('cpu')  # Use CPU for analysis
            g = sample_subgraph_by_chromosomes(g, config['max_nodes'], device)
        
        # Collect node features
        all_node_features.append(g.x)
        
        # Count nodes
        num_nodes = g.x.shape[0]
        total_nodes += num_nodes
        node_counts.append(num_nodes)
        
        # Count edges
        num_edges = g.edge_index.shape[1]
        total_edges += num_edges
        edge_counts.append(num_edges)
        
        # Count edges by type
        if hasattr(g, 'edge_type'):
            num_edges_type0 = (g.edge_type == 0).sum().item()
            num_edges_type1 = (g.edge_type == 1).sum().item()
            
            total_edges_type0 += num_edges_type0
            total_edges_type1 += num_edges_type1
            
            edge_type0_counts.append(num_edges_type0)
            edge_type1_counts.append(num_edges_type1)
    
    # Calculate node feature statistics
    if all_node_features:
        all_features = torch.cat(all_node_features, dim=0)
        feature_means = all_features.mean(dim=0)
        feature_stds = all_features.std(dim=0)
    
    # Calculate statistics
    num_graphs = len(data_selection)
    
    stats = {
        "num_graphs": num_graphs,
        "nodes_total": total_nodes,
        "nodes_mean": total_nodes / num_graphs if num_graphs > 0 else 0,
        "nodes_std": np.std(node_counts) if node_counts else 0,
        "nodes_min": min(node_counts) if node_counts else 0,
        "nodes_max": max(node_counts) if node_counts else 0,
        "edges_total": total_edges,
        "edges_mean": total_edges / num_graphs if num_graphs > 0 else 0,
        "edges_std": np.std(edge_counts) if edge_counts else 0,
        "edges_min": min(edge_counts) if edge_counts else 0,
        "edges_max": max(edge_counts) if edge_counts else 0,
    }
    
    if edge_type0_counts:
        stats.update({
            "edges_type0_total": total_edges_type0,
            "edges_type0_mean": total_edges_type0 / num_graphs if num_graphs > 0 else 0,
            "edges_type0_std": np.std(edge_type0_counts),
            "edges_type0_min": min(edge_type0_counts),
            "edges_type0_max": max(edge_type0_counts),
            "edges_type1_total": total_edges_type1,
            "edges_type1_mean": total_edges_type1 / num_graphs if num_graphs > 0 else 0,
            "edges_type1_std": np.std(edge_type1_counts),
            "edges_type1_min": min(edge_type1_counts),
            "edges_type1_max": max(edge_type1_counts),
        })
    
    # Print statistics
    print(f"Dataset Statistics:")
    print(f"  Number of graphs: {stats['num_graphs']}")
    print(f"  Nodes: total={stats['nodes_total']}, mean={stats['nodes_mean']:.2f}, std={stats['nodes_std']:.2f}, min={stats['nodes_min']}, max={stats['nodes_max']}")
    print(f"  Edges: total={stats['edges_total']}, mean={stats['edges_mean']:.2f}, std={stats['edges_std']:.2f}, min={stats['edges_min']}, max={stats['edges_max']}")
    
    if edge_type0_counts:
        print(f"  Edges (type 0): total={stats['edges_type0_total']}, mean={stats['edges_type0_mean']:.2f}, std={stats['edges_type0_std']:.2f}, min={stats['edges_type0_min']}, max={stats['edges_type0_max']}")
        print(f"  Edges (type 1): total={stats['edges_type1_total']}, mean={stats['edges_type1_mean']:.2f}, std={stats['edges_type1_std']:.2f}, min={stats['edges_type1_min']}, max={stats['edges_type1_max']}")
    
    # Print node feature statistics
    if all_node_features:
        print("\nNode Feature Statistics:")
        for i, (mean, std) in enumerate(zip(feature_means, feature_stds)):
            print(f"  Feature {i}: mean={mean:.4f}, std={std:.4f}")
    
    return stats

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="experiment eval script")
    parser.add_argument("--load_checkpoint", type=str, default='', help="dataset path")
    parser.add_argument("--save_model_path", type=str, default='/mnt/sod2-project/csb4/wgs/martin/trained_models', help="dataset path")
    parser.add_argument("--data_path", type=str, default='data', help="dataset path")
    parser.add_argument("--run_name", type=str, default='test', help="dataset path")
    parser.add_argument("--device", type=str, default='cpu', help="dataset path")
    parser.add_argument("--data_config", type=str, default='dataset_config.yml', help="dataset path")
    parser.add_argument("--hyper_config", type=str, default='train_config.yml', help="dataset path")
    parser.add_argument('--diploid', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument("--wandb", type=str, default='debug', help="dataset path")
    parser.add_argument("--seed", type=int, default=0, help="dataset path")
    parser.add_argument("--mode", type=str, default='pairloss_global', help="dataset path")
    parser.add_argument("--project", action='store_true', default=False, help="Enable debug mode")

    args = parser.parse_args()

    wandb_project = args.wandb
    run_name = args.run_name
    load_checkpoint = args.load_checkpoint
    save_model_path = args.save_model_path
    data_path = args.data_path
    device = args.device
    data_config = args.data_config
    diploid = args.diploid
    hyper_config = args.hyper_config
    mode = args.mode
    full_dataset, valid_selection, train_selection = train_utils.create_dataset_dicts(data_config=data_config)
    train_data = train_utils.get_numbered_graphs(train_selection)
    valid_data = train_utils.get_numbered_graphs(valid_selection, starting_counts=train_selection)
    project_embeddings = args.project

    # Verify device and print device information
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"Warning: CUDA device {device} requested but CUDA is not available. Falling back to CPU.")
            device = 'cpu'
        else:
            # Get the GPU index from the device string
            gpu_idx = int(device.split(':')[1]) if ':' in device else 0
            if gpu_idx >= torch.cuda.device_count():
                print(f"Warning: GPU {gpu_idx} not available. Available GPUs: {torch.cuda.device_count()}. Falling back to GPU 0.")
                device = f'cuda:0'
            else:
                # Set the current device
                torch.cuda.set_device(gpu_idx)
                print(f"Using GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3:.1f} GB")
    else:
        print(f"Using device: {device}")

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    with open(hyper_config) as file:
        config = yaml.safe_load(file)['training']

    train_utils.set_seed(args.seed)
   
    #model = SGFormerMulti(in_channels=config['node_features'], hidden_channels=config['hidden_features'], out_channels=1, trans_num_layers=config['num_trans_layers'], gnn_num_layers=config['num_gnn_layers'], gnn_dropout=config['gnn_dropout'], layer_norm=config['layer_norm'], num_blocks=config['num_blocks']).to(device)
    if project_embeddings:
        model = SGFormer_HGC(
            in_channels=config['node_features'],
            hidden_channels=config['hidden_features'],
            out_channels= config['emb_dim'],
            projection_dim=config['projection_dim'],
            trans_num_layers=config['num_trans_layers'],
            trans_dropout= 0, #config['dropout'],
            gnn_num_layers=config['num_gnn_layers'],
            gnn_dropout= config['dropout'],
            layer_norm=config['layer_norm'],
            direct_ftrs=config['direct_ftrs']
        ).to(device)
    else:
        """model = SGFormer(in_channels=config['node_features'], hidden_channels=config['hidden_features'],
                          out_channels=1, trans_num_layers=config['num_trans_layers'],
                          gnn_num_layers=config['num_gnn_layers'], gnn_dropout=config['gnn_dropout'],
                            norm=config['norm'], direct_ftrs=config['direct_ftrs']).to(device)"""
        
        model = SGFormerEdgeEmbs(in_channels=config['node_features'], hidden_channels=config['hidden_features'],
                          out_channels=1, trans_num_layers=config['num_trans_layers'],
                          gnn_num_layers=config['num_gnn_layers'], gnn_dropout=config['gnn_dropout'],
                            norm=config['norm'], direct_ftrs=config['direct_ftrs']).to(device)
        """model = HeteroGIN(
            in_channels=config['node_features'],
            hidden_channels=config['hidden_features'],
            out_channels=1,  # For prediction tasks, output scalar per node
            num_layers=config['num_gnn_layers'],
            dropout=config['gnn_dropout'],
            direct_ftrs=config['direct_ftrs']
        ).to(device)
        
        
        model = SGFormer_NoGate(
            in_channels=config['node_features'],
            hidden_channels=config['hidden_features'],
            out_channels=1,
            trans_num_layers=config['num_trans_layers'],
            trans_dropout=0,
            gnn_num_layers=config['num_gnn_layers'],
            gnn_dropout=config['gnn_dropout'],
            edge_feature_dim=1,
            direct_ftrs=config['direct_ftrs']
        ).to(device)"""
    #latest change: half hidden channels, reduce gnn_layers, remove dropout
    to_undirected = False
    model.to(device)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint, map_location=device))
        print(f"Loaded model from {load_checkpoint}")

    # Get dataset statistics before training
    print("Analyzing training data:")
    train_stats = get_graph_data(data_path, train_data, config)
    if valid_data:
        print("\nAnalyzing validation data:")
        valid_stats = get_graph_data(data_path, valid_data, config)

    train(model, data_path, train_data, valid_data, device, mode, project_embeddings, config)

    #python train_yak2.py --save_model_path /mnt/sod2-project/csb4/wgs/martin/rl_dgl_datasets/trained_models/ --data_path /mnt/sod2-project/csb4/wgs/martin/diploid_datasets/diploid_dataset_hg002_cent/pyg_graphs/ --data_config dataset_diploid_cent.yml --hyper_config config_yak.yml --wandb pred_yak --run_name yak2_SGformer_base --device cuda:4 --diploid
