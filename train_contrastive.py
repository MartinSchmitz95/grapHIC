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
#from SGformer_contrastive import SGFormerContrastive
from SGformer_HG import SGFormer
from contrastive_loss import MultiLabelConLoss
    

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
    """
    Preprocess graph for contrastive learning.
    
    Args:
        g: Input graph
        device: Device to run computation on
        
    Returns:
        Processed graph and multi-label tensor
    """
    # Get chromosome and haplotype labels
    chr_tensor = g.chr
    y_tensor = g.y
    
    # Create a 2-dimensional multi-label vector
    multi_labels = torch.zeros((len(y_tensor), 2), dtype=torch.float, device=device)
    
    # First entry is 1 if y is 1 or 0
    multi_labels[:, 0] = (y_tensor == 1) | (y_tensor == 0)
    
    # Second entry is 1 if y is -1 or 0
    multi_labels[:, 1] = (y_tensor == -1) | (y_tensor == 0)
    
    return g, multi_labels

def train(model, data_path, train_selection, valid_selection, device, config):
    best_valid_loss = 10000
    overfit = not bool(valid_selection)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)  # Add weight decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['decay'], patience=config['patience'], verbose=True)

    # Track best evaluation metrics
    best_chr_separation = 0
    best_hap_separation = 0

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

            # Evaluate embeddings every n epochs
            if epoch % config['compute_metrics_every'] == 0:
                print('===> EVALUATING EMBEDDINGS')
                eval_metrics = evaluate_embeddings(model, data_path, valid_selection[:min(3, len(valid_selection))], device)
                
                # Track best scores
                if eval_metrics['chromosome_separation'] > best_chr_separation:
                    best_chr_separation = eval_metrics['chromosome_separation']
                if eval_metrics['haplotype_separation'] > best_hap_separation:
                    best_hap_separation = eval_metrics['haplotype_separation']
                
                # Add evaluation metrics to log
                valid_metrics.update({
                    'chr_separation': eval_metrics['chromosome_separation'],
                    'hap_separation': eval_metrics['haplotype_separation'],
                    'best_chr_separation': best_chr_separation,
                    'best_hap_separation': best_hap_separation
                })
                
                # Optionally visualize embeddings
                if config['visualize_during_training'] and epoch > 0:
                    for graph_name in valid_selection[:1]:  # Visualize just the first validation graph
                        output_dir = os.path.join('embedding_plots', f'epoch_{epoch}')
                        visualize_embeddings(model, data_path, graph_name, device, output_dir)

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
    compute_metrics = False #(epoch % config['compute_metrics_every'] == 0)
    train_phasing_acc, train_scaffolding_acc = [], []
    emb_loss = MultiLabelConLoss()
    batch_size = 4096  # Set batch size for sampling nodes

    for idx, graph_name in enumerate(train_selection):
        print(f"Training graph {graph_name}, id: {idx} of {len(train_selection)}")
        g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
        # Normalize edge weights once when loading the graph
        g = normalize_edge_weights(g, edge_type=1, device=device)
        g, labels = preprocess_graph(g, device)
        optimizer.zero_grad()
                
        # Get embeddings for all nodes
        embs = model(g)
        
        # Randomly sample nodes if we have more than batch_size
        num_nodes = embs.size(0)
        if num_nodes > batch_size:
            # Generate random indices for sampling
            indices = torch.randperm(num_nodes, device=device)[:batch_size]
            
            # Sample embeddings and labels
            sampled_embs = embs[indices]
            sampled_labels = labels[indices]
            
            # Compute loss on the sampled batch
            loss = emb_loss(sampled_embs, sampled_labels)
        else:
            # If we have fewer nodes than batch_size, use all nodes
            loss = emb_loss(embs, labels)

        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss.append(loss.item())

        # Print loss for debugging
        print(f"Loss: {loss.item():.4f}")
    metrics = {
        'train_loss': np.mean(train_loss),
    }
    
    
    return metrics

def validate_epoch(model, valid_selection, data_path, device, 
                  config, epoch):
    """Validate the model on the validation set using contrastive learning."""
    valid_loss = []
    emb_loss = MultiLabelConLoss()
    batch_size = 4096  # Same batch size as in training
    
    with torch.no_grad():
        for idx, graph_name in enumerate(valid_selection):
            print(f"Validating graph {graph_name}, id: {idx+1} of {len(valid_selection)}")
            g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
            
            # Normalize edge weights
            g = normalize_edge_weights(g, edge_type=1, device=device)
            
            # Preprocess graph
            g, labels = preprocess_graph(g, device)
            
            # Get embeddings for all nodes
            embs = model(g)
            
            # Randomly sample nodes if we have more than batch_size
            num_nodes = embs.size(0)
            if num_nodes > batch_size:
                # Generate random indices for sampling
                indices = torch.randperm(num_nodes, device=device)[:batch_size]
                
                # Sample embeddings and labels
                sampled_embs = embs[indices]
                sampled_labels = labels[indices]
                
                # Compute loss on the sampled batch
                loss = emb_loss(sampled_embs, sampled_labels)
            else:
                # If we have fewer nodes than batch_size, use all nodes
                loss = emb_loss(embs, labels)
            
            valid_loss.append(loss.item())
            
            # Print loss for debugging
            print(f"Validation Loss: {loss.item():.4f}")
    
    # Calculate average metrics
    metrics = {
        'valid_loss': np.mean(valid_loss) if valid_loss else 0.0,
    }
    
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
            g = normalize_edge_weights(g, edge_type=1, device=device)
            
            # Preprocess graph
            g, _ = preprocess_graph(g, device)
            
            # Get embeddings
            embeddings = model(g)
            
            # Get chromosome and haplotype labels
            chr_labels = g.chr
            hap_labels = g.y
            
            # Calculate chromosome separation score
            chr_score = calculate_silhouette_score_torch(embeddings, chr_labels)
            chr_separation_scores.append(chr_score)
            
            # Calculate haplotype separation score within each chromosome
            hap_score = calculate_haplotype_separation_torch(embeddings, chr_labels, hap_labels)
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
        g = normalize_edge_weights(g, edge_type=1, device=device)
        
        # Preprocess graph
        g, _ = preprocess_graph(g, device)
        
        # Get embeddings
        embeddings = model(g)
        
        # Get chromosome and haplotype labels
        chr_labels = g.chr.cpu().numpy()
        hap_labels = g.y.cpu().numpy()
        
        # Convert embeddings to numpy for t-SNE
        embeddings_np = embeddings.cpu().numpy()
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_np)
        
        # Plot by chromosome
        plt.figure(figsize=(12, 10))
        unique_chrs = np.unique(chr_labels)
        for chr_val in unique_chrs:
            mask = chr_labels == chr_val
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Chr {chr_val}', alpha=0.7)
        
        plt.title(f'Embeddings by Chromosome - {graph_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{graph_name}_chr_embeddings.png'))
        plt.close()
        
        # Plot by haplotype for each chromosome
        for chr_val in unique_chrs:
            plt.figure(figsize=(12, 10))
            chr_mask = chr_labels == chr_val
            
            # Skip chromosomes with too few nodes
            if np.sum(chr_mask) < 10:
                plt.close()
                continue
                
            chr_embeddings = embeddings_2d[chr_mask]
            chr_hap_labels = hap_labels[chr_mask]
            
            # Plot each haplotype
            for hap_val in np.unique(chr_hap_labels):
                hap_mask = chr_hap_labels == hap_val
                plt.scatter(chr_embeddings[hap_mask, 0], chr_embeddings[hap_mask, 1], 
                           label=f'Haplotype {hap_val}', alpha=0.7)
            
            plt.title(f'Embeddings for Chromosome {chr_val} by Haplotype - {graph_name}')
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
    parser.add_argument('--diploid', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument("--wandb", type=str, default='debug', help="dataset path")
    parser.add_argument('--symmetry', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--aux', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--bce', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--hap_switch', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument('--quick', action='store_true', default=False, help="Enable evaluation (default: True)")
    parser.add_argument("--seed", type=int, default=0, help="dataset path")
    parser.add_argument('--evaluate', action='store_true', default=False, help="Evaluate model performance")
    parser.add_argument('--visualize', action='store_true', default=False, help="Visualize embeddings")

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
   
    model = SGFormer(
        in_channels=config['node_features'],
        hidden_channels=config['hidden_features'],
        out_channels= config['emb_dim'],
        trans_num_layers=config['num_trans_layers'],
        trans_dropout= 0, #config['dropout'],
        gnn_num_layers_0=config['num_gnn_layers_overlap'],
        gnn_num_layers_1=config['num_gnn_layers_hic'],
        gnn_dropout= 0, #config['gnn_dropout'],
    ).to(device)

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