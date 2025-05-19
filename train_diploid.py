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
from SGformer_HGC import SGFormer as SGFormer_HGC
from SGformer import SGFormer, SGFormerMulti
from losses_diploid import GlobalPairLossConsise, LocalPairLossConsise, TripletLoss, ContrastiveEmbeddingLoss, SharedToZeroLoss

emb_losses = ["subcon", "info_nce", "full_cont"]
pred_losses = ["pairloss_global", "pairloss_local", "triplet_loss"]

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

def process_batch(model, data_selection, data_path, device, optimizer, objective, aux_loss, config, epoch, mode, is_training=True, project_embeddings=False):
    losses = []
    pred_mean, pred_std = [], []
    compute_metrics = (epoch % config['compute_metrics_every'] == 0)
    if not project_embeddings:
        compute_metrics = False
    global_phasing_metrics, local_phasing_metrics = [], []
    
    phase = "Training" if is_training else "Validating"
    
    for idx, graph_name in enumerate(data_selection):
        print(f"{phase} graph {graph_name}, id: {idx} of {len(data_selection)}")
        g = torch.load(os.path.join(data_path, graph_name + '.pt'), map_location=device)
        
        x_addon = torch.abs(g.y).float().unsqueeze(1).to(device)
        g.x = torch.cat([g.x, x_addon], dim=1)
        #g.x = torch.zeros_like(torch.abs(g.y).float().unsqueeze(1).to(device))
        if not hasattr(g, 'chr'):
            g.chr = torch.ones(g.x.shape[0], dtype=torch.long, device=g.x.device)
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
            
            loss = objective(g.y, predictions, g.edge_index[0], g.edge_index[1], g.chr)
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
    
    prefix = "valid_" if not is_training else "train_"
    metrics = {
        f'{prefix}loss': np.mean(losses),
        f'{prefix}mean': np.mean(pred_mean) if pred_mean else 0,
        f'{prefix}std': np.mean(pred_std) if pred_std else 0
    }
    
    if compute_metrics:
        metrics.update({
            f'{prefix}global_phasing': np.mean(global_phasing_metrics),
            f'{prefix}local_phasing': np.mean(local_phasing_metrics)
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
        aux_loss = SharedToZeroLoss(weight=config['aux_loss_weight']).to(device)
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

            if (epoch+1) % 10 == 0:
                model_path = os.path.join(save_model_path, f'{args.run_name}_{epoch+1}.pt')
                torch.save(model.state_dict(), model_path)
                print("Saved model")

def get_graph_data(data_path, data_selection):
    """
    Analyze graph data and display statistics about nodes and edges.
    
    Args:
        data_path: Path to the directory containing graph data
        data_selection: List of graph names to analyze
    
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
    
    for graph_name in data_selection:
        g = torch.load(os.path.join(data_path, graph_name + '.pt'))
        
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
            layer_norm=config['layer_norm']
        ).to(device)
    else:
        model = SGFormer(in_channels=config['node_features'], hidden_channels=config['hidden_features'], out_channels=1, trans_num_layers=config['num_trans_layers'], gnn_num_layers=config['num_gnn_layers'], gnn_dropout=config['gnn_dropout'], layer_norm=config['layer_norm']).to(device)
    #latest change: half hidden channels, reduce gnn_layers, remove dropout
    to_undirected = False
    model.to(device)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint, map_location=device))
        print(f"Loaded model from {load_checkpoint}")

    # Get dataset statistics before training
    print("Analyzing training data:")
    train_stats = get_graph_data(data_path, train_data)
    if valid_data:
        print("\nAnalyzing validation data:")
        valid_stats = get_graph_data(data_path, valid_data)

    train(model, data_path, train_data, valid_data, device, mode, project_embeddings, config)

    #python train_yak2.py --save_model_path /mnt/sod2-project/csb4/wgs/martin/rl_dgl_datasets/trained_models/ --data_path /mnt/sod2-project/csb4/wgs/martin/diploid_datasets/diploid_dataset_hg002_cent/pyg_graphs/ --data_config dataset_diploid_cent.yml --hyper_config config_yak.yml --wandb pred_yak --run_name yak2_SGformer_base --device cuda:4 --diploid
