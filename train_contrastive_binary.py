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
# Disable
#from torch_geometric.utils import to_undirected
from SGformer_HGC import SGFormer
from torch_geometric.utils import degree
import torch_sparse

class SwitchLoss(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0):
        super(SwitchLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def forward(self, y_true, y_pred, src, dst, g, chr, multi=True):
        if multi:
            return self.multi_chr_forward(y_true, y_pred, src, dst, g, chr)
        else:
            return self.single_chr_forward(y_true, y_pred, src, dst, g, chr)

    def single_chr_forward(self, y_true, y_pred, src, dst, g, chr):
        # Filter edges of type 0 (overlap edges)
        edge_type_mask = g.edge_type == 0
        filtered_edge_index = g.edge_index[:, edge_type_mask]
        
        if filtered_edge_index.size(1) == 0:
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)
            
        # Sample random edges from the filtered edges
        n = g.num_nodes
        num_edges = filtered_edge_index.size(1)
        edge_ids = torch.randint(0, num_edges, (n,), device=y_true.device)
        
        src = filtered_edge_index[0][edge_ids]
        dst = filtered_edge_index[1][edge_ids]

        # Get labels and predictions for each pair
        y_true_i = y_true[src]
        y_true_j = y_true[dst]
        y_pred_i = y_pred[src]
        y_pred_j = y_pred[dst]

        # Calculate the indicators
        indicator_same_label = (y_true_i == y_true_j).float()
        indicator_diff_label = (y_true_i != y_true_j).float()
        indicator_label_zero = (y_true == 0).float()

        # Calculate the margin
        margin = torch.abs(y_true_i - y_true_j)

        # Calculate the loss terms
        term_same_label = indicator_same_label * (y_pred_i - y_pred_j) ** 2
        term_diff_label = indicator_diff_label * torch.max(torch.zeros_like(margin),
                                                       margin - torch.abs(y_pred_i - y_pred_j)) ** 2 * 10
        term_label_zero = indicator_label_zero * (y_pred ** 2)

        # Combine the terms with the weights
        loss = (self.lambda_1 * term_same_label.mean() +
                self.lambda_2 * term_diff_label.mean() +
                self.lambda_3 * term_label_zero.mean())

        return loss

    def multi_chr_forward(self, y_true, y_pred, src, dst, g, chr):
        # Get unique chromosomes
        unique_chr = torch.unique(chr)
        
        all_src = []
        all_dst = []
        
        # Filter edges of type 0 (overlap edges)
        edge_type_mask = g.edge_type == 0
        filtered_edge_index = g.edge_index[:, edge_type_mask]
        
        # For each chromosome, sample edges between nodes of the same chromosome
        for c in unique_chr:
            # Create mask for current chromosome
            chr_mask = (chr == c)
            chr_nodes = torch.where(chr_mask)[0]
            
            if len(chr_nodes) < 2:
                continue
                
            # Get edges where both nodes are in the current chromosome
            edge_mask = chr[filtered_edge_index[0]] == c
            edge_mask &= chr[filtered_edge_index[1]] == c
            chr_edges = filtered_edge_index[:, edge_mask]
            
            if chr_edges.size(1) == 0:
                continue
                
            # Sample random edges for this chromosome
            n_samples = len(chr_nodes)
            edge_ids = torch.randint(0, chr_edges.size(1), (n_samples,), device=y_true.device)
            
            all_src.append(chr_edges[0][edge_ids])
            all_dst.append(chr_edges[1][edge_ids])

        if not all_src:
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)

        # Concatenate all sampled edges
        src = torch.cat(all_src)
        dst = torch.cat(all_dst)

        # Get labels and predictions for each pair
        y_true_i = y_true[src]
        y_true_j = y_true[dst]
        y_pred_i = y_pred[src]
        y_pred_j = y_pred[dst]

        # Calculate the indicators
        indicator_same_label = (y_true_i == y_true_j).float()
        indicator_diff_label = (y_true_i != y_true_j).float()
        indicator_label_zero = (y_true == 0).float()

        # Calculate the margin
        margin = torch.abs(y_true_i - y_true_j)

        # Calculate the loss terms
        term_same_label = indicator_same_label * (y_pred_i - y_pred_j) ** 2
        term_diff_label = indicator_diff_label * torch.max(torch.zeros_like(margin),
                                                       margin - torch.abs(y_pred_i - y_pred_j)) ** 2 * 10
        term_label_zero = indicator_label_zero * (y_pred ** 2)

        # Calculate the final loss by averaging across all samples
        loss = (self.lambda_1 * term_same_label.mean() +
                self.lambda_2 * term_diff_label.mean() +
                self.lambda_3 * term_label_zero.mean())

        return loss

class HammingLoss(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0):
        super(HammingLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def forward(self, y_true, y_pred, src, dst, chr, multi=True):
        if multi:
            return self.multi_chr_forward(y_true, y_pred, src, dst, chr)
        else:
            return self.single_chr_forward(y_true, y_pred, src, dst, chr)

    def single_chr_forward(self, y_true, y_pred, src, dst, chr):
        nodes = list(range(len(y_true)))
        shuffled_nodes = nodes.copy()
        random.shuffle(shuffled_nodes)

        src = torch.tensor(nodes, device=y_true.device)
        dst = torch.tensor(shuffled_nodes, device=y_true.device)

        # Get labels and predictions for each pair
        y_true_i = y_true[src].to(device)
        y_true_j = y_true[dst].to(device)
        y_pred_i = y_pred[src].to(device)
        y_pred_j = y_pred[dst].to(device)

        # Clip predictions to prevent extreme values
        #y_pred_i = torch.clamp(y_pred_i, min=-10, max=10)
        #y_pred_j = torch.clamp(y_pred_j, min=-10, max=10)

        # Calculate the indicators
        indicator_same_label = (y_true_i == y_true_j).float()
        indicator_diff_label = (y_true_i != y_true_j).float()
        indicator_label_zero = (y_true == 0).float()

        # Calculate the margin
        margin = torch.abs(y_true_i - y_true_j)

        # Calculate the loss terms
        term_same_label = indicator_same_label * (y_pred_i - y_pred_j) ** 2
        term_diff_label = indicator_diff_label * torch.max(torch.zeros_like(margin),
                                                       margin - torch.abs(y_pred_i - y_pred_j)) ** 2
        term_label_zero = indicator_label_zero * (y_pred ** 2)

        # Combine the terms with the weights
        loss = (self.lambda_1 * term_same_label.mean() +
                self.lambda_2 * term_diff_label.mean() +
                self.lambda_3 * term_label_zero.mean())

        return loss

    def multi_chr_forward(self, y_true, y_pred, src, dst, chr):
        # Get unique chromosomes and their counts
        unique_chr, chr_counts = torch.unique(chr, return_counts=True)
        
        # Filter out chromosomes with less than 2 nodes
        valid_mask = chr_counts >= 2
        unique_chr = unique_chr[valid_mask]
        
        if len(unique_chr) == 0:
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)

        # Create a mask tensor for all chromosomes at once
        chr_masks = chr.unsqueeze(0) == unique_chr.unsqueeze(1)  # Broadcasting
        
        # Initialize tensors to store results for all chromosomes
        all_true_i = []
        all_true_j = []
        all_pred_i = []
        all_pred_j = []
        all_zero_indicators = []
        
        for idx, mask in enumerate(chr_masks):
            # Get nodes for current chromosome
            chr_nodes = torch.where(mask)[0]
            
            # Create shuffled pairs within chromosome
            shuffled_indices = torch.randperm(len(chr_nodes))
            
            # Gather the values using the masks
            all_true_i.append(y_true[chr_nodes])
            all_true_j.append(y_true[chr_nodes[shuffled_indices]])
            all_pred_i.append(y_pred[chr_nodes])
            all_pred_j.append(y_pred[chr_nodes[shuffled_indices]])
            all_zero_indicators.append(y_true[mask] == 0)

        # Stack all tensors
        y_true_i = torch.cat(all_true_i)
        y_true_j = torch.cat(all_true_j)
        y_pred_i = torch.cat(all_pred_i)
        y_pred_j = torch.cat(all_pred_j)
        indicator_label_zero = torch.cat(all_zero_indicators).float()

        # Calculate indicators for all pairs at once
        indicator_same_label = (y_true_i == y_true_j).float()
        indicator_diff_label = (y_true_i != y_true_j).float()

        # Calculate margin for all pairs at once
        margin = torch.abs(y_true_i - y_true_j)

        # Calculate loss terms for all pairs at once
        term_same_label = indicator_same_label * (y_pred_i - y_pred_j) ** 2
        term_diff_label = indicator_diff_label * torch.max(torch.zeros_like(margin),
                                                   margin - torch.abs(y_pred_i - y_pred_j)) ** 2
        term_label_zero = indicator_label_zero * torch.cat([y_pred[mask] for mask in chr_masks]) ** 2

        # Calculate the final loss by averaging across all samples
        loss = (self.lambda_1 * term_same_label.mean() +
                self.lambda_2 * term_diff_label.mean() +
                self.lambda_3 * term_label_zero.mean())

        return loss
    
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


def train(model, data_path, train_selection, valid_selection, device, config):
    best_valid_loss = 10000
    overfit = not bool(valid_selection)

    hamming_loss = HammingLoss().to(device)
    switch_loss = SwitchLoss().to(device)

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
            train_metrics = train_epoch(model, train_selection, data_path, device, optimizer, 
                                      hamming_loss, switch_loss, config, epoch)

            # Validation phase (if applicable)
            valid_metrics = {}
            if not overfit:
                print('===> VALIDATION')
                model.eval()
                valid_metrics = validate_epoch(model, valid_selection, data_path, device,
                                            hamming_loss, switch_loss, config, epoch)
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

def train_epoch(model, train_selection, data_path, device, optimizer, 
                hamming_loss, switch_loss, config, epoch):
    train_loss = []
    pred_mean, pred_std = [], []
    compute_metrics = (epoch % config['compute_metrics_every'] == 0)
    global_phasing_train, local_phasing_train = [], []

    for idx, graph_name in enumerate(train_selection):
        print(f"Training graph {graph_name}, id: {idx} of {len(train_selection)}")
        g = torch.load(os.path.join(data_path, graph_name + '.pt'), map_location=device)

        optimizer.zero_grad()
        #g.x = torch.cat([g.x.to(device), g.pe_0.to(device), g.pe_1.to(device)], dim=1)    
        g.x = torch.abs(g.y).float().unsqueeze(1).to(device) #torch.cat([torch.abs(g.y).float(), torch.abs(g.y).float()], dim=1)

        predictions = model(g).squeeze()
        
        pred_mean.append(predictions.mean().item())
        pred_std.append(predictions.std().item())

        loss_1 = hamming_loss(g.y, predictions, g.edge_index[0], g.edge_index[1], g.chr)
        if config['alpha'] > 0:
            loss_2 = switch_loss(g.y, predictions, g.edge_index[0], g.edge_index[1], g, g.chr)
            loss = loss_1 + config['alpha'] * loss_2
        else:
            loss = loss_1
            
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        if compute_metrics:
            global_phasing_train.append(global_phasing_quotient(g.y, predictions, g.chr))
            local_phasing_train.append(local_phasing_quotient(g.y, predictions, g.chr, 
                                                            g.edge_index, g.edge_type))

    metrics = {
        'train_loss': np.mean(train_loss),
        'mean': np.mean(pred_mean),
        'std': np.mean(pred_std)
    }
    
    if compute_metrics:
        metrics.update({
            'train_global_phasing': np.mean(global_phasing_train),
            'train_local_phasing': np.mean(local_phasing_train)
        })
    
    return metrics

def validate_epoch(model, valid_selection, data_path, device, 
                  hamming_loss, switch_loss, config, epoch):
    valid_loss = []
    valid_pred_mean, valid_pred_std = [], []
    compute_metrics = (epoch % config['compute_metrics_every'] == 0)
    valid_global_phasing, valid_local_phasing = [], []

    with torch.no_grad():
        for idx, graph_name in enumerate(valid_selection):
            print(f"Validating graph {graph_name}, id: {idx} of {len(valid_selection)}")
            g = torch.load(os.path.join(data_path, graph_name + '.pt'), map_location=device)
            g.x = torch.abs(g.y).float().unsqueeze(1).to(device) #torch.cat([torch.abs(g.y).float(), torch.abs(g.y).float()], dim=1)

            predictions = model(g).squeeze()
            valid_pred_mean.append(predictions.mean().item())
            valid_pred_std.append(predictions.std().item())

            loss_1 = hamming_loss(g.y, predictions, g.edge_index[0], g.edge_index[1], g.chr)
            if config['alpha'] > 0:
                loss_2 = switch_loss(g.y, predictions, g.edge_index[0], g.edge_index[1], g, g.chr)
                loss = loss_1 + config['alpha'] * loss_2
            else:
                loss = loss_1
            valid_loss.append(loss.item())

            if compute_metrics:
                valid_global_phasing.append(global_phasing_quotient(g.y, predictions, g.chr, debug=True))
                valid_local_phasing.append(local_phasing_quotient(g.y, predictions, g.chr, 
                                                                g.edge_index, g.edge_type, debug=True))

    metrics = {
        'valid_loss': np.mean(valid_loss),
        'valid_mean': np.mean(valid_pred_mean),
        'valid_std': np.mean(valid_pred_std)
    }
    
    if compute_metrics:
        metrics.update({
            'valid_global_phasing': np.mean(valid_global_phasing),
            'valid_local_phasing': np.mean(valid_local_phasing)
        })
    
    return metrics

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
   
    model = SGFormer(in_channels=config['node_features'], hidden_channels=config['hidden_features'], out_channels=1, trans_num_layers=config['num_trans_layers'], gnn_num_layers=config['num_gnn_layers'], gnn_dropout=config['gnn_dropout'], layer_norm=config['layer_norm']).to(device)

    #latest change: half hidden channels, reduce gnn_layers, remove dropout
    to_undirected = False
    model.to(device)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint, map_location=device))
        print(f"Loaded model from {load_checkpoint}")

    train(model, data_path, train_data, valid_data, device, config)

    #python train_yak2.py --save_model_path /mnt/sod2-project/csb4/wgs/martin/rl_dgl_datasets/trained_models/ --data_path /mnt/sod2-project/csb4/wgs/martin/diploid_datasets/diploid_dataset_hg002_cent/pyg_graphs/ --data_config dataset_diploid_cent.yml --hyper_config config_yak.yml --wandb pred_yak --run_name yak2_SGformer_base --device cuda:4 --diploid
