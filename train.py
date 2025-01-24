import torch
import torch.nn as nn
import wandb
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
#import dgl
import os
from datetime import datetime
import numpy as np
import utils
import argparse
import yaml
#from SymGatedGCN_DGL import SymGatedGCNYakModel
#from YakModel import YakGATModel
# Disable
#from torch_geometric.utils import to_undirected

from SGformer import SGFormer
from torch_geometric.utils import degree
import torch_sparse

class SwitchLoss(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0):
        super(SwitchLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def forward(self, y_true, y_pred, src, dst, g):

        """nodes = list(range(len(y_true)))
        shuffled_nodes = nodes.copy()
        random.shuffle(shuffled_nodes)

        src = torch.tensor(nodes, device=y_true.device)
        dst = torch.tensor(shuffled_nodes, device=y_true.device)"""

        # Initialize a list to store pairs of nodes and their neighbors
        """nodes = list(range(len(y_true)))
        src = torch.tensor(nodes, device=y_true.device)
        dst = torch.tensor(nodes, device=y_true.device)

        # Iterate over all nodes and get their neighbors

        for i, node in enumerate(nodes):
            neighbors = list(g.successors(node))
            if neighbors:
                dst[i] = random.choice(neighbors)
            else:
                dst[i] = node"""

        # Assuming `g` is your PyG graph and `n` is the number of edges to sample
        n = g.num_nodes  # Example number of edges to sample

        # Sample `n` random edges from the graph
        num_edges = g.num_edges
        edge_ids = torch.randint(0, num_edges, (n,)).to(device)

        # Get the source and destination nodes of these edges
        src = g.edge_index[0][edge_ids]
        dst = g.edge_index[1][edge_ids]


        # Get labels and predictions for each pair
        y_true_i = y_true[src].to(device)
        y_true_j = y_true[dst].to(device)
        y_pred_i = y_pred[src].to(device)
        y_pred_j = y_pred[dst].to(device)

        #print(y_pred_i.shape, )
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

class HammingLoss(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0):
        super(HammingLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

    def forward(self, y_true, y_pred, src, dst, g):

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

        #print(y_pred_i.shape, )
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
    
def fraction_correct_yak(y_true, y_pred):

    device = y_true.device
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    #print(y_true)
    #print(f"y_true and y_pred {y_true.shape} {y_pred.shape}")

    def calculate_fractions(label):
        mask = (y_true == label)
        #print(mask)
        preds = y_pred[mask]
        if len(preds) > 0:
            sum_smaller_zero = (preds < 0).float().sum().item()
            sum_larger_zero = (preds >= 0).float().sum().item()
        else:
            sum_smaller_zero = 0.0
            sum_larger_zero = 0.0
        return sum_smaller_zero, sum_larger_zero

    # Calculate fractions for -1 and 1
    sum_neg1_smaller_zero, sum_neg1_larger_zero = calculate_fractions(-1)
    sum_pos1_smaller_zero, sum_pos1_larger_zero = calculate_fractions(1)

    print(f"fractions: {sum_neg1_smaller_zero} {sum_neg1_larger_zero} {sum_pos1_smaller_zero} {sum_pos1_larger_zero}")
    # Evaluate both combinations
    combination1 = (sum_neg1_smaller_zero + sum_pos1_larger_zero)
    combination2 = (sum_neg1_larger_zero + sum_pos1_smaller_zero)
    print(f"combinations: {combination1} {combination2}, total: {combination1 + combination2}")

    # Return the best combination
    metric = max(combination1, combination2)/(combination1 + combination2)
    return metric

def train(model, data_path, train_selection, valid_selection, device, config, diploid=False, symmetry=False, aux=False, quick=False, dr_loss=True):
    best_valid_loss = 10000
    overfit = not bool(valid_selection)

    hamming_loss = HammingLoss().to(device)
    #switch_loss = SwitchLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['decay'], patience=config['patience'], verbose=True)

    time_start = datetime.now()
    with wandb.init(project=wandb_project, config=config, mode=config['wandb_mode'], name=run_name):

        for epoch in range(config['num_epochs']):
            print('===> TRAINING')
            model.train()
            random.shuffle(train_selection)
            train_loss = []
            pred_mean, pred_std, yak_frac_train,  = [], [], []

            for idx, graph_name in enumerate(train_selection):
                g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
                g = g.to(device)

                yak_predictions = model(g.x, g.edge_index).squeeze()
                pred_mean.append(yak_predictions.mean().item())
                pred_std.append(yak_predictions.std().item())

                # Assume you have a PyG graph 'g'
                # src and dst correspond to the rows in edge_index
                src = g.edge_index[0]  # Source nodes
                dst = g.edge_index[1]  # Destination nodes
                y = g.y
                optimizer.zero_grad()
                loss_1 = hamming_loss(y, yak_predictions, src, dst, g)
                #loss_2 = switch_loss(y, yak_predictions, src, dst, g)
                #print(loss_1, loss_2)
                loss = loss_1 #+ loss_2
                loss.backward()
                train_loss.append(loss.item())
                yak_frac_train.append(fraction_correct_yak(y, yak_predictions))
                optimizer.step()

                elapsed = datetime.now() - time_start

                print(f'\nTRAINING (one training graph): Epoch = {epoch}, Graph = {idx}, Loss = {loss.item()}, Elapsed Time = {elapsed}')

            if not overfit:
                print('===> VALIDATION')
                

            print(f'Completed Epoch = {epoch}, elapsed time: {elapsed}\n\n')
            train_loss = np.mean(train_loss)
            yak_frac_train = np.mean(yak_frac_train)

            if not overfit:

                valid_loss = np.mean(valid_loss)
                yak_frac_valid = np.mean(yak_frac_valid)
                scheduler.step(valid_loss)
                wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss, 'train_yak_frac': yak_frac_train, 'valid_yak_frac': yak_frac_valid, 'mean': np.mean(pred_mean), 'std': np.mean(pred_std)})
            
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    model_path = os.path.join(save_model_path, f'{args.run_name}_best.pt')
                    torch.save(model.state_dict(), model_path)
                    print("Saved model")
            else:

                wandb.log({'train_loss': train_loss, 'train_yak_frac': yak_frac_train, 'mean': np.mean(pred_mean), 'std': np.mean(pred_std)})


            #torch.save(model.state_dict(), model_path)
            if (epoch+1) % 10 == 0:
                model_path = os.path.join(save_model_path, f'{args.run_name}_{epoch+1}.pt')
                torch.save(model.state_dict(), model_path)
                print("Saved model")

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

    full_dataset, valid_selection, train_selection = utils.create_dataset_dicts(data_config=data_config)
    valid_data = utils.get_numbered_graphs(valid_selection)
    train_data = utils.get_numbered_graphs(train_selection, starting_counts=valid_selection)

    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    with open(hyper_config) as file:
        config = yaml.safe_load(file)['training']

    utils.set_seed(args.seed)
   
    model = SGFormer(in_channels=config['node_features'], hidden_channels=config['hidden_features'], out_channels=1, trans_num_layers=config['num_trans_layers'], gnn_num_layers=config['num_gnn_layers'], gnn_dropout=config['gnn_dropout']).to(device)
    #model = MultiSGFormer(num_sgformers=3, in_channels=2, hidden_channels=128, out_channels=1, trans_num_layers=2, gnn_num_layers=4, gnn_dropout=0.0).to(device)

    #latest change: half hidden channels, reduce gnn_layers, remove dropout
    to_undirected = False
    model.to(device)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint, map_location=device))
        print(f"Loaded model from {load_checkpoint}")

    train(model, data_path, train_data, valid_data, device, config, diploid, symmetry, aux, quick = False, dr_loss = not args.bce)

    #python train_yak2.py --save_model_path /mnt/sod2-project/csb4/wgs/martin/rl_dgl_datasets/trained_models/ --data_path /mnt/sod2-project/csb4/wgs/martin/diploid_datasets/diploid_dataset_hg002_cent/pyg_graphs/ --data_config dataset_diploid_cent.yml --hyper_config config_yak.yml --wandb pred_yak --run_name yak2_SGformer_base --device cuda:4 --diploid
