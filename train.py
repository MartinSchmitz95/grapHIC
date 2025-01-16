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
stds_and_means = utils.STDS_AND_MEANS
# Disable
#from torch_geometric.utils import to_undirected

from SGformer import SGFormer
from torch_geometric.utils import degree
import torch_sparse


class YakLossLocal(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0):
        super(YakLossLocal, self).__init__()
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

class YakLossGlobal(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0):
        super(YakLossGlobal, self).__init__()
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

    if combination1 == 0 and combination2 == 0: #no idea when s=this should be the case
        return 0.0
    # Return the best combination
    metric = max(combination1, combination2)/(combination1 + combination2)
    return metric

def prepare_subgraph_virtual(g, sub_g, x_attr, device):

    # Assume you have an existing DGL graph `g`
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()
    g.edata['glob'] = torch.zeros(num_edges, 1).to(device)
    d = {'glob': torch.ones(num_nodes, 1).to(device),
         'overlap_similarity': torch.zeros(num_nodes, 1).to(device),
         'overlap_length': torch.zeros(num_nodes, 1).to(device)}
    # Add the global node
    src = torch.tensor([num_nodes] * num_nodes).to(device)  # global node index is num_nodes (last node)
    dst = torch.arange(num_nodes).to(device)
    sub_g.add_edges(src, dst, data=d)
    sub_g.add_nodes(1, {'gt_hap': torch.tensor([0]).to(device)})

    sub_g = sub_g.to(device)

    src_, dst_ = g.edges()
    src = src_[sub_g.edata['_ID']]
    dst = dst_[sub_g.edata['_ID']]

    ol_len = g.edata['overlap_length'][sub_g.edata['_ID']].float()
    ol_len = (ol_len - stds_and_means["ol_len_mean"]) / stds_and_means["ol_len_std"]
    ol_sim = g.edata['overlap_similarity'][sub_g.edata['_ID']]
    e = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)

    pe_in = sub_g.in_degrees().float().unsqueeze(1)
    pe_out = sub_g.out_degrees().float().unsqueeze(1)
    pe_in = ((pe_in - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(device)
    pe_out = ((pe_out - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(device)

    if x_attr == 'h':
        print("Hap h Not implemented")
        exit()
    elif x_attr == 'm':
        y = g.ndata['gt_hap'][sub_g.ndata['_ID']].to(device)
    elif x_attr == 'p':
        y = - g.ndata['gt_hap'][sub_g.ndata['_ID']].to(device)

    else:
        raise ValueError(f'Unknown x_attr: {x_attr}')

    pe = torch.cat((pe_in, pe_out), dim=1).to(device)
    return sub_g, src, dst, y, e, pe
def prepare_subgraph(g, sub_g, x_attr, device):
    sub_g = sub_g.to(device)

    src_, dst_ = g.edges()
    src = src_[sub_g.edata['_ID']]
    dst = dst_[sub_g.edata['_ID']]

    ol_len = g.edata['overlap_length'][sub_g.edata['_ID']].float()
    ol_len = (ol_len - stds_and_means["ol_len_mean"]) / stds_and_means["ol_len_std"]
    ol_sim = g.edata['overlap_similarity'][sub_g.edata['_ID']]
    e = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)

    pe_in = sub_g.in_degrees().float().unsqueeze(1)
    pe_out = sub_g.out_degrees().float().unsqueeze(1)
    pe_in = ((pe_in - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(device)
    pe_out = ((pe_out - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(device)

    if x_attr == 'h':
        print("Hap h Not implemented")
        exit()
    elif x_attr == 'm':
        y = g.ndata['gt_hap'][sub_g.ndata['_ID']].to(device)
    elif x_attr == 'p':
        y = - g.ndata['gt_hap'][sub_g.ndata['_ID']].to(device)

    else:
        raise ValueError(f'Unknown x_attr: {x_attr}')

    pe = torch.cat((pe_in, pe_out), dim=1).to(device)
    return sub_g, src, dst, y, e, pe

def prepare_graph_virtual(g, x_attr, device):

    # Assume you have an existing DGL graph `g`
    g = g.to('cpu')
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()
    df = {'globef': torch.ones(num_nodes, dtype=torch.float32)}
    db = {'globeb': torch.ones(num_nodes, dtype=torch.float32)}
    nf = {'globn': torch.ones(1, dtype=torch.float32)}

    # Add the global node
    g.add_nodes(1, nf)
    src = torch.tensor([num_nodes] * num_nodes)  # global node index is num_nodes (last node)
    dst = torch.arange(num_nodes)
    g.add_edges(src, dst, data=df)
    g.add_edges(dst, src, data=db)


    g = g.to(device)
    src, dst = g.edges()
    ol_len = g.edata['overlap_length'].float()
    ol_len = (ol_len - stds_and_means["ol_len_mean"]) / stds_and_means["ol_len_std"]
    ol_sim = g.edata['overlap_similarity']
    e = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1), g.edata['globef'].unsqueeze(-1), g.edata['globeb'].unsqueeze(-1)), dim=1)

    pe_in = g.in_degrees().float().unsqueeze(1)
    pe_out = g.out_degrees().float().unsqueeze(1)
    pe_in = ((pe_in - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(device)
    pe_out = ((pe_out - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(device)

    if x_attr == 'h':
        print("Hap h Not implemented")
        exit()
    elif x_attr == 'm':
        y = g.ndata['gt_hap'].to(device)
    elif x_attr == 'p':
        y = - g.ndata['gt_hap'].to(device)

    else:
        raise ValueError(f'Unknown x_attr: {x_attr}')
    pe = torch.cat((pe_in, pe_out, g.ndata['globn'].unsqueeze(-1)), dim=1).to(device)

    return g, src, dst, y, e, pe
def prepare_graph(g, x_attr, device):
    g = g.to(device)
    src, dst = g.edges()
    ol_len = g.edata['overlap_length'].float()
    ol_len = (ol_len - stds_and_means["ol_len_mean"]) / stds_and_means["ol_len_std"]
    ol_sim = g.edata['overlap_similarity']
    e = torch.cat((ol_len.unsqueeze(-1), ol_sim.unsqueeze(-1)), dim=1)

    pe_in = g.in_degrees().float().unsqueeze(1)
    pe_out = g.out_degrees().float().unsqueeze(1)
    pe_in = ((pe_in - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(device)
    pe_out = ((pe_out - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(device)

    if x_attr == 'h':
        print("Hap h Not implemented")
        exit()
    elif x_attr == 'm':
        y = g.ndata['gt_hap'].to(device)
    elif x_attr == 'p':
        y = - g.ndata['gt_hap'].to(device)

    else:
        raise ValueError(f'Unknown x_attr: {x_attr}')
    pe = torch.cat((pe_in, pe_out), dim=1).to(device)

    return g, src, dst, y, e, pe

def k_step_pagerank(edge_index, num_nodes, k=10, alpha=0.85):
    pagerank = torch.zeros((num_nodes, k))

    # Compute degree matrix
    row, col = edge_index.to('cpu')
    deg = degree(row, num_nodes, dtype=torch.float)

    # Avoid division by zero
    deg[deg == 0] = 1.0
    D_inv = torch.diag(1.0 / deg)

    # Compute the transition matrix
    A = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    M = torch.matmul(D_inv, A.to_dense())

    # PageRank iterations with a small random noise
    rank = torch.rand((num_nodes, 1)) * 0.01 + 1.0  # Small noise + initial rank
    for i in range(k):
        rank = alpha * torch.matmul(M, rank) + (1 - alpha) / num_nodes  # Damping factor + teleportation
        pagerank[:, i] = rank.squeeze()

    return pagerank.to(device)


def train(model, data_path, train_selection, valid_selection, device, config, diploid=False, symmetry=False, aux=False, quick=False, dr_loss=True):
    best_valid_loss = 10000
    overfit = not bool(valid_selection)

    global_loss = YakLossGlobal().to(device)
    #local_loss = YakLossLocal().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['decay'], patience=config['patience'], verbose=True)

    time_start = datetime.now()
    x_attrs = ['m', 'p']

    with wandb.init(project=wandb_project, config=config, mode=config['wandb_mode'], name=run_name):
        # Comment out the wandb.watch() call
        # wandb.watch(model, global_loss, log='all', log_freq=1000)

        best_decision_quotient, min_loss = 0, 100000
        for epoch in range(config['num_epochs']):
            print('===> TRAINING')
            model.train()
            random.shuffle(train_selection)
            train_loss, train_rank_loss, train_symm_loss, train_aux_loss, train_stop_loss = [], [], [], [], []
            pred_mean, pred_std, yak_frac_train,  = [], [], []
            train_all_predictions, train_pos_predictions, train_neg_predictions = [], [], []

            for idx, graph_name in enumerate(train_selection):
                g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
                sub_gs = [g]
                """num_clusters = utils.get_num_nodes_per_cluster(g, config)
                g = g.long()
                full_graph = bool(num_clusters <= 1)
                if not full_graph:
                    d = dgl.metis_partition(g, num_clusters, extra_cached_hops=config['k_extra_hops'])
                    sub_gs = list(d.values())
                    random.shuffle(sub_gs)
                else:
                    sub_gs = [g]"""
                g = g.to(device)

                # For loop over all mini-batch in the graph
                # print("Amount of subgraphs: ", len(sub_gs))
                for sub_g in sub_gs:
                    for x_attr in x_attrs:

                        #if full_graph:
                        #    sub_g, src, dst, y, e, x = prepare_graph(g, x_attr, device)
                        #else:
                        #    sub_g, src, dst, y, e, x = prepare_subgraph(g, sub_g, x_attr, device)

                        edge_index = g.edge_index

                        num_nodes = g.num_nodes
                        # Compute out-degree (nuber of outgoing edges for each node)
                        out_degrees = degree(edge_index[0], num_nodes=num_nodes)
                        # Compute in-degree (number of incoming edges for each node)
                        in_degrees = degree(edge_index[1], num_nodes=num_nodes)
                        pe_in = ((in_degrees - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(device)
                        pe_out = ((out_degrees - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(device)
                        # Stack the in-degree and out-degree to create a node feature tensor
                        #x = torch.stack([pe_in, pe_out], dim=1)

                        #k_step_features = k_step_pagerank(edge_index, num_nodes, k=8)
                        #k_step_features = k_step_features[:, -1].unsqueeze(1)  # Shape [num_nodes, 1]

                        #print(k_step_features)
                        #pos_enc_features = [g[f'pos_enc_{n}'] for n in range(64)]
                        #pos_enc_tensor = torch.stack(pos_enc_features, dim=1)
                        #x = torch.cat((pe_in.unsqueeze(1), pe_out.unsqueeze(1), pos_enc_tensor), dim=1).to(device)
                        pog_median = g.pog_median.to(device, dtype=torch.float32)
                        pog_min = g.pog_min.to(device, dtype=torch.float32)
                        pog_max = g.pog_max.to(device, dtype=torch.float32)
                        x = torch.cat((pe_in.unsqueeze(1), pe_out.unsqueeze(1), pog_median.unsqueeze(1), pog_min.unsqueeze(1), pog_max.unsqueeze(1)), dim=1).to(device)

                        #x = torch.cat([pe_in.unsqueeze(1) , pe_out.unsqueeze(1)], dim=1)
                        #x = g.x_ftrs
                        if to_undirected:
                            reversed_edge_index = edge_index.flip([0])
                            undirected_edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)
                            # Remove duplicate edges by sorting and using unique function
                            edge_index = torch.unique(undirected_edge_index, dim=1)


                        yak_predictions = model(x, edge_index).squeeze()
                        pred_mean.append(yak_predictions.mean().item())
                        pred_std.append(yak_predictions.std().item())

                        # Assume you have a PyG graph 'g'
                        edge_index = g.edge_index  # edge_index is a tensor of shape [2, num_edges]
                        # src and dst correspond to the rows in edge_index
                        src = edge_index[0]  # Source nodes
                        dst = edge_index[1]  # Destination nodes
                        # Ensure src and dst are on the same device as y_true
                        src = src.to(device)
                        dst = dst.to(device)
                        y = g.gt_hap
                        optimizer.zero_grad()
                        loss_1 = global_loss(y, yak_predictions, src, dst, sub_g)
                        #loss_2 = local_loss(y, yak_predictions, src, dst, sub_g)
                        #print(loss_1, loss_2)
                        loss = loss_1 #+ loss_2
                        loss.backward()
                        train_loss.append(loss.item())
                        yak_frac_train.append(fraction_correct_yak(y, yak_predictions))
                        optimizer.step()

                        elapsed = datetime.now() - time_start

                        print(f'\nTRAINING (one training graph): Epoch = {epoch}, Graph = {idx}, Loss = {loss.item()}, Elapsed Time = {elapsed}')
            train_all_predictions = yak_predictions.cpu().detach()
            train_pos_predictions = yak_predictions[y == 1].cpu().detach()
            train_neg_predictions = yak_predictions[y == -1].cpu().detach()

            if not overfit:
                print('===> VALIDATION')
                model.eval()
                random.shuffle(valid_selection)
                valid_loss, yak_frac_valid, valid_all_predictions, valid_pos_predictions, valid_neg_predictions = [], [], [], [], []

                for idx, graph_name in enumerate(valid_selection):
                    time_graph_start = datetime.now()
                    g = torch.load(os.path.join(data_path, graph_name + '.pt')).to(device)
                    #num_clusters = utils.get_num_nodes_per_cluster(g, config)

                    #print(f'Num clusters:', num_clusters)
                    #g = g.long()
                    sub_gs = [g]
                    """if num_clusters > 1:
                        d = dgl.metis_partition(g, num_clusters, extra_cached_hops=config['k_extra_hops'])
                        sub_gs = list(d.values())
                        random.shuffle(sub_gs)
                        full_graph = False
                    else:
                        sub_gs = [g]
                        full_graph = True"""
                    g = g.to(device)
                    # For loop over all mini-batch in the graph

                    for sub_g in sub_gs:
                        for x_attr in x_attrs:
                            edge_index = g.edge_index
                            num_nodes = g.num_nodes
                            # Compute out-degree (nuber of outgoing edges for each node)
                            out_degrees = degree(edge_index[0], num_nodes=num_nodes)
                            # Compute in-degree (number of incoming edges for each node)
                            in_degrees = degree(edge_index[1], num_nodes=num_nodes)
                            pe_in = ((in_degrees - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(
                                device)
                            pe_out = ((out_degrees - stds_and_means["degree_mean"]) / stds_and_means["degree_std"]).to(
                                device)
                            # Stack the in-degree and out-degree to create a node feature tensor
                            x = torch.cat([pe_in.unsqueeze(1) , pe_out.unsqueeze(1)], dim=1)
                            #pos_enc_features = [g[f'pos_enc_{n}'] for n in range(64)]
                            #pos_enc_tensor = torch.stack(pos_enc_features, dim=1)
                            #x = torch.cat((pe_in.unsqueeze(1), pe_out.unsqueeze(1), pos_enc_tensor), dim=1).to(device)
                            pog_median = g.pog_median.to(device, dtype=torch.float32)
                            pog_min = g.pog_min.to(device, dtype=torch.float32)
                            pog_max = g.pog_max.to(device, dtype=torch.float32)
                            x = torch.cat((pe_in.unsqueeze(1), pe_out.unsqueeze(1), pog_median.unsqueeze(1), pog_min.unsqueeze(1), pog_max.unsqueeze(1)), dim=1).to(device)

                            
                            #x = g.x_ftrs

                            if to_undirected:
                                reversed_edge_index = edge_index.flip([0])
                                undirected_edge_index = torch.cat([edge_index, reversed_edge_index], dim=1)
                                # Remove duplicate edges by sorting and using unique function
                                edge_index = torch.unique(undirected_edge_index, dim=1)

                            yak_predictions = model(x, edge_index).squeeze()
                            pred_mean.append(yak_predictions.mean().item())
                            pred_std.append(yak_predictions.std().item())

                            # Assume you have a PyG graph 'g'
                            edge_index = g.edge_index  # edge_index is a tensor of shape [2, num_edges]
                            # src and dst correspond to the rows in edge_index
                            src = edge_index[0]  # Source nodes
                            dst = edge_index[1]  # Destination nodes
                            # Ensure src and dst are on the same device as y_true
                            src = src.to(device)
                            dst = dst.to(device)
                            y = g.gt_hap
                            optimizer.zero_grad()
                            loss_1 = global_loss(y, yak_predictions, src, dst, sub_g)
                            #loss_2 = local_loss(y, yak_predictions, src, dst, sub_g)
                            loss = loss_1 #+ loss_2
                            valid_loss.append(loss.item())
                            yak_frac_valid.append(fraction_correct_yak(y, yak_predictions))


                            elapsed = datetime.now() - time_start
                            print(f'\nVALIDATION (one valid graph): Epoch = {epoch}, Graph = {idx}')
                #valid_all_predictions = yak_predictions.cpu().detach()
                #valid_pos_predictions = yak_predictions[y == 1].cpu().detach()
                #valid_neg_predictions = yak_predictions[y == -1].cpu().detach()

            print(f'Completed Epoch = {epoch}, elapsed time: {elapsed}\n\n')
            train_loss = np.mean(train_loss)
            yak_frac_train = np.mean(yak_frac_train)


            #train_data = [[s] for s in train_all_predictions.numpy()]
            #train_data_pos = [[s] for s in train_pos_predictions.numpy()]
            #train_data_neg = [[s] for s in train_neg_predictions.numpy()]

            #train_preds_all = wandb.plot.histogram(wandb.Table(data=train_data, columns=["pred"]), "scores", title="Train All")
            #train_preds_pos = wandb.plot.histogram(wandb.Table(data=train_data_pos, columns=["pred"]), "scores", title="Train Paternal")
            #train_preds_neg = wandb.plot.histogram(wandb.Table(data=train_data_neg, columns=["pred"]), "scores", title="Train Maternal")

            if not overfit:
                #valid_data = [[s.item()] for s in valid_all_predictions.numpy()]
                #valid_data_pos = [[s.item()] for s in valid_pos_predictions.numpy()]
                #valid_data_neg = [[s.item()] for s in valid_neg_predictions.numpy()]

                #valid_preds_all = wandb.plot.histogram(wandb.Table(data=valid_data, columns=["pred"]), "scores",
                #                                       title="Valid All")
                #valid_preds_pos = wandb.plot.histogram(wandb.Table(data=valid_data_pos, columns=["pred"]), "scores",
                #                                       title="Valid Paternal")
                #valid_preds_neg = wandb.plot.histogram(wandb.Table(data=valid_data_neg, columns=["pred"]), "scores",
                #                                       title="Valid Maternal")

                valid_loss = np.mean(valid_loss)
                yak_frac_valid = np.mean(yak_frac_valid)
                scheduler.step(valid_loss)
                wandb.log({'train_loss': train_loss, 'valid_loss': valid_loss, 'train_yak_frac': yak_frac_train, 'valid_yak_frac': yak_frac_valid, 'mean': np.mean(pred_mean), 'std': np.mean(pred_std)})
                #,
                #               'std': np.mean(pred_std),'train_preds_hist': train_preds_all, 'train_pos_preds_hist': train_preds_pos, 'train_neg_preds_hist': train_preds_neg,
                #           'valid_preds_hist': valid_preds_all, 'valid_pos_preds_hist': valid_preds_pos, 'valid_neg_preds_hist': valid_preds_neg})
            
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
    parser.add_argument("--data_config", type=str, default='data_debug.yml', help="dataset path")
    parser.add_argument("--hyper_config", type=str, default='configs/config_yak.yml', help="dataset path")
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
    """
    model = YakGATModel(
        config['node_features'],
        config['edge_features'],
        config['hidden_features'],
        config['hidden_edge_features'],
        config['num_gnn_layers'],
        config['hidden_edge_scores'],
        config['dropout'],
    )
    """
    """model = SymGatedGCNYakModel(
        config['node_features'],
        config['edge_features'],
        config['hidden_features'],
        config['hidden_edge_features'],
        config['num_gnn_layers'],
        config['hidden_edge_scores'],
        config['batch_norm'],
        config['nb_pos_enc'],
        config['nr_classes'],
        dropout=config['dropout'],
        stop_head=config['stop_head'],
        pred_dropout=config['pred_dropout']
    )"""
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
