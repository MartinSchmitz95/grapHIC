import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class ContrastiveEmbeddingLoss(nn.Module):
    
    """Adaptive clustering loss for node embeddings in a graph with multiple modes.
    
    This loss pulls nodes with the same label closer together in the embedding space
    while pushing nodes with different labels apart.
    
    Special handling for label 0: nodes with label 0 are considered to belong to both
    clusters (-1 and 1), so they form positive pairs with nodes from both clusters.
    
    Modes:
    - 'all_vs_all': Compare all nodes against all others (original implementation)
    - '1_vs_n': Standard InfoNCE with one positive and n negatives per anchor
    - 'n_vs_n': Supervised contrastive learning with n positives and n negatives
    """

    def __init__(self, temperature=0.5, mode='all_vs_all', n_samples=None):
        super(ContrastiveEmbeddingLoss, self).__init__()
        self.temperature = temperature
        self.mode = mode
        self.n_samples = n_samples  # Used for '1_vs_n' and 'n_vs_n' modes
        
    def forward(self, labels, embeddings, src, dst, chr):
        if self.mode == 'all_vs_all':
            return self._all_vs_all_forward(labels, embeddings)
        elif self.mode == '1_vs_n':
            return self._one_vs_n_forward(labels, embeddings)
        elif self.mode == 'n_vs_n':
            return self._n_vs_n_forward(labels, embeddings)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose from 'all_vs_all', '1_vs_n', or 'n_vs_n'")
    
    def _all_vs_all_forward(self, labels, embeddings):
        """Original all-vs-all implementation."""
        # Get unique labels for this specific graph
        unique_labels = torch.unique(labels)
        
        # Compute pairwise similarities
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Create positive and negative masks based on same/different clusters
        positive_mask = torch.zeros_like(sim_matrix)
        
        # Handle regular labels (1 and -1)
        for label in unique_labels:
            if label == 0:
                continue  # Handle label 0 separately
                
            indices = (labels == label).nonzero(as_tuple=True)[0]
            for i in indices:
                positive_mask[i, indices] = 1.0
                
                # Nodes with label 0 are positive pairs for both clusters
                zero_indices = (labels == 0).nonzero(as_tuple=True)[0]
                positive_mask[i, zero_indices] = 1.0
        
        # Handle label 0 (belongs to both clusters)
        zero_indices = (labels == 0).nonzero(as_tuple=True)[0]
        for i in zero_indices:
            # Label 0 forms positive pairs with all other labels
            pos_indices = ((labels == 1) | (labels == -1) | (labels == 0)).nonzero(as_tuple=True)[0]
            positive_mask[i, pos_indices] = 1.0
        
        # Remove self-connections from positive mask
        positive_mask.fill_diagonal_(0)
        
        negative_mask = 1.0 - positive_mask
        negative_mask.fill_diagonal_(0)
        
        # Compute contrastive loss
        sim_matrix = sim_matrix / self.temperature
        
        # Apply numerical stability measures
        # 1. Subtract max for numerical stability (prevent overflow)
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max.detach()
        
        exp_sim = torch.exp(sim_matrix)
        
        # For each node, pull same-cluster nodes closer
        pos_sim = torch.sum(positive_mask * exp_sim, dim=1)
        neg_sim = torch.sum(negative_mask * exp_sim, dim=1)
        
        # Nodes with no positive pairs should be excluded
        valid_nodes = torch.sum(positive_mask, dim=1) > 0
        
        # Add small epsilon to prevent division by zero or log(0)
        eps = 1e-8
        loss = -torch.log((pos_sim + eps) / (pos_sim + neg_sim + eps))
        
        # Only compute loss for valid nodes
        if valid_nodes.any():
            loss = torch.mean(loss[valid_nodes])
        else:
            # Return zero loss if no valid nodes
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss
    
    def _one_vs_n_forward(self, labels, embeddings):
        """Standard InfoNCE with one positive and n negatives per anchor - optimized version."""
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Get indices for each label type
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == -1).nonzero(as_tuple=True)[0]
        zero_indices = (labels == 0).nonzero(as_tuple=True)[0]
        
        # If we don't have enough samples of each type, return zero loss
        if len(pos_indices) < 1 or len(neg_indices) < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute all similarities at once
        all_similarities = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Initialize loss accumulator
        total_loss = 0.0
        valid_anchors = 0
        
        # Process positive anchors
        if len(pos_indices) > 0:
            # For each positive anchor, sample positives (including zeros) and negatives
            pos_anchors = pos_indices[torch.randperm(len(pos_indices))[:min(len(pos_indices), 100)]]
            
            for anchor_idx in pos_anchors:
                # Potential positives are other positive nodes and zero nodes
                potential_pos = torch.cat([
                    pos_indices[pos_indices != anchor_idx],
                    zero_indices
                ]) if len(zero_indices) > 0 else pos_indices[pos_indices != anchor_idx]
                
                if len(potential_pos) == 0:
                    continue
                
                # Sample one positive
                pos_idx = potential_pos[torch.randint(0, len(potential_pos), (1,))]
                
                # Sample negatives
                n_neg = min(self.n_samples or len(neg_indices), len(neg_indices))
                if n_neg == 0:
                    continue
                
                neg_samples = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]
                
                # Get similarities for this anchor
                anchor_sims = all_similarities[anchor_idx]
                
                # Get positive and negative similarities
                pos_sim = anchor_sims[pos_idx]
                neg_sims = anchor_sims[neg_samples]
                
                # Compute InfoNCE loss for this anchor
                #logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                
                # Use the direct InfoNCE formula instead of cross_entropy
                pos_exp = torch.exp(pos_sim)
                all_exp = pos_exp + torch.sum(torch.exp(neg_sims))
                loss = -torch.log(pos_exp / all_exp)
                
                total_loss += loss
                valid_anchors += 1
        
        # Process negative anchors
        if len(neg_indices) > 0:
            # For each negative anchor, sample positives (including zeros) and negatives
            neg_anchors = neg_indices[torch.randperm(len(neg_indices))[:min(len(neg_indices), 100)]]
            
            for anchor_idx in neg_anchors:
                # Potential positives are other negative nodes and zero nodes
                potential_pos = torch.cat([
                    neg_indices[neg_indices != anchor_idx],
                    zero_indices
                ]) if len(zero_indices) > 0 else neg_indices[neg_indices != anchor_idx]
                
                if len(potential_pos) == 0:
                    continue
                
                # Sample one positive
                pos_idx = potential_pos[torch.randint(0, len(potential_pos), (1,))]
                
                # Sample negatives
                n_neg = min(self.n_samples or len(pos_indices), len(pos_indices))
                if n_neg == 0:
                    continue
                
                pos_samples = pos_indices[torch.randperm(len(pos_indices))[:n_neg]]
                
                # Get similarities for this anchor
                anchor_sims = all_similarities[anchor_idx]
                
                # Get positive and negative similarities
                pos_sim = anchor_sims[pos_idx]
                neg_sims = anchor_sims[pos_samples]
                
                # Compute InfoNCE loss for this anchor
                # Use the direct InfoNCE formula instead of cross_entropy
                pos_exp = torch.exp(pos_sim)
                all_exp = pos_exp + torch.sum(torch.exp(neg_sims))
                loss = -torch.log(pos_exp / all_exp)
                
                total_loss += loss
                valid_anchors += 1
        
        if valid_anchors > 0:
            return total_loss / valid_anchors
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _n_vs_n_forward(self, labels, embeddings):
        """Supervised contrastive learning with n positives and n negatives - optimized version."""
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Get indices for each label type
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == -1).nonzero(as_tuple=True)[0]
        zero_indices = (labels == 0).nonzero(as_tuple=True)[0]
        
        # If we don't have enough samples of each type, return zero loss
        if len(pos_indices) < 1 or len(neg_indices) < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute all similarities at once
        all_similarities = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Initialize loss accumulator
        total_loss = 0.0
        valid_anchors = 0
        
        # Process positive anchors (batch processing)
        if len(pos_indices) > 0:
            # Sample a subset of positive anchors to reduce computation
            pos_anchors = pos_indices[torch.randperm(len(pos_indices))[:min(len(pos_indices), 50)]]
            
            # Create a mask for positive samples (including zeros)
            pos_mask = torch.zeros(batch_size, device=device)
            pos_mask[pos_indices] = 1.0
            pos_mask[zero_indices] = 1.0
            
            # Create a mask for negative samples
            neg_mask = torch.zeros(batch_size, device=device)
            neg_mask[neg_indices] = 1.0
            
            for anchor_idx in pos_anchors:
                # Remove self from positive mask for this anchor
                curr_pos_mask = pos_mask.clone()
                curr_pos_mask[anchor_idx] = 0.0
                
                # Count positives and negatives
                n_pos = curr_pos_mask.sum().int().item()
                n_neg = neg_mask.sum().int().item()
                
                if n_pos == 0 or n_neg == 0:
                    continue
                
                # Sample n positives and n negatives
                n_to_sample = min(self.n_samples or min(n_pos, n_neg), min(n_pos, n_neg))
                
                # Get anchor similarities
                anchor_sims = all_similarities[anchor_idx]
                
                # Create a mask for the final samples
                final_mask = torch.zeros_like(anchor_sims)
                
                # Sample positives
                pos_probs = curr_pos_mask.float() / curr_pos_mask.sum()
                pos_samples = torch.multinomial(pos_probs, n_to_sample, replacement=False)
                final_mask[pos_samples] = 1.0
                
                # Sample negatives
                neg_probs = neg_mask.float() / neg_mask.sum()
                neg_samples = torch.multinomial(neg_probs, n_to_sample, replacement=False)
                
                # Compute loss using the sampled positives and negatives
                pos_sims = anchor_sims[pos_samples]
                neg_sims = anchor_sims[neg_samples]
                
                # Compute supervised contrastive loss
                pos_exp = torch.exp(pos_sims)
                neg_exp = torch.exp(neg_sims)
                
                loss = -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum() + 1e-8))
                
                total_loss += loss
                valid_anchors += 1
        
        # Process negative anchors (batch processing)
        if len(neg_indices) > 0:
            # Sample a subset of negative anchors to reduce computation
            neg_anchors = neg_indices[torch.randperm(len(neg_indices))[:min(len(neg_indices), 50)]]
            
            # Create a mask for negative samples (including zeros)
            neg_mask = torch.zeros(batch_size, device=device)
            neg_mask[neg_indices] = 1.0
            neg_mask[zero_indices] = 1.0
            
            # Create a mask for positive samples
            pos_mask = torch.zeros(batch_size, device=device)
            pos_mask[pos_indices] = 1.0
            
            for anchor_idx in neg_anchors:
                # Remove self from negative mask for this anchor
                curr_neg_mask = neg_mask.clone()
                curr_neg_mask[anchor_idx] = 0.0
                
                # Count positives and negatives
                n_pos = curr_neg_mask.sum().int().item()
                n_neg = pos_mask.sum().int().item()
                
                if n_pos == 0 or n_neg == 0:
                    continue
                
                # Sample n positives and n negatives
                n_to_sample = min(self.n_samples or min(n_pos, n_neg), min(n_pos, n_neg))
                
                # Get anchor similarities
                anchor_sims = all_similarities[anchor_idx]
                
                # Create a mask for the final samples
                final_mask = torch.zeros_like(anchor_sims)
                
                # Sample positives (other negatives and zeros)
                pos_probs = curr_neg_mask.float() / curr_neg_mask.sum()
                pos_samples = torch.multinomial(pos_probs, n_to_sample, replacement=False)
                final_mask[pos_samples] = 1.0
                
                # Sample negatives (positives)
                neg_probs = pos_mask.float() / pos_mask.sum()
                neg_samples = torch.multinomial(neg_probs, n_to_sample, replacement=False)
                
                # Compute loss using the sampled positives and negatives
                pos_sims = anchor_sims[pos_samples]
                neg_sims = anchor_sims[neg_samples]
                
                # Compute supervised contrastive loss
                pos_exp = torch.exp(pos_sims)
                neg_exp = torch.exp(neg_sims)
                
                loss = -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum() + 1e-8))
                
                total_loss += loss
                valid_anchors += 1
        
        if valid_anchors > 0:
            return total_loss / valid_anchors
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning for node embeddings"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, labels, features, src, dst , chr):
        """Compute supervised contrastive loss for node embeddings.
        
        Args:
            features: Node embeddings of shape [num_nodes, embedding_dim].
            labels: Node labels of shape [num_nodes]. Values should be 1, -1, or 0.
                   Anchors should be 1 or -1 labeled, samples with class 0 count as positive class.
        Returns:
            A loss scalar.
        """
        device = features.device

        # Verify input dimensions
        if len(features.shape) != 2:
            raise ValueError('`features` must be 2D tensor of shape [num_nodes, embedding_dim]')
            
        batch_size = features.shape[0]
        
        # Verify labels
        if len(labels.shape) != 1 or labels.shape[0] != batch_size:
            raise ValueError('`labels` must be 1D tensor of shape [num_nodes]')
            
        # Create mask based on labels with special handling for 0 labels
        labels = labels.contiguous().view(-1, 1)
        
        # Create a binary mask where:
        # - For anchors with label 1: positive pairs are samples with label 1 or 0
        # - For anchors with label -1: positive pairs are samples with label -1 or 0
        # - For anchors with label 0: they don't contribute to the loss
        anchor_is_pos = (labels == 1)
        anchor_is_neg = (labels == -1)
        sample_is_pos = (labels == 1) | (labels == 0)
        sample_is_neg = (labels == -1) | (labels == 0)
        
        # Create positive pair mask - Convert to float before matrix multiplication
        mask = torch.zeros(batch_size, batch_size, device=device)
        mask = torch.where(torch.matmul(anchor_is_pos.float(), sample_is_pos.float().T) > 0, 1.0, mask)
        mask = torch.where(torch.matmul(anchor_is_neg.float(), sample_is_neg.float().T) > 0, 1.0, mask)
        
        # For contrast_mode 'one', we use only the first half of nodes as anchors
        # For contrast_mode 'all', all nodes are used as anchors
        if self.contrast_mode == 'one':
            # Use first half of batch as anchors
            anchor_count = batch_size // 2
            anchor_feature = features[:anchor_count]
            contrast_feature = features
            # Adjust mask for the reduced anchor set
            anchor_mask = mask[:anchor_count]
        elif self.contrast_mode == 'all':
            anchor_count = batch_size
            anchor_feature = features
            contrast_feature = features
            anchor_mask = mask
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Filter out anchors with label 0 (they don't contribute to the loss)
        if self.contrast_mode == 'all':
            valid_anchors = (labels != 0).squeeze()
            if not valid_anchors.any():
                # If no valid anchors, return zero loss
                return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask to exclude self-contrast
        logits_mask = torch.ones((anchor_count, batch_size), device=device)
        # For each anchor, mask out its self-contrast if it's also in the contrast set
        for i in range(min(anchor_count, batch_size)):
            logits_mask[i, i] = 0
            
        # Apply the positive-negative mask
        mask = anchor_mask * logits_mask
        
        # If using all nodes as anchors, zero out rows for anchors with label 0
        if self.contrast_mode == 'all':
            mask = mask * valid_anchors.view(-1, 1)
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Handle cases with no positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 
                                     torch.ones_like(mask_pos_pairs), 
                                     mask_pos_pairs)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # Calculate loss only for valid anchors (label 1 or -1)
        if self.contrast_mode == 'all':
            # Only compute loss for valid anchors
            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
            # Average only over valid anchors
            valid_count = valid_anchors.sum()
            if valid_count > 0:
                loss = (loss * valid_anchors).sum() / valid_count
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()

        return loss
    
class LocalPairLoss(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0):
        super(LocalPairLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, y_true, y_pred, src, dst, g, chr, multi=False):
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

        # Calculate the margin
        margin = torch.abs(y_true_i - y_true_j)

        # Calculate the loss terms
        term_same_label = indicator_same_label * (y_pred_i - y_pred_j) ** 2
        term_diff_label = indicator_diff_label * torch.max(torch.zeros_like(margin),
                                                       margin - torch.abs(y_pred_i - y_pred_j)) ** 2 * 10

        # Combine the terms with the weights
        loss = (self.lambda_1 * term_same_label.mean() +
                self.lambda_2 * term_diff_label.mean())

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

        # Calculate the margin
        margin = torch.abs(y_true_i - y_true_j)

        # Calculate the loss terms
        term_same_label = indicator_same_label * (y_pred_i - y_pred_j) ** 2
        term_diff_label = indicator_diff_label * torch.max(torch.zeros_like(margin),
                                                       margin - torch.abs(y_pred_i - y_pred_j)) ** 2 * 10

        # Calculate the final loss by averaging across all samples
        loss = (self.lambda_1 * term_same_label.mean() +
                self.lambda_2 * term_diff_label.mean())

        return loss

class GlobalPairLossConsise(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0):
        super(GlobalPairLossConsise, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, y_true, y_pred, src, dst, chr, multi=False):
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
        y_true_i = y_true[src]
        y_true_j = y_true[dst]
        y_pred_i = y_pred[src]
        y_pred_j = y_pred[dst]

        # Calculate sign variable s: -1 if same label, +1 if different label
        s = torch.where(y_true_i == y_true_j, 
                        torch.tensor(-1.0, device=y_true.device),
                        torch.tensor(1.0, device=y_true.device))
        
        # Calculate the margin
        margin = torch.abs(y_true_i - y_true_j)
        
        # Calculate the unified loss term
        loss_term = torch.max(torch.zeros_like(margin),
                             margin - s * torch.abs(y_pred_i - y_pred_j))**2
        
        # Apply weights based on whether labels are same or different
        weighted_loss = torch.where(s == -1.0,
                                   self.lambda_1 * loss_term,
                                   self.lambda_2 * loss_term)
        
        return weighted_loss.mean()

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

        # Stack all tensors
        y_true_i = torch.cat(all_true_i)
        y_true_j = torch.cat(all_true_j)
        y_pred_i = torch.cat(all_pred_i)
        y_pred_j = torch.cat(all_pred_j)

        # Calculate sign variable s: -1 if same label, +1 if different label
        s = torch.where(y_true_i == y_true_j, 
                        torch.tensor(-1.0, device=y_true.device),
                        torch.tensor(1.0, device=y_true.device))
        
        # Calculate the margin
        margin = torch.abs(y_true_i - y_true_j)
        
        # Calculate the unified loss term
        loss_term = torch.max(torch.zeros_like(margin),
                             margin - s * torch.abs(y_pred_i - y_pred_j)**2)
        
        # Apply weights based on whether labels are same or different
        weighted_loss = torch.where(s == -1.0,
                                   self.lambda_1 * loss_term,
                                   self.lambda_2 * loss_term)
        
        return weighted_loss.mean()
    
    
class GlobalPairLoss(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0):
        super(GlobalPairLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, y_true, y_pred, src, dst, chr, multi=False):
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
        y_true_i = y_true[src]
        y_true_j = y_true[dst]
        y_pred_i = y_pred[src]
        y_pred_j = y_pred[dst]

        # Calculate the indicators
        indicator_same_label = (y_true_i == y_true_j).float()
        indicator_diff_label = (y_true_i != y_true_j).float()

        # Calculate the margin
        margin = torch.abs(y_true_i - y_true_j)

        # Calculate the loss terms
        term_same_label = indicator_same_label * (y_pred_i - y_pred_j) ** 2
        term_diff_label = indicator_diff_label * torch.max(torch.zeros_like(margin),
                                                       margin - torch.abs(y_pred_i - y_pred_j)) ** 2

        # Combine the terms with the weights
        loss = (self.lambda_1 * term_same_label.mean() +
                self.lambda_2 * term_diff_label.mean())

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

        # Stack all tensors
        y_true_i = torch.cat(all_true_i)
        y_true_j = torch.cat(all_true_j)
        y_pred_i = torch.cat(all_pred_i)
        y_pred_j = torch.cat(all_pred_j)

        # Calculate indicators for all pairs at once
        indicator_same_label = (y_true_i == y_true_j).float()
        indicator_diff_label = (y_true_i != y_true_j).float()

        # Calculate margin for all pairs at once
        margin = torch.abs(y_true_i - y_true_j)

        # Calculate loss terms for all pairs at once
        term_same_label = indicator_same_label * (y_pred_i - y_pred_j) ** 2
        term_diff_label = indicator_diff_label * torch.max(torch.zeros_like(margin),
                                                   margin - torch.abs(y_pred_i - y_pred_j)) ** 2

        # Calculate the final loss by averaging across all samples
        loss = (self.lambda_1 * term_same_label.mean() +
                self.lambda_2 * term_diff_label.mean())

        return loss

class SharedToZeroLoss(nn.Module):
    """Loss that pushes predictions for nodes with label 0 towards 0."""
    def __init__(self, weight=1.0):
        super(SharedToZeroLoss, self).__init__()
        self.weight = weight
        
    def forward(self, y_true, y_pred, src=None, dst=None, g=None, chr=None, multi=False):
        """
        Args:
            y_true: Node labels
            y_pred: Node predictions
            src, dst, g, chr: Not used but included for API compatibility
            multi: Not used but included for API compatibility
        """
        # Create mask for nodes with label 0
        zero_mask = (y_true == 0)
        
        # If no nodes with label 0, return zero loss
        if not zero_mask.any():
            return torch.tensor(0.0, device=y_true.device, requires_grad=True)
        
        # Calculate squared error for nodes with label 0
        zero_loss = (y_pred[zero_mask] ** 2).mean()
        
        return self.weight * zero_loss
    


class TripletLoss(nn.Module):
    """Triplet contrastive loss for node predictions.
    
    This loss pulls nodes with the same label closer together in prediction space
    while pushing nodes with different labels apart.
    
    Special handling for label 0: nodes with label 0 can serve as positive samples
    for both class 1 and class -1 nodes.
    """
    def __init__(self, margin=1.0, reduction='mean'):
        """
        Args:
            margin: Minimum distance between positive and negative pairs
            reduction: 'mean' or 'sum' for the final loss reduction
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, labels, predictions, src, dst, chr, multi=False):
        if multi:
            return self.multi_chr_forward(labels, predictions, src, dst, chr)
        else:
            return self.single_chr_forward(labels, predictions, src, dst, chr)
    
    def single_chr_forward(self, labels, predictions, src, dst, chr):
        device = predictions.device
        batch_size = predictions.shape[0]
        
        # Get indices for each label type
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == -1).nonzero(as_tuple=True)[0]
        zero_indices = (labels == 0).nonzero(as_tuple=True)[0]
        
        # If we don't have enough samples of each type, return zero loss
        if len(pos_indices) < 1 or len(neg_indices) < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Initialize loss accumulator
        total_loss = 0.0
        valid_triplets = 0
        
        # Process positive anchors - vectorized approach
        if len(pos_indices) > 0:
            # Sample all triplets at once
            num_pos_anchors = len(pos_indices)
            
            # Create potential positives for each anchor
            anchor_indices = pos_indices
            
            # For each anchor, get potential positive samples (other positives and zeros)
            # Create a mask for valid positive samples for each anchor
            pos_pos_mask = torch.ones((num_pos_anchors, len(pos_indices)), device=device)
            # Remove self-connections
            for i in range(num_pos_anchors):
                pos_pos_mask[i, i] = 0
            
            # Sample positives (including zeros if available)
            if len(zero_indices) > 0:
                # Sample from both positive nodes (excluding self) and zero nodes
                num_samples = num_pos_anchors
                
                # Create indices for sampling
                pos_sample_indices = torch.multinomial(
                    pos_pos_mask.view(num_pos_anchors, -1), 
                    1, replacement=True
                ).squeeze()
                pos_samples = pos_indices[pos_sample_indices]
                
                # Decide randomly whether to use a positive or zero sample
                use_zero = torch.rand(num_pos_anchors, device=device) < 0.5
                if use_zero.any() and len(zero_indices) > 0:
                    zero_sample_indices = torch.randint(0, len(zero_indices), (num_pos_anchors,), device=device)
                    zero_samples = zero_indices[zero_sample_indices]
                    # Replace some positive samples with zero samples
                    pos_samples = torch.where(use_zero, zero_samples, pos_samples)
            else:
                # Sample only from other positive nodes
                pos_sample_indices = torch.multinomial(
                    pos_pos_mask.view(num_pos_anchors, -1), 
                    1, replacement=True
                ).squeeze()
                pos_samples = pos_indices[pos_sample_indices]
            
            # Sample negative nodes
            if len(neg_indices) > 0:
                neg_sample_indices = torch.randint(0, len(neg_indices), (num_pos_anchors,), device=device)
                neg_samples = neg_indices[neg_sample_indices]
                
                # Get predictions for all triplets
                anchor_preds = predictions[anchor_indices]
                pos_preds = predictions[pos_samples]
                neg_preds = predictions[neg_samples]
                
                # Compute distances
                pos_dists = torch.abs(anchor_preds - pos_preds)
                neg_dists = torch.abs(anchor_preds - neg_preds)
                
                # Compute triplet loss
                losses = torch.clamp(pos_dists - neg_dists + self.margin, min=0.0)
                
                total_loss += losses.sum()
                valid_triplets += num_pos_anchors
        
        # Process negative anchors - vectorized approach
        if len(neg_indices) > 0:
            # Sample all triplets at once
            num_neg_anchors = len(neg_indices)
            
            # Create potential positives for each anchor
            anchor_indices = neg_indices
            
            # For each anchor, get potential positive samples (other negatives and zeros)
            # Create a mask for valid positive samples for each anchor
            neg_neg_mask = torch.ones((num_neg_anchors, len(neg_indices)), device=device)
            # Remove self-connections
            for i in range(num_neg_anchors):
                neg_neg_mask[i, i] = 0
            
            # Sample positives (including zeros if available)
            if len(zero_indices) > 0:
                # Sample from both negative nodes (excluding self) and zero nodes
                num_samples = num_neg_anchors
                
                # Create indices for sampling
                neg_sample_indices = torch.multinomial(
                    neg_neg_mask.view(num_neg_anchors, -1), 
                    1, replacement=True
                ).squeeze()
                neg_samples_as_pos = neg_indices[neg_sample_indices]
                
                # Decide randomly whether to use a negative or zero sample
                use_zero = torch.rand(num_neg_anchors, device=device) < 0.5
                if use_zero.any() and len(zero_indices) > 0:
                    zero_sample_indices = torch.randint(0, len(zero_indices), (num_neg_anchors,), device=device)
                    zero_samples = zero_indices[zero_sample_indices]
                    # Replace some negative samples with zero samples
                    neg_samples_as_pos = torch.where(use_zero, zero_samples, neg_samples_as_pos)
            else:
                # Sample only from other negative nodes
                neg_sample_indices = torch.multinomial(
                    neg_neg_mask.view(num_neg_anchors, -1), 
                    1, replacement=True
                ).squeeze()
                neg_samples_as_pos = neg_indices[neg_sample_indices]
            
            # Sample positive nodes as negatives for negative anchors
            if len(pos_indices) > 0:
                pos_sample_indices = torch.randint(0, len(pos_indices), (num_neg_anchors,), device=device)
                pos_samples_as_neg = pos_indices[pos_sample_indices]
                
                # Get predictions for all triplets
                anchor_preds = predictions[anchor_indices]
                pos_preds = predictions[neg_samples_as_pos]
                neg_preds = predictions[pos_samples_as_neg]
                
                # Compute distances
                pos_dists = torch.abs(anchor_preds - pos_preds)
                neg_dists = torch.abs(anchor_preds - neg_preds)
                
                # Compute triplet loss
                losses = torch.clamp(pos_dists - neg_dists + self.margin, min=0.0)
                
                total_loss += losses.sum()
                valid_triplets += num_neg_anchors
        
        if valid_triplets > 0:
            return total_loss / valid_triplets if self.reduction == 'mean' else total_loss
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def multi_chr_forward(self, labels, predictions, src, dst, chr):
        device = predictions.device
        
        # Get unique chromosomes
        unique_chr = torch.unique(chr)
        
        total_loss = 0.0
        total_valid_triplets = 0
        
        # Process each chromosome separately
        for c in unique_chr:
            # Create mask for current chromosome
            chr_mask = (chr == c)
            chr_nodes = torch.where(chr_mask)[0]
            
            if len(chr_nodes) < 3:  # Need at least 3 nodes for a triplet
                continue
            
            # Get labels and predictions for this chromosome
            chr_labels = labels[chr_nodes]
            chr_predictions = predictions[chr_nodes]
            
            # Get indices for each label type within this chromosome
            pos_indices = (chr_labels == 1).nonzero(as_tuple=True)[0]
            neg_indices = (chr_labels == -1).nonzero(as_tuple=True)[0]
            zero_indices = (chr_labels == 0).nonzero(as_tuple=True)[0]
            
            # If we don't have enough samples of different types, skip this chromosome
            if len(pos_indices) < 1 or len(neg_indices) < 1:
                continue
            
            chr_loss = 0.0
            chr_valid_triplets = 0
            
            # Process positive anchors for this chromosome
            if len(pos_indices) > 0:
                # Sample a subset of positive anchors to reduce computation
                max_anchors = min(len(pos_indices), 50)  # Limit to 50 anchors per chromosome
                pos_anchors = pos_indices[torch.randperm(len(pos_indices))[:max_anchors]]
                
                for anchor_idx in pos_anchors:
                    # Potential positives are other positive nodes and zero nodes
                    potential_pos = torch.cat([
                        pos_indices[pos_indices != anchor_idx],
                        zero_indices
                    ]) if len(zero_indices) > 0 else pos_indices[pos_indices != anchor_idx]
                    
                    if len(potential_pos) == 0:
                        continue
                    
                    # Sample one positive
                    pos_idx = potential_pos[torch.randint(0, len(potential_pos), (1,))]
                    
                    # Sample one negative
                    if len(neg_indices) == 0:
                        continue
                    
                    neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,))]
                    
                    # Get predictions for the triplet
                    anchor_pred = chr_predictions[anchor_idx]
                    pos_pred = chr_predictions[pos_idx]
                    neg_pred = chr_predictions[neg_idx]
                    
                    # Compute distances in prediction space
                    pos_dist = torch.abs(anchor_pred - pos_pred)
                    neg_dist = torch.abs(anchor_pred - neg_pred)
                    
                    # Compute triplet loss
                    loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
                    
                    chr_loss += loss
                    chr_valid_triplets += 1
            
            # Process negative anchors for this chromosome
            if len(neg_indices) > 0:
                # Sample a subset of negative anchors to reduce computation
                max_anchors = min(len(neg_indices), 50)  # Limit to 50 anchors per chromosome
                neg_anchors = neg_indices[torch.randperm(len(neg_indices))[:max_anchors]]
                
                for anchor_idx in neg_anchors:
                    # Potential positives are other negative nodes and zero nodes
                    potential_pos = torch.cat([
                        neg_indices[neg_indices != anchor_idx],
                        zero_indices
                    ]) if len(zero_indices) > 0 else neg_indices[neg_indices != anchor_idx]
                    
                    if len(potential_pos) == 0:
                        continue
                    
                    # Sample one positive
                    pos_idx = potential_pos[torch.randint(0, len(potential_pos), (1,))]
                    
                    # Sample one negative
                    if len(pos_indices) == 0:
                        continue
                    
                    neg_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
                    
                    # Get predictions for the triplet
                    anchor_pred = chr_predictions[anchor_idx]
                    pos_pred = chr_predictions[pos_idx]
                    neg_pred = chr_predictions[neg_idx]
                    
                    # Compute distances in prediction space
                    pos_dist = torch.abs(anchor_pred - pos_pred)
                    neg_dist = torch.abs(anchor_pred - neg_pred)
                    
                    # Compute triplet loss
                    loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
                    
                    chr_loss += loss
                    chr_valid_triplets += 1
            
            if chr_valid_triplets > 0:
                total_loss += chr_loss
                total_valid_triplets += chr_valid_triplets
        
        if total_valid_triplets > 0:
            return total_loss / total_valid_triplets if self.reduction == 'mean' else total_loss
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    

class LocalPairLossConsise(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_2=1.0):
        super(LocalPairLossConsise, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(self, y_true, y_pred, src, dst, chr, multi=False):
        if multi:
            return self.multi_chr_forward(y_true, y_pred, src, dst, chr)
        else:
            return self.single_chr_forward(y_true, y_pred, src, dst, chr)

    def single_chr_forward(self, y_true, y_pred, src, dst, chr):

        # Get labels and predictions for each pair
        y_true_i = y_true[src]
        y_true_j = y_true[dst]
        y_pred_i = y_pred[src]
        y_pred_j = y_pred[dst]

        # Calculate sign variable s: -1 if same label, +1 if different label
        s = torch.where(y_true_i == y_true_j, 
                        torch.tensor(-1.0, device=y_true.device),
                        torch.tensor(1.0, device=y_true.device))
        
        # Calculate the margin
        margin = torch.abs(y_true_i - y_true_j)
        
        # Calculate the unified loss term
        loss_term = torch.max(torch.zeros_like(margin),
                             margin - s * torch.abs(y_pred_i - y_pred_j))**2
        
        # Apply weights based on whether labels are same or different
        weighted_loss = torch.where(s == -1.0,
                                   self.lambda_1 * loss_term,
                                   self.lambda_2 * loss_term)
        
        return weighted_loss.mean()

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

        # Calculate sign variable s: -1 if same label, +1 if different label
        s = torch.where(y_true_i == y_true_j, 
                        torch.tensor(-1.0, device=y_true.device),
                        torch.tensor(1.0, device=y_true.device))
        
        # Calculate the margin
        margin = torch.abs(y_true_i - y_true_j)
        
        # Calculate the unified loss term
        loss_term = torch.max(torch.zeros_like(margin),
                             margin - s * torch.abs(y_pred_i - y_pred_j))**2
        
        # Apply weights based on whether labels are same or different
        weighted_loss = torch.where(s == -1.0,
                                   self.lambda_1 * loss_term,
                                   self.lambda_2 * loss_term)
        
        return weighted_loss.mean()
    
