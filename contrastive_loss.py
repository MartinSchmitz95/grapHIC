# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import random



def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.get_world_size())]
    torch.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class ConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, device, temperature=0.1):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None
        self.device = device

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, feats, labels):
        # Removed normalization line
        # feats = F.normalize(feats, dim=-1, p=2)
        
        batch_size = feats.size(0)

        # No distributed gathering - use features directly
        all_feats = feats
        all_labels = labels

        # compute the mask based on labels
        if batch_size != self.last_local_batch_size:
            mask = torch.eq(labels.view(-1, 1),
                            all_labels.contiguous().view(1, -1)).float().to(self.device)
            self.logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(self.device),
                0
            )

            self.last_local_batch_size = batch_size
            self.mask = mask * self.logits_mask

        mask = self.mask

        # compute logits
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

        return loss

class MultiLabelConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss with support for multi-label classification.
    Samples can belong to multiple classes simultaneously.
    """

    def __init__(self, device, temperature=0.1):
        super(MultiLabelConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None
        self.device = device

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, feats, labels):
        """
        Args:
            feats: Feature vectors [batch_size, feature_dim]
            labels: Multi-hot encoded labels [batch_size, num_classes] where 1 indicates membership
                   in that class, and a sample can have multiple 1s
        """
        # Removed normalization line
        # feats = F.normalize(feats, dim=-1, p=2)
        
        local_batch_size = feats.size(0)

        all_feats = feats
        all_labels = labels

        # compute the mask based on multi-label overlap
        if local_batch_size != self.last_local_batch_size:
            # For multi-label: two samples are positive if they share at least one class
            # Compute dot product between multi-hot label vectors
            # If result > 0, they share at least one class
            label_overlap = torch.matmul(labels, all_labels.T) > 0
            mask = label_overlap.float().to(self.device)
            
            self.logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(self.device),
                0
            )

            self.last_local_batch_size = local_batch_size
            self.mask = mask * self.logits_mask

        mask = self.mask

        # compute logits
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # optional: minus the largest logit to stabilize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

        return loss

"""class SupervisedContrastiveLoss(nn.Module):

    def __init__(self, device, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def set_temperature(self, temp=0.07):
        self.temperature = temp

    def forward(self, feats, targets):
        device = self.device

        # Compute dot product between all pairs of features
        dot_product_tempered = torch.mm(feats, feats.T) / self.temperature
        
        # Minus max for numerical stability with exponential
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        # Create mask for samples of the same class
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        
        # Create mask to exclude self-comparisons
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        
        # Combine masks
        mask_combined = mask_similar_class * mask_anchor_out
        
        # Count number of positives per sample
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        # Compute log probabilities
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        
        # Compute loss per sample and average
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples.clamp(min=1.0)
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss"""

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, device, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.device = device
    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        
        # Original implementation for smaller batch sizes
        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss
    
    def _memory_efficient_forward(self, projections, targets):
        """
        Memory-efficient implementation that avoids creating the full N×N matrices
        
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """        # Ensure targets are on the same device
        targets = targets.to(self.device)
        
        batch_size = projections.shape[0]
        
        # Normalize projections for cosine similarity
        projections = F.normalize(projections, p=2, dim=1)
        
        # Process in chunks to avoid OOM
        chunk_size = min(256, batch_size)
        loss_sum = 0
        valid_samples = 0
        
        for i in range(0, batch_size, chunk_size):
            # Get current chunk
            end_idx = min(i + chunk_size, batch_size)
            chunk_size_actual = end_idx - i
            
            anchor_features = projections[i:end_idx]
            anchor_labels = targets[i:end_idx]
            
            # Compute similarities for this chunk with all projections
            similarities = torch.matmul(anchor_features, projections.T) / self.temperature
            
            # Create mask for samples with the same class
            mask_similar_class = (anchor_labels.unsqueeze(1) == targets.unsqueeze(0)).to(device)
            
            # Create mask to exclude self-comparisons
            mask_self = torch.zeros(chunk_size_actual, batch_size, device=device)
            for j in range(chunk_size_actual):
                mask_self[j, i+j] = 1
            
            mask_anchor_out = 1 - mask_self
            mask_combined = mask_similar_class * mask_anchor_out
            
            # Count positives per sample
            cardinality_per_samples = torch.sum(mask_combined, dim=1)
            
            # Skip samples with no positives
            valid_sample_mask = cardinality_per_samples > 0
            if not torch.any(valid_sample_mask):
                continue
                
            # Compute exp similarities with numerical stability
            similarities_max, _ = torch.max(similarities, dim=1, keepdim=True)
            exp_similarities = torch.exp(similarities - similarities_max)
            
            # Compute denominator (sum over all except self)
            exp_similarities_masked = exp_similarities * mask_anchor_out
            denominator = torch.sum(exp_similarities_masked, dim=1, keepdim=True)
            
            # Compute log probabilities
            log_probs = torch.log(exp_similarities / denominator + 1e-5)
            
            # Compute loss for valid samples
            loss_per_sample = -torch.sum(mask_combined * log_probs, dim=1) / cardinality_per_samples.clamp(min=1.0)
            
            # Add to total loss
            loss_sum += torch.sum(loss_per_sample[valid_sample_mask])
            valid_samples += torch.sum(valid_sample_mask)
        
        # Return average loss
        if valid_samples > 0:
            return loss_sum / valid_samples
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
    

class MultiDimHammingLoss(nn.Module):
    """
    Multi-dimensional extension of HammingLoss that works with embedding vectors
    instead of scalar predictions.
    """
    def __init__(self, lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, margin=1.0):
        super(MultiDimHammingLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.margin = margin  # Margin for separation between different classes

    def forward(self, y_true, embeddings, src=None, dst=None, chr=None, multi=True):
        """
        Args:
            y_true: Ground truth labels
            embeddings: Multi-dimensional embeddings [batch_size, embedding_dim]
            src, dst, chr: Optional parameters for chromosome-based sampling
            multi: Whether to use multi-chromosome sampling
        """
        if multi and chr is not None:
            return self.multi_chr_forward(y_true, embeddings, src, dst, chr)
        else:
            return self.single_chr_forward(y_true, embeddings, src, dst)

    def single_chr_forward(self, y_true, embeddings, src=None, dst=None):
        device = embeddings.device
        
        # Create pairs if not provided
        if src is None or dst is None:
            nodes = list(range(len(y_true)))
            shuffled_nodes = nodes.copy()
            random.shuffle(shuffled_nodes)
            
            src = torch.tensor(nodes, device=device)
            dst = torch.tensor(shuffled_nodes, device=device)
        
        # Get labels and embeddings for each pair
        y_true_i = y_true[src]
        y_true_j = y_true[dst]
        embed_i = embeddings[src]
        embed_j = embeddings[dst]
        
        # Calculate pairwise distances in embedding space
        # Using squared Euclidean distance for efficiency
        distances = torch.sum((embed_i - embed_j)**2, dim=1)
        
        # Calculate the indicators
        indicator_same_label = (y_true_i == y_true_j).float()
        indicator_diff_label = (y_true_i != y_true_j).float()
        indicator_label_zero = (y_true == 0).float()
        
        # Calculate the label margin
        label_margin = torch.abs(y_true_i - y_true_j)
        
        # Calculate the loss terms:
        # 1. Pull same-class embeddings together
        term_same_label = indicator_same_label * distances
        
        # 2. Push different-class embeddings apart with margin
        # If classes are different, distance should be at least margin * label_difference
        desired_min_distance = label_margin * self.margin
        term_diff_label = indicator_diff_label * torch.relu(desired_min_distance - torch.sqrt(distances + 1e-6))**2
        
        # 3. Push zero-labeled samples toward origin
        zero_distances = torch.sum(embeddings**2, dim=1)
        term_label_zero = indicator_label_zero * zero_distances
        
        # Combine the terms with the weights
        loss = (self.lambda_1 * term_same_label.mean() +
                self.lambda_2 * term_diff_label.mean() +
                self.lambda_3 * term_label_zero.mean())
        
        return loss
    
    def multi_chr_forward(self, y_true, embeddings, src, dst, chr):
        device = embeddings.device
        
        # Get unique chromosomes and their counts
        unique_chr, chr_counts = torch.unique(chr, return_counts=True)
        
        # Filter out chromosomes with less than 2 nodes
        valid_mask = chr_counts >= 2
        unique_chr = unique_chr[valid_mask]
        
        if len(unique_chr) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Create a mask tensor for all chromosomes at once
        chr_masks = chr.unsqueeze(0) == unique_chr.unsqueeze(1)  # Broadcasting
        
        # Initialize tensors to store results for all chromosomes
        all_true_i = []
        all_true_j = []
        all_embed_i = []
        all_embed_j = []
        all_zero_indicators = []
        
        for idx, mask in enumerate(chr_masks):
            # Get nodes for current chromosome
            chr_nodes = torch.where(mask)[0]
            
            if len(chr_nodes) < 2:
                continue  # Skip if not enough nodes
            
            # Create pairs within chromosome using a safer approach
            num_nodes = len(chr_nodes)
            
            # Create a safe permutation that won't cause out-of-bounds indexing
            indices = torch.arange(num_nodes, device=device)
            shuffled_indices = indices[torch.randperm(num_nodes, device=device)]
            
            # Safely gather the values
            nodes_i = chr_nodes
            nodes_j = chr_nodes[shuffled_indices]
            
            # Double-check that indices are valid
            if torch.any(nodes_j >= len(embeddings)):
                # Skip this chromosome if any indices are invalid
                continue
                
            # Gather the values using the masks
            all_true_i.append(y_true[nodes_i])
            all_true_j.append(y_true[nodes_j])
            all_embed_i.append(embeddings[nodes_i])
            all_embed_j.append(embeddings[nodes_j])
            all_zero_indicators.append(y_true[mask] == 0)
        
        # Check if we have any valid pairs
        if not all_true_i:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Stack all tensors
        y_true_i = torch.cat(all_true_i)
        y_true_j = torch.cat(all_true_j)
        embed_i = torch.cat(all_embed_i)
        embed_j = torch.cat(all_embed_j)
        
        # Calculate pairwise distances in embedding space
        distances = torch.sum((embed_i - embed_j)**2, dim=1)
        
        # Calculate indicators for all pairs at once
        indicator_same_label = (y_true_i == y_true_j).float()
        indicator_diff_label = (y_true_i != y_true_j).float()
        
        # Safely handle zero indicators
        if all_zero_indicators:
            indicator_label_zero = torch.cat(all_zero_indicators).float()
        else:
            indicator_label_zero = torch.zeros(0, device=device)
        
        # Calculate label margin
        label_margin = torch.abs(y_true_i - y_true_j)
        
        # Calculate loss terms
        term_same_label = indicator_same_label * distances
        
        # Push different classes apart with margin proportional to label difference
        desired_min_distance = label_margin * self.margin
        term_diff_label = indicator_diff_label * torch.relu(desired_min_distance - torch.sqrt(distances + 1e-6))**2
        
        # Push zero-labeled samples toward origin - with safer handling
        zero_masks = [mask for mask in chr_masks if torch.any(mask)]
        if zero_masks:
            zero_embeddings = torch.cat([embeddings[mask] for mask in zero_masks])
            zero_distances = torch.sum(zero_embeddings**2, dim=1)
            
            # Make sure indicator_label_zero has the same length as zero_distances
            if len(indicator_label_zero) > 0:
                if len(indicator_label_zero) != len(zero_distances):
                    # Resize to match
                    min_len = min(len(indicator_label_zero), len(zero_distances))
                    indicator_label_zero = indicator_label_zero[:min_len]
                    zero_distances = zero_distances[:min_len]
                
                term_label_zero = indicator_label_zero * zero_distances
                term_label_zero = term_label_zero.mean()
            else:
                term_label_zero = torch.tensor(0.0, device=device)
        else:
            term_label_zero = torch.tensor(0.0, device=device)
        
        # Calculate the final loss by averaging across all samples
        loss = (self.lambda_1 * term_same_label.mean() +
                self.lambda_2 * term_diff_label.mean() +
                self.lambda_3 * term_label_zero)
        
        return loss
    

def adaptive_clustering_loss(embeddings, labels, edge_index, temperature=0.5):
    # Get unique labels for this specific graph
    unique_labels = torch.unique(labels)
    num_clusters = len(unique_labels)
    
    # Compute pairwise similarities
    sim_matrix = torch.mm(embeddings, embeddings.t())
    
    # Create positive and negative masks based on same/different clusters
    positive_mask = torch.zeros_like(sim_matrix)
    for label in unique_labels:
        indices = (labels == label).nonzero(as_tuple=True)[0]
        for i in indices:
            positive_mask[i, indices] = 1.0
    
    # Remove self-connections from positive mask
    positive_mask.fill_diagonal_(0)
    
    negative_mask = 1.0 - positive_mask
    negative_mask.fill_diagonal_(0)
    
    # Compute contrastive loss
    sim_matrix = sim_matrix / temperature
    exp_sim = torch.exp(sim_matrix)
    
    # For each node, pull same-cluster nodes closer
    pos_sim = torch.sum(positive_mask * exp_sim, dim=1)
    neg_sim = torch.sum(negative_mask * exp_sim, dim=1)
    
    # Nodes with no positive pairs should be excluded
    valid_nodes = torch.sum(positive_mask, dim=1) > 0
    
    # Compute loss only for valid nodes
    loss = -torch.log(pos_sim / (pos_sim + neg_sim))
    loss = torch.mean(loss[valid_nodes])
    
    return loss

def multi_adaptive_clustering_loss(embeddings, multi_labels, edge_index=None, temperature=0.5, device='cpu'):
    """
    Adaptive clustering loss for multi-label classification where samples can belong to multiple classes.
    
    Args:
        embeddings: Feature embeddings of shape [batch_size, embedding_dim]
        multi_labels: Multi-hot encoded labels of shape [batch_size, num_classes] where 1 indicates
                     membership in that class, and a sample can have multiple 1s
        edge_index: Optional edge indices (not used in this implementation but kept for API consistency)
        temperature: Temperature parameter for scaling similarities
    
    Returns:
        Contrastive loss value
    """
    # Compute pairwise similarities
    sim_matrix = torch.mm(embeddings, embeddings.t()).to(device)
    
    # Create positive mask based on label overlap
    # Two samples are positive pairs if they share at least one class
    # Compute dot product between multi-hot label vectors
    # If result > 0, they share at least one class
    label_overlap = torch.mm(multi_labels, multi_labels.t()).to(device) > 0
    positive_mask = label_overlap.float()
    
    # Remove self-connections from positive mask
    positive_mask.fill_diagonal_(0)
    
    # Create negative mask (samples that don't share any classes)
    negative_mask = 1.0 - positive_mask
    negative_mask.fill_diagonal_(0)
    
    # Apply temperature scaling
    sim_matrix = sim_matrix / temperature
    
    # For numerical stability, subtract max from each row
    sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()
    
    exp_sim = torch.exp(sim_matrix)
    
    # For each node, pull nodes with shared classes closer
    pos_sim = torch.sum(positive_mask * exp_sim, dim=1)
    neg_sim = torch.sum(negative_mask * exp_sim, dim=1)
    
    # Nodes with no positive pairs should be excluded
    valid_nodes = torch.sum(positive_mask, dim=1) > 0
    
    if not torch.any(valid_nodes):
        # Return zero loss if no valid nodes (no positive pairs)
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute loss only for valid nodes
    # Add small epsilon to prevent log(0)
    loss = -torch.log((pos_sim + 1e-8) / (pos_sim + neg_sim + 1e-8))
    loss = torch.mean(loss[valid_nodes])
    
    return loss