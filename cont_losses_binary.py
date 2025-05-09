import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import random

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