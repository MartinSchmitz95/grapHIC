
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import itertools
from collections import defaultdict

def visualize_clusters(scaffolding_embs, phasing_embs, final_labels, final_labels_primary, output_dir, 
                      chr_labels=None, hap_labels=None):
    """
    Visualize clustering results using t-SNE.
    
    Args:
        scaffolding_embs: Embeddings for scaffolding
        phasing_embs: Embeddings for phasing
        final_labels: Final cluster assignments (multi-cluster version)
        final_labels_primary: Primary cluster assignments (single label per point)
        output_dir: Directory to save plots
        chr_labels: Ground truth chromosome labels (optional)
        hap_labels: Ground truth haplotype labels (optional)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert embeddings to numpy
    scaffolding_np = scaffolding_embs.cpu().numpy()
    phasing_np = phasing_embs.cpu().numpy()
    
    # Apply t-SNE for dimensionality reduction
    print("Applying t-SNE to scaffolding embeddings...")
    tsne_scaffold = TSNE(n_components=2, random_state=42)
    scaffold_2d = tsne_scaffold.fit_transform(scaffolding_np)
    
    # Plot scaffolding embeddings colored by predicted primary cluster
    plt.figure(figsize=(12, 10))
    unique_clusters = np.unique(final_labels_primary)
    for cluster_val in unique_clusters:
        mask = final_labels_primary == cluster_val
        plt.scatter(scaffold_2d[mask, 0], scaffold_2d[mask, 1], label=f'Cluster {cluster_val}', alpha=0.7)
    
    plt.title('Scaffolding Embeddings by Primary Predicted Cluster')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'scaffolding_by_cluster.png'))
    plt.close()
    
    # Plot points that belong to multiple clusters
    plt.figure(figsize=(12, 10))
    # Points with single cluster
    single_cluster_mask = np.array([len(labels) == 1 for labels in final_labels])
    plt.scatter(scaffold_2d[single_cluster_mask, 0], scaffold_2d[single_cluster_mask, 1], 
               label='Single cluster', alpha=0.5, color='blue')
    
    # Points with multiple clusters
    multi_cluster_mask = np.array([len(labels) > 1 for labels in final_labels])
    plt.scatter(scaffold_2d[multi_cluster_mask, 0], scaffold_2d[multi_cluster_mask, 1], 
               label='Multiple clusters', alpha=0.8, color='red')
    
    plt.title('Points Belonging to Multiple Clusters')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'multi_cluster_points.png'))
    plt.close()
    
    # If ground truth is available, also plot by true labels
    if chr_labels is not None:
        plt.figure(figsize=(12, 10))
        unique_chrs = np.unique(chr_labels)
        for chr_val in unique_chrs:
            mask = chr_labels == chr_val
            plt.scatter(scaffold_2d[mask, 0], scaffold_2d[mask, 1], label=f'Chr {chr_val}', alpha=0.7)
        
        plt.title('Scaffolding Embeddings by Ground Truth Chromosome')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'scaffolding_by_chr.png'))
        plt.close()
    
    # For each scaffold cluster, visualize phasing embeddings
    scaffold_labels = np.unique(final_labels_primary)
    for scaffold_id in scaffold_labels:
        scaffold_mask = final_labels_primary == scaffold_id
        
        # Skip if too few nodes
        if np.sum(scaffold_mask) < 10:
            continue
        
        # Get phasing embeddings for this scaffold
        scaffold_phasing_embs = phasing_np[scaffold_mask]
        
        # Apply t-SNE
        tsne_phasing = TSNE(n_components=2, random_state=42)
        phasing_2d = tsne_phasing.fit_transform(scaffold_phasing_embs)
        
        # Get cluster labels for this scaffold
        scaffold_final_labels = final_labels_primary[scaffold_mask]
        
        # Plot by predicted cluster
        plt.figure(figsize=(12, 10))
        for cluster_val in np.unique(scaffold_final_labels):
            cluster_mask = scaffold_final_labels == cluster_val
            plt.scatter(phasing_2d[cluster_mask, 0], phasing_2d[cluster_mask, 1], 
                       label=f'Cluster {cluster_val}', alpha=0.7)
        
        plt.title(f'Phasing Embeddings for Scaffold Cluster {scaffold_id}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'phasing_scaffold{scaffold_id}.png'))
        plt.close()
        
        # If ground truth is available, also plot by true haplotype
        if hap_labels is not None:
            scaffold_hap_labels = hap_labels[scaffold_mask]
            plt.figure(figsize=(12, 10))
            for hap_val in np.unique(scaffold_hap_labels):
                hap_mask = scaffold_hap_labels == hap_val
                plt.scatter(phasing_2d[hap_mask, 0], phasing_2d[hap_mask, 1], 
                           label=f'Haplotype {hap_val}', alpha=0.7)
            
            plt.title(f'Phasing Embeddings for Scaffold Cluster {scaffold_id} by Ground Truth')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'phasing_scaffold{scaffold_id}_by_hap.png'))
            plt.close()

def to_one_hot(haplotype_labels, n_haps=2):
    """
    Convert haplotype labels to one-hot encoded binary patterns.
    
    Args:
        haplotype_labels: Array of haplotype labels (-1, 0, 1)
        n_haps: Number of haplotypes (default 2)
        
    Returns:
        Numpy array of shape (len(haplotype_labels), n_haps) with binary patterns:
        - 1 becomes (1, 0)
        - -1 becomes (0, 1)
        - 0 becomes (1, 1)
    """
    # Initialize output array
    one_hot = np.ones((len(haplotype_labels), n_haps))
    
    # Set patterns based on labels
    for i, label in enumerate(haplotype_labels):
        if label == 1:
            one_hot[i, 1] = 0  # (1, 0)
        elif label == -1:
            one_hot[i, 0] = 0  # (0, 1)
        # 0 remains (1, 1)
    
    return one_hot

    
def main(phasing_embs, fuzzy_labels, metrics, chr_labels, hap_labels, n_haps, output_dir, visualize=False):
    """
    Evaluate phasing clustering on synthetic data with ground truth labels.
  
    Returns:
        metrics: Dictionary of evaluation metrics
    """

    # Verify that the dimensions match
    print(f"Number of embeddings: {len(phasing_embs)}, Number of labels: {len(chr_labels)}")
    assert len(phasing_embs) == len(chr_labels), "Mismatch between number of embeddings and labels"
    
    # Perform phasing clustering directly
    print("Performing phasing clustering...")
    
    # Use chromosome labels as scaffold labels
    scaffold_labels = chr_labels
    
    # Store the final labels for visualization and saving
    final_labels = metrics.get('final_labels', np.zeros_like(scaffold_labels))
    
    # Evaluate fuzzy clustering
    print("Evaluating fuzzy clustering...")
    fuzzy_metrics, confusion_df = evaluate_fuzzy_clustering(fuzzy_labels, scaffold_labels, hap_labels, n_haps)
    metrics.update(fuzzy_metrics)
    
    # Print unique values of hap_labels and final_labels
    unique_hap_labels = np.unique(hap_labels)
    print(f"Unique hap_labels: {unique_hap_labels}")
    print(f"Number of unique hap_labels: {len(unique_hap_labels)}")
    
    # Calculate metrics per scaffold
    print(f"Calculating per-scaffold metrics...")
    unique_scaffolds = np.unique(scaffold_labels)
    scaffold_fuzzy_accs = []
    
    for scaffold in unique_scaffolds:
        # Get fuzzy accuracy for this scaffold
        if 'per_scaffold_fuzzy_accuracy' in metrics:
            scaffold_fuzzy_acc = metrics['per_scaffold_fuzzy_accuracy'].get(scaffold, 0)
            scaffold_fuzzy_accs.append(scaffold_fuzzy_acc)
            print(f"Scaffold {scaffold}: Fuzzy Accuracy = {scaffold_fuzzy_acc:.4f}")
    
    if scaffold_fuzzy_accs:
        metrics['mean_scaffold_fuzzy_accuracy'] = np.mean(scaffold_fuzzy_accs)
    print("\n=== Confusion Matrix ===")
    print(confusion_df)

    # Print metrics
    print("\nClustering Metrics:")
    if 'overall_fuzzy_accuracy' in metrics:
        print(f"Overall Fuzzy Accuracy: {metrics['overall_fuzzy_accuracy']:.4f}")
    if 'mean_scaffold_fuzzy_accuracy' in metrics:
        print(f"Mean Scaffold Fuzzy Accuracy: {metrics['mean_scaffold_fuzzy_accuracy']:.4f}")
    print(f"Number of final clusters: {metrics['n_final_clusters']}")


    
    # Visualize clusters
    if visualize:
        print("\nVisualizing clusters...")
        visualize_clusters(
            None, phasing_embs, final_labels, 
            output_dir, chr_labels, hap_labels
        )
    
    # Save results
    #save_results(final_labels, metrics, {}, output_dir)
    
    return metrics

def evaluate_fuzzy_clustering(fuzzy_memberships, scaffold_labels, true_haplotypes, n_haps=2):
    """
    Evaluate fuzzy clustering and compute a confusion matrix.
    
    Args:
        fuzzy_memberships: List of lists of haplotype assignments for each node.
        scaffold_labels: Array of scaffold IDs for each node.
        true_haplotypes: Array of true haplotype labels (-1, 0, 1).
        n_haps: Number of haplotypes (default=2).
        
    Returns:
        metrics: Dict with accuracy, per-scaffold stats, and confusion matrix.
        confusion_mat: DataFrame summarizing misclassifications.
    """
    # Initialize metrics
    metrics = {
        'overall_fuzzy_accuracy': 0,
        'per_scaffold_fuzzy_accuracy': {},
        'per_scaffold_best_permutation': {},
        'class_accuracies': {},
    }
    
    # Initialize confusion matrix storage
    pattern_labels = ['(1, 0)', '(0, 1)', '(1, 1)']  # For n_haps=2
    confusion_counts = defaultdict(int)  # Keys: (true_pattern, pred_pattern)
    
    total_nodes = 0
    total_correct = 0
    
    unique_scaffolds = np.unique(scaffold_labels)
    
    for scaffold in unique_scaffolds:
        scaffold_mask = scaffold_labels == scaffold
        scaffold_true_haps = true_haplotypes[scaffold_mask]
        scaffold_fuzzy = [fuzzy_memberships[i] for i in np.where(scaffold_mask)[0]]
        
        # Convert true haplotypes to binary patterns
        true_patterns = to_one_hot(scaffold_true_haps, n_haps)
        
        # Get unique fuzzy haplotypes in this scaffold
        unique_fuzzy_haps = sorted(list(set(np.concatenate(scaffold_fuzzy))))
        
        if not unique_fuzzy_haps:
            continue  # Skip if no predictions
        
        # Convert fuzzy memberships to binary patterns
        fuzzy_patterns = np.zeros((len(scaffold_fuzzy), len(unique_fuzzy_haps)))
        for i, membership in enumerate(scaffold_fuzzy):
            for hap in membership:
                hap_idx = unique_fuzzy_haps.index(hap)
                fuzzy_patterns[i, hap_idx] = 1
        
        # Find best permutation of predicted labels
        best_accuracy = 0
        best_perm = None
        best_reordered = None
        
        for perm in itertools.permutations(range(len(unique_fuzzy_haps))):
            reordered = fuzzy_patterns[:, perm]
            matches = np.all(reordered == true_patterns, axis=1)
            accuracy = np.mean(matches)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_perm = perm
                best_reordered = reordered
        
        # Update metrics
        metrics['per_scaffold_fuzzy_accuracy'][scaffold] = best_accuracy
        metrics['per_scaffold_best_permutation'][scaffold] = best_perm
        total_nodes += len(scaffold_true_haps)
        total_correct += best_accuracy * len(scaffold_true_haps)
        
        # Update confusion matrix counts
        for true_row, pred_row in zip(true_patterns, best_reordered):
            true_key = tuple(true_row)
            pred_key = tuple(pred_row)
            confusion_counts[(true_key, pred_key)] += 1
    
    # Compute overall accuracy
    metrics['overall_fuzzy_accuracy'] = total_correct / total_nodes if total_nodes > 0 else 0
    
    # Convert confusion counts to a readable matrix
    unique_patterns = [(1, 0), (0, 1), (1, 1)]  # Possible true/pred patterns for n_haps=2
    confusion_mat = np.zeros((len(unique_patterns), len(unique_patterns)), dtype=int)
    
    for i, true_pat in enumerate(unique_patterns):
        for j, pred_pat in enumerate(unique_patterns):
            confusion_mat[i, j] = confusion_counts.get((true_pat, pred_pat), 0)
    
    # Convert to a labeled DataFrame (for better readability)
    import pandas as pd
    confusion_df = pd.DataFrame(
        confusion_mat,
        index=[f"True {p}" for p in unique_patterns],
        columns=[f"Pred {p}" for p in unique_patterns],
    )
    
    return metrics, confusion_df