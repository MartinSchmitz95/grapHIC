# Set environment variables to limit threads for numerical libraries
# Add these at the top of the file before other imports are processed
import os

max_threads = "16"
os.environ["OMP_NUM_THREADS"] = max_threads  # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = max_threads  # OpenBLAS
os.environ["MKL_NUM_THREADS"] = max_threads  # MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = max_threads  # Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = max_threads  # NumExpr

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.manifold import TSNE
from collections import Counter, defaultdict
import yaml
import json
from skfuzzy import cmeans


# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def compute_embeddings(model, graph, device='cpu'):
    """
    Compute scaffolding and phasing embeddings for a graph.
    
    Args:
        model: Trained model
        graph: Input graph
        device: Device to run computation on
        exclude_homozygous: If True, exclude nodes with graph.y=0 from the returned embeddings
        
    Returns:
        phasing_embs: Phasing embeddings
        scaffolding_embs: Scaffolding embeddings
    """
    # Prepare graph features
    graph.x = torch.abs(graph.y).float().unsqueeze(1).to(device)
    
    """# Check if PE features exist and use them
    if hasattr(graph, 'pe_0') and hasattr(graph, 'pe_1'):
        graph.x = torch.cat([graph.x.to(device), graph.pe_0.to(device), graph.pe_1.to(device)], dim=1)
    """
    with torch.no_grad():
        phasing_embs = model(graph)
    
    return phasing_embs

def scaffold_clustering(scaffolding_embs, n_scaffold_clusters=None, chr_labels=None, fuzzy=False, 
                        scaffold_overlap_threshold=0.3, scaffold_fuzziness=2):
    """
    Perform clustering on scaffolding embeddings (chromosome-level).
    
    Args:
        scaffolding_embs: Embeddings for scaffolding (chromosome-level)
        n_scaffold_clusters: Number of scaffold clusters
        chr_labels: Ground truth chromosome labels (optional, for evaluation)
        fuzzy: Whether to use fuzzy clustering (True) or hard clustering (False)
        scaffold_overlap_threshold: Threshold for determining if a point belongs to multiple clusters (for fuzzy)
        scaffold_fuzziness: Fuzziness parameter for c-means (m > 1, higher values = softer clustering)
        
    Returns:
        scaffold_labels: Cluster assignments (list of lists for fuzzy, numpy array for hard)
        metrics: Dictionary of evaluation metrics (if ground truth provided)
        next_label: Next available cluster label
    """
    # Convert embeddings to numpy for clustering
    scaffolding_np = scaffolding_embs.cpu().numpy()
    
    # Determine number of scaffold clusters if not provided
    if n_scaffold_clusters is None:
        # Estimate using silhouette score or other methods
        best_score = -1
        best_n = 2
        for n in range(2, min(30, len(scaffolding_np) // 10)):
            clustering = KMeans(n_clusters=n, random_state=42)
            labels = clustering.fit_predict(scaffolding_np)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(scaffolding_np, labels)
            if score > best_score:
                best_score = score
                best_n = n
        n_scaffold_clusters = best_n
    
    print(f"Performing scaffolding clustering with {n_scaffold_clusters} clusters")
    
    # Metrics dictionary
    metrics = {}
    
    if not fuzzy:
        # Hard clustering using KMeans
        scaffold_clustering = KMeans(n_clusters=n_scaffold_clusters, random_state=42)
        scaffold_labels = scaffold_clustering.fit_predict(scaffolding_np)
        
        # Evaluate scaffolding clustering if ground truth is provided
        if chr_labels is not None:
            scaffold_ari = adjusted_rand_score(chr_labels, scaffold_labels)
            scaffold_nmi = normalized_mutual_info_score(chr_labels, scaffold_labels)
            
            print(f"Scaffolding clustering metrics:")
            print(f"  Adjusted Rand Index [-1,1]: {scaffold_ari:.4f}")
            print(f"  Normalized Mutual Information [0,1]: {scaffold_nmi:.4f}")
            
            metrics['scaffold_ari'] = scaffold_ari
            metrics['scaffold_nmi'] = scaffold_nmi
        
        return scaffold_labels, metrics, n_scaffold_clusters
    
    else:
        # Fuzzy clustering using cmeans
        cntr, u, u0, d, jm, p, fpc = cmeans(
            scaffolding_np.T,  # Transpose for skfuzzy format
            c=n_scaffold_clusters,  # Number of clusters
            m=scaffold_fuzziness,   # Fuzziness parameter
            error=0.005,       # Stopping criterion
            maxiter=1000,      # Maximum number of iterations
            init=None          # Initial fuzzy c-partitioned matrix
        )
        
        # Create fuzzy labels (list of lists)
        scaffold_labels_fuzzy = [[] for _ in range(len(scaffolding_np))]
        
        # For each point, find clusters it belongs to based on membership threshold
        for i in range(u.shape[1]):  # For each sample
            # Assign to clusters where membership exceeds threshold
            for cluster_idx in range(u.shape[0]):  # For each cluster
                if u[cluster_idx, i] >= scaffold_overlap_threshold:
                    scaffold_labels_fuzzy[i].append(cluster_idx)
            
            # Ensure each point belongs to at least one cluster
            if not scaffold_labels_fuzzy[i]:
                best_cluster = np.argmax(u[:, i])
                scaffold_labels_fuzzy[i].append(best_cluster)
        
        # For evaluation, create hard labels using the highest membership
        hard_labels = np.argmax(u, axis=0)
        
        # Evaluate scaffolding clustering if ground truth is provided
        if chr_labels is not None:
            scaffold_ari = adjusted_rand_score(chr_labels, hard_labels)
            scaffold_nmi = normalized_mutual_info_score(chr_labels, hard_labels)
            
            print(f"Scaffolding clustering metrics:")
            print(f"  Adjusted Rand Index [-1,1]: {scaffold_ari:.4f}")
            print(f"  Normalized Mutual Information [0,1]: {scaffold_nmi:.4f}")
            
            metrics['scaffold_ari'] = scaffold_ari
            metrics['scaffold_nmi'] = scaffold_nmi
        
        return scaffold_labels_fuzzy, metrics, n_scaffold_clusters

def phasing_clustering(phasing_embs, scaffold_labels, n_haps=2, fuzzy=False, 
                       phasing_overlap_threshold=0.3, phasing_fuzziness=2):
    """
    Perform phasing clustering within each scaffold cluster.
    
    Args:
        phasing_embs: Embeddings for phasing (haplotype-level)
        scaffold_labels: Scaffold cluster assignments (numpy array)
        n_haps: Number of haplotypes per scaffold (default: 2 for diploid)
        fuzzy: Whether to use fuzzy clustering (True) or hard clustering (False)
        phasing_overlap_threshold: Threshold for determining if a point belongs to multiple clusters (for fuzzy)
        phasing_fuzziness: Fuzziness parameter for c-means (m > 1, higher values = softer clustering)
        
    Returns:
        final_labels: Final cluster assignments (numpy array for hard clustering, list of lists for fuzzy)
        metrics: Dictionary of evaluation metrics
    """
    # Convert embeddings to numpy for clustering
    phasing_np = phasing_embs.cpu().numpy()
    
    # Metrics dictionary
    metrics = {}
    
    # Get unique scaffold cluster IDs
    unique_scaffold_ids = np.unique(scaffold_labels)
    
    # Initialize final labels
    final_labels_multi = [[] for _ in range(len(phasing_np))]
    next_label = 0
    
    # For each scaffold cluster
    for scaffold_id in unique_scaffold_ids:
        # Get indices of nodes in this scaffold cluster
        print(f"Processing scaffold cluster {scaffold_id}")
        cluster_mask = scaffold_labels == scaffold_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Get phasing embeddings for this cluster
        cluster_phasing_embs = phasing_np[cluster_indices]
        

        
        if fuzzy:
            # Perform fuzzy c-means clustering
            # Note: cmeans expects data in shape (features, samples) so we transpose
            cntr, u, u0, d, jm, p, fpc = cmeans(
                cluster_phasing_embs.T,  # Transpose for skfuzzy format
                c=n_haps,                # Number of clusters
                m=phasing_fuzziness,     # Fuzziness parameter
                error=0.005,             # Stopping criterion
                maxiter=1000,            # Maximum number of iterations
                init=None                # Initial fuzzy c-partitioned matrix
            )
            
            # u contains membership values for each point to each cluster
            # u shape is (n_clusters, n_samples)
            
            # For each point, find clusters it belongs to based on membership threshold
            for i in range(u.shape[1]):  # For each sample
                # Assign to clusters where membership exceeds threshold
                for cluster_idx in range(u.shape[0]):  # For each cluster
                    if u[cluster_idx, i] >= phasing_overlap_threshold:
                        cluster_label = next_label + cluster_idx
                        final_labels_multi[cluster_indices[i]].append(cluster_label)
                
                # Ensure each point belongs to at least one cluster
                if not final_labels_multi[cluster_indices[i]]:
                    best_cluster = np.argmax(u[:, i])
                    final_labels_multi[cluster_indices[i]].append(next_label + best_cluster)
        else:
            # Perform hard KMeans clustering
            kmeans = KMeans(n_clusters=min(n_haps, len(cluster_indices)), random_state=42)
            cluster_labels = kmeans.fit_predict(cluster_phasing_embs)
            
            # Assign labels
            for i, idx in enumerate(cluster_indices):
                final_labels_multi[idx].append(next_label + cluster_labels[i])
        
        next_label += n_haps
    
    metrics['n_final_clusters'] = next_label
    
    # If using hard clustering, convert to numpy array
    if not fuzzy:
        final_labels_hard = np.zeros(len(phasing_np), dtype=int)
        for i, labels in enumerate(final_labels_multi):
            if labels:
                final_labels_hard[i] = labels[0]
        return final_labels_hard, metrics
    
    return final_labels_multi, metrics

def two_step_clustering(scaffolding_embs, phasing_embs, n_scaffold_clusters=None,
                        n_haps=2, chr_labels=None, scaffold_fuzzy=False, phasing_fuzzy=True, 
                        scaffold_overlap_threshold=0.3, scaffold_fuzziness=2,
                        phasing_overlap_threshold=0.3, phasing_fuzziness=2):
    """
    Perform hierarchical clustering: first cluster by scaffolding embeddings,
    then by phasing embeddings within each scaffold cluster.
    
    Args:
        scaffolding_embs: Embeddings for scaffolding (chromosome-level)
        phasing_embs: Embeddings for phasing (haplotype-level)
        n_scaffold_clusters: Number of scaffold clusters
        chr_labels: Ground truth chromosome labels (optional, for evaluation)
        n_haps: Number of haplotypes per scaffold (default: 2 for diploid)
        scaffold_fuzzy: Whether to use fuzzy clustering for scaffolding
        phasing_fuzzy: Whether to use fuzzy clustering for phasing
        scaffold_overlap_threshold: Threshold for scaffold fuzzy clustering
        scaffold_fuzziness: Fuzziness parameter for scaffold fuzzy clustering
        phasing_overlap_threshold: Threshold for phasing fuzzy clustering
        phasing_fuzziness: Fuzziness parameter for phasing fuzzy clustering
        
    Returns:
        final_labels: Final cluster assignments (list of lists for fuzzy, numpy array for hard)
        metrics: Dictionary of evaluation metrics (if ground truth provided)
    """
    # Step 1: Cluster using scaffolding embeddings (chromosome-level)
    scaffold_labels, scaffold_metrics, n_scaffold_clusters = scaffold_clustering(
        scaffolding_embs, 
        n_scaffold_clusters=n_scaffold_clusters, 
        chr_labels=chr_labels,
        fuzzy=scaffold_fuzzy,
        scaffold_overlap_threshold=scaffold_overlap_threshold,
        scaffold_fuzziness=scaffold_fuzziness
    )
    
    # Step 2: For each scaffold cluster, perform phasing clustering
    final_labels, phasing_metrics = phasing_clustering(
        phasing_embs,
        scaffold_labels,
        n_haps=n_haps,
        fuzzy=phasing_fuzzy,
        phasing_overlap_threshold=phasing_overlap_threshold,
        phasing_fuzziness=phasing_fuzziness
    )
    
    # Combine metrics
    metrics = {**scaffold_metrics, **phasing_metrics}
    
    return final_labels, metrics

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

def analyze_clusters(labels, chr_labels=None, hap_labels=None):
    """
    Analyze cluster composition.
    
    Args:
        labels: Cluster labels
        chr_labels: Optional ground truth chromosome labels
        hap_labels: Optional ground truth haplotype labels
        
    Returns:
        cluster_stats: Dictionary of cluster statistics
    """
    # Convert labels to numpy array if it's not already
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Get unique cluster IDs
    all_cluster_ids = set(labels)
    
    cluster_stats = {}
    
    for cluster_id in all_cluster_ids:
        # Get indices of nodes in this cluster
        cluster_mask = labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        stats = {
            'size': int(cluster_size),
        }
        
        # If ground truth labels are provided, analyze composition
        if chr_labels is not None and hap_labels is not None:
            # Get chromosome and haplotype labels for this cluster
            cluster_chr_labels = chr_labels[cluster_mask]
            cluster_hap_labels = hap_labels[cluster_mask]
            
            # Count occurrences of each chromosome
            chr_counts = {}
            for chr_id in np.unique(cluster_chr_labels):
                chr_count = np.sum(cluster_chr_labels == chr_id)
                chr_counts[int(chr_id)] = int(chr_count)
            
            # Count occurrences of each haplotype
            hap_counts = {}
            for hap_id in np.unique(cluster_hap_labels):
                hap_count = np.sum(cluster_hap_labels == hap_id)
                hap_counts[int(hap_id)] = int(hap_count)
            
            # Calculate purity (percentage of most common chromosome)
            if cluster_size > 0:
                most_common_chr = max(chr_counts.items(), key=lambda x: x[1])
                chr_purity = most_common_chr[1] / cluster_size
                
                # Calculate haplotype purity within the most common chromosome
                most_common_chr_mask = cluster_chr_labels == most_common_chr[0]
                most_common_chr_haps = cluster_hap_labels[most_common_chr_mask]
                
                hap_counts_in_chr = {}
                for hap_id in np.unique(most_common_chr_haps):
                    hap_count = np.sum(most_common_chr_haps == hap_id)
                    hap_counts_in_chr[int(hap_id)] = int(hap_count)
                
                if len(most_common_chr_haps) > 0:
                    most_common_hap = max(hap_counts_in_chr.items(), key=lambda x: x[1])
                    hap_purity = most_common_hap[1] / len(most_common_chr_haps)
                else:
                    hap_purity = 0.0
            else:
                chr_purity = 0.0
                hap_purity = 0.0
            
            stats.update({
                'chr_counts': chr_counts,
                'hap_counts': hap_counts,
                'chr_purity': float(chr_purity),
                'hap_purity': float(hap_purity),
                'dominant_chr': int(most_common_chr[0]),
                'dominant_chr_count': int(most_common_chr[1])
            })
        
        cluster_stats[int(cluster_id)] = stats
    
    return cluster_stats

def save_results(final_labels, final_labels_primary, metrics, cluster_stats, output_dir, filename='clustering_results.json'):
    """
    Save clustering results to a JSON file.
    
    Args:
        final_labels: Multi-cluster assignments (list of lists)
        final_labels_primary: Primary cluster assignments (single label per point)
        metrics: Dictionary of evaluation metrics
        cluster_stats: Dictionary with cluster statistics
        output_dir: Directory to save results
        filename: Name of the output file
    """
    results = {
        'metrics': metrics,
        'cluster_stats': {str(k): v for k, v in cluster_stats.items()},  # Convert keys to strings for JSON
        'final_labels_primary': final_labels_primary.tolist(),
        'final_labels_multi': [labels for labels in final_labels]  # Already a list of lists
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    return output_path