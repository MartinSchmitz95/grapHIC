# Set environment variables at the very top of the file
import os
os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"  # MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr

import argparse
import torch
import numpy as np
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from cluster_utils import (
    set_seed, load_model_and_graph, compute_embeddings, 
    two_step_clustering, visualize_clusters, analyze_clusters, save_results
)

def evaluate_synthetic_data(model_path, graph_path, output_dir, device, estimate_clusters, n_haps, seed=42, visualize=False):
    """
    Evaluate clustering on synthetic data with ground truth labels.
    
    Args:
        model_path: Path to trained model
        graph_path: Path to graph file
        output_dir: Directory to save results
        device: Device to run computation on
        estimate_clusters: Whether to automatically estimate number of clusters
        n_haps: Number of haplotypes per scaffold
        seed: Random seed
        visualize: Whether to visualize clusters
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Set random seed
    set_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and graph
    model, graph = load_model_and_graph(model_path, graph_path, device)
    
    # Compute embeddings
    print("Computing embeddings...")
    phasing_embs, scaffolding_embs = compute_embeddings(model, graph, device)
    
    # Get ground truth labels and apply the same mask
    exclude_homozygous = True
    if exclude_homozygous:
        # Create mask for heterozygous nodes (where graph.y != 0)
        het_mask = graph.y != 0
        
        # Apply mask to embeddings
        phasing_embs = phasing_embs[het_mask]
        scaffolding_embs = scaffolding_embs[het_mask]
        chr_labels = graph.chr[het_mask].cpu().numpy()
        hap_labels = graph.y[het_mask].cpu().numpy()
        print(f"Excluded {(~het_mask).sum().item()} homozygous nodes, {het_mask.sum().item()} heterozygous nodes remaining")
        
    else:
        chr_labels = graph.chr.cpu().numpy()
        hap_labels = graph.y.cpu().numpy()
    
    # Verify that the dimensions match
    print(f"Number of embeddings: {len(scaffolding_embs)}, Number of labels: {len(chr_labels)}")
    assert len(scaffolding_embs) == len(chr_labels), "Mismatch between number of embeddings and labels"
    
    if estimate_clusters:
        n_scaffold_clusters = None
    else:
        n_scaffold_clusters = len(np.unique(chr_labels))
    
    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    final_labels, metrics = two_step_clustering(
        scaffolding_embs, phasing_embs, n_scaffold_clusters, n_haps,
        chr_labels, scaffold_fuzzy=False, phasing_fuzzy=False, 
        scaffold_overlap_threshold=0.3, scaffold_fuzziness=2,
        phasing_overlap_threshold=0.3, phasing_fuzziness=2
    )
     
    # Calculate overall metrics
    overall_ari = adjusted_rand_score(hap_labels, final_labels)
    overall_nmi = normalized_mutual_info_score(hap_labels, final_labels)
    
    metrics.update({
        'overall_ari': overall_ari,
        'overall_nmi': overall_nmi,
    })
    
    # Analyze clusters
    print("Analyzing cluster composition...")
    cluster_stats = analyze_clusters(final_labels, chr_labels, hap_labels)
    
    # Print metrics
    print("\nClustering Metrics:")
    print(f"Scaffolding ARI: {metrics['scaffold_ari']:.4f}")
    print(f"Scaffolding NMI: {metrics['scaffold_nmi']:.4f}")
    print(f"Overall ARI: {metrics['overall_ari']:.4f}")
    print(f"Overall NMI: {metrics['overall_nmi']:.4f}")
    print(f"Number of final clusters: {metrics['n_final_clusters']}")
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    for cluster_id, stats in sorted(cluster_stats.items()):
        print(f"Cluster {cluster_id} (size: {stats['size']}):")
        print(f"  Most common chromosome: {stats['most_common_chr']} (purity: {stats['chr_purity']:.2f})")
        print(f"  Most common haplotype: {stats['most_common_hap']} (purity: {stats['hap_purity']:.2f})")
    
    # Visualize clusters
    if visualize:
        print("\nVisualizing clusters...")
        visualize_clusters(
            scaffolding_embs, phasing_embs, final_labels, 
            output_dir, chr_labels, hap_labels
        )
    
    # Save results
    save_results(final_labels, metrics, cluster_stats, output_dir)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering on synthetic data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to graph file")
    parser.add_argument("--output_dir", type=str, default="/home/schmitzmf/scratch/cluster_results/test", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run computation on")
    parser.add_argument("--estimate_clusters", type=int, default=None, 
                        help="Number of scaffold clusters (default: auto-detect)")
    parser.add_argument("--n_haps", type=int, default=2, help="Number of haplotypes per scaffold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Visualize clusters")
    
    args = parser.parse_args()
    

    evaluate_synthetic_data(
        args.model_path, 
        args.graph_path, 
        args.output_dir, 
        args.device, 
        args.estimate_clusters, 
        args.n_haps,
        args.seed,
        args.visualize
    )

if __name__ == "__main__":
    main()