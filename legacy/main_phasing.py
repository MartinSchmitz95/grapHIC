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
import yaml
import evaluate_phasing
import inference_phasing
from skfuzzy.cluster import cmeans
from sklearn.cluster import KMeans

def compute_embeddings(model, graph, device='cpu'):

    # Prepare graph features
    graph.x = torch.abs(graph.y).float().unsqueeze(1).to(device)
    
    with torch.no_grad():
        phasing_embs = model(graph)
    
    return phasing_embs

def load_model_and_graph(model_path, graph_path, device='cpu'):
    """
    Load a trained model and a graph.
    
    Args:
        model_path: Path to the trained model
        graph_path: Path to the graph
        device: Device to run computation on
        
    Returns:
        model: Loaded model
        graph: Loaded graph
    """
    print(f"Loading graph from {graph_path}")
    graph = torch.load(graph_path, map_location=device)
    
    # Load model configuration
    with open('train_config.yml') as file:
        config = yaml.safe_load(file)['training']
    
    # Import the model class
    from SGformer_HGC import SGFormer
    
    # Create model instance
    model = SGFormer(
        in_channels=config['node_features'],
        hidden_channels=config['hidden_features'],
        out_channels= config['emb_dim'],
        projection_dim=config['projection_dim'],
        trans_num_layers=config['num_trans_layers'],
        trans_dropout= 0, #config['dropout'],
        gnn_num_layers=config['num_gnn_layers_overlap'],
        gnn_dropout= 0, #config['gnn_dropout'],
    ).to(device)

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, graph

def get_embeddings(model_path, graph_path, output_dir, embeddings_path=None, save_embeddings=False, device="cpu"):
# Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get embeddings and labels
    if embeddings_path and os.path.exists(embeddings_path) and not save_embeddings:
        # Load pre-computed embeddings and labels from file
        print(f"Loading embeddings from {embeddings_path}...")
        checkpoint = torch.load(embeddings_path, map_location=device)
        phasing_embs = checkpoint['embeddings']
        chr_labels = checkpoint['chr_labels']
        hap_labels = checkpoint['hap_labels']
        print(f"Loaded embeddings with shape {phasing_embs.shape}")
    else:
        # Load model and graph
        model, graph = load_model_and_graph(model_path, graph_path, device)
        
        # Compute embeddings
        print("Computing embeddings...")
        phasing_embs = compute_embeddings(model, graph, device)
        
        # Get ground truth labels (include homozygous nodes)
        if graph.chr is not None:
            chr_labels = graph.chr.cpu().numpy()
            hap_labels = graph.y.cpu().numpy()
        else:
            chr_labels = None
            hap_labels = None
        
        # Save embeddings if requested
        if save_embeddings:
            embeddings_file = os.path.join(output_dir, "embeddings.pt")
            print(f"Saving embeddings to {embeddings_file}...")
            torch.save({
                'embeddings': phasing_embs,
                'chr_labels': chr_labels,
                'hap_labels': hap_labels
            }, embeddings_file)
    
    original_id = graph.original_id
    return phasing_embs, chr_labels, hap_labels, original_id

def phasing_clustering(phasing_embs, scaffold_labels, original_id, n_haps=2, fuzzy=False, 
                       phasing_overlap_threshold=0.3, phasing_fuzziness=2):
    """
    Perform phasing clustering within each scaffold cluster.
    
    Args:
        phasing_embs: Embeddings for phasing (haplotype-level)
        scaffold_labels: Scaffold cluster assignments (numpy array)
        graph: PyG graph containing original_id attribute
        n_haps: Number of haplotypes per scaffold (default: 2 for diploid)
        fuzzy: Whether to use fuzzy clustering (True) or hard clustering (False)
        phasing_overlap_threshold: Threshold for determining if a point belongs to multiple clusters (for fuzzy)
        phasing_fuzziness: Fuzziness parameter for c-means (m > 1, higher values = softer clustering)
        
    Returns:
        final_labels: Dictionary mapping original node IDs to cluster assignments
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
    
    print(f"Fuzzy Clustering Done. Created {next_label} clusters")
    
    # Convert to dictionary mapping original node IDs to labels
    final_labels_dict = {}
    for i, labels in enumerate(final_labels_multi):
        node_id = original_id[i].item()
        final_labels_dict[node_id] = labels
    
    return final_labels_dict, metrics

def phasing_clustering_from_gt(chr_labels, hap_labels, graph_path, device="cpu"):

    graph = torch.load(graph_path, map_location=device)

    """
    Create phasing clusters directly from ground truth chromosome and haplotype labels.
    
    Args:
        chr_labels: Chromosome/scaffold labels (numpy array)
        hap_labels: Haplotype labels (-1, 0, 1) where:
                    -1: belongs to first haplotype of chromosome
                     0: belongs to both haplotypes (homozygous)
                     1: belongs to second haplotype of chromosome
        graph: PyG graph containing original_id attribute
        
    Returns:
        final_labels: Dictionary mapping original node IDs to cluster assignments
        metrics: Dictionary of evaluation metrics
    """

    # Check if ground truth labels are available
    if chr_labels is None or hap_labels is None:
        raise ValueError("Ground truth labels are required for phasing_clustering_from_gt")
    
    # Convert to numpy if needed
    if isinstance(chr_labels, torch.Tensor):
        chr_labels = chr_labels.cpu().numpy()
    if isinstance(hap_labels, torch.Tensor):
        hap_labels = hap_labels.cpu().numpy()
    
    # Get unique chromosome IDs
    unique_chr_ids = np.unique(chr_labels)
    
    # Initialize final labels (list of lists for fuzzy clustering format)
    final_labels = [[] for _ in range(len(chr_labels))]
    next_label = 0
    metrics = {}
    
    # For tracking bases in each bin
    bin_base_counts = {}
    chr_stats = {}
    
    # For each chromosome
    for chr_id in unique_chr_ids:
        # Get indices of nodes in this chromosome
        print(f"Processing chromosome {chr_id}")
        chr_mask = chr_labels == chr_id
        chr_indices = np.where(chr_mask)[0]
        
        # Count haplotype labels for this chromosome
        hap_minus_one_count = np.sum(hap_labels[chr_indices] == -1)
        hap_zero_count = np.sum(hap_labels[chr_indices] == 0)
        hap_plus_one_count = np.sum(hap_labels[chr_indices] == 1)
        total_nodes = len(chr_indices)
        
        # Initialize base counts for this chromosome's bins
        hap1_bin = next_label
        hap2_bin = next_label + 1
        bin_base_counts[hap1_bin] = 0
        bin_base_counts[hap2_bin] = 0
        
        # Initialize chromosome statistics
        chr_stats[chr_id] = {
            'hap1_only_bases': 0,  # Bases only in haplotype 1
            'hap2_only_bases': 0,  # Bases only in haplotype 2
            'homozygous_bases': 0,  # Bases in both haplotypes
            'total_bases': 0       # Total bases across both haplotypes
        }
        
        print(f"  Chromosome {chr_id} statistics:")
        print(f"    Total nodes: {total_nodes}")
        print(f"    Haplotype -1 (first haplotype): {hap_minus_one_count} ({hap_minus_one_count/total_nodes:.2%})")
        print(f"    Haplotype  0 (homozygous): {hap_zero_count} ({hap_zero_count/total_nodes:.2%})")
        print(f"    Haplotype +1 (second haplotype): {hap_plus_one_count} ({hap_plus_one_count/total_nodes:.2%})")
        
        # For each node in this chromosome
        for i in chr_indices:
            # Get the node's sequence length from the feature matrix
            # The read_length is at index 2 in the feature matrix
            node_id = graph.original_id[i].item()
            node_length = graph.x[i, 2].item()   # Get the normalized read length
            
            if hap_labels[i] == -1:
                # Add to first bin of chromosome
                final_labels[i].append(next_label)
                bin_base_counts[hap1_bin] += node_length
                chr_stats[chr_id]['hap1_only_bases'] += node_length
                chr_stats[chr_id]['total_bases'] += node_length
            elif hap_labels[i] == 1:
                # Add to second bin of chromosome
                final_labels[i].append(next_label + 1)
                bin_base_counts[hap2_bin] += node_length
                chr_stats[chr_id]['hap2_only_bases'] += node_length
                chr_stats[chr_id]['total_bases'] += node_length
            elif hap_labels[i] == 0:
                # Add to both bins of chromosome (homozygous)
                final_labels[i].append(next_label)
                final_labels[i].append(next_label + 1)
                bin_base_counts[hap1_bin] += node_length
                bin_base_counts[hap2_bin] += node_length
                chr_stats[chr_id]['homozygous_bases'] += node_length
                chr_stats[chr_id]['total_bases'] += node_length * 2  # Count twice since it's in both haplotypes
            else:
                print(f"Unknown haplotype label: {hap_labels[i]}")
                exit()
        
        # Calculate total unique bases (counting homozygous regions only once)
        unique_bases = chr_stats[chr_id]['hap1_only_bases'] + chr_stats[chr_id]['hap2_only_bases'] + chr_stats[chr_id]['homozygous_bases']
        
        # Print base counts and percentages for this chromosome
        print(f"    Base pair statistics:")
        print(f"      Haplotype 1 (bin {hap1_bin}) total bases: {bin_base_counts[hap1_bin]:,}")
        print(f"      Haplotype 2 (bin {hap2_bin}) total bases: {bin_base_counts[hap2_bin]:,}")
        
        # Only print percentages if we have bases
        if unique_bases > 0:
            print(f"      Haplotype 1 only bases: {chr_stats[chr_id]['hap1_only_bases']:,} ({chr_stats[chr_id]['hap1_only_bases']/unique_bases:.2%} of unique bases)")
            print(f"      Haplotype 2 only bases: {chr_stats[chr_id]['hap2_only_bases']:,} ({chr_stats[chr_id]['hap2_only_bases']/unique_bases:.2%} of unique bases)")
            print(f"      Homozygous bases: {chr_stats[chr_id]['homozygous_bases']:,} ({chr_stats[chr_id]['homozygous_bases']/unique_bases:.2%} of unique bases)")
            print(f"      Total unique bases: {unique_bases:,}")
        else:
            print(f"      No sequence length information available for this chromosome")
        
        # Move to next chromosome (2 bins per chromosome)
        next_label += 2
    
    # Calculate global statistics
    total_hap1_only = sum(stats['hap1_only_bases'] for stats in chr_stats.values())
    total_hap2_only = sum(stats['hap2_only_bases'] for stats in chr_stats.values())
    total_homozygous = sum(stats['homozygous_bases'] for stats in chr_stats.values())
    total_unique_bases = total_hap1_only + total_hap2_only + total_homozygous
    total_bin_bases = sum(bin_base_counts.values())
    
    print("\nGlobal statistics:")
    print(f"  Total bins created: {next_label}")
    print(f"  Total bases across all bins: {total_bin_bases:,}")
    
    if total_unique_bases > 0:
        print(f"  Total unique bases: {total_unique_bases:,}")
        print(f"  Haplotype 1 only bases: {total_hap1_only:,} ({total_hap1_only/total_unique_bases:.2%} of unique bases)")
        print(f"  Haplotype 2 only bases: {total_hap2_only:,} ({total_hap2_only/total_unique_bases:.2%} of unique bases)")
        print(f"  Homozygous bases: {total_homozygous:,} ({total_homozygous/total_unique_bases:.2%} of unique bases)")
    else:
        print("  No sequence length information available")
    
    metrics['n_final_clusters'] = next_label
    metrics['bin_base_counts'] = bin_base_counts
    metrics['chr_stats'] = chr_stats
    metrics['global_stats'] = {
        'total_hap1_only': total_hap1_only,
        'total_hap2_only': total_hap2_only,
        'total_homozygous': total_homozygous,
        'total_unique_bases': total_unique_bases,
        'total_bin_bases': total_bin_bases
    }
    
    # Convert to dictionary mapping original node IDs to labels
    final_labels_dict = {}
    for i, labels in enumerate(final_labels):
        node_id = graph.original_id[i].item()
        final_labels_dict[node_id] = labels
    
    return final_labels_dict, metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate phasing clustering on synthetic data")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--graph_path", type=str, help="Path to graph file")
    parser.add_argument("--output_dir", type=str, default="/home/schmitzmf/scratch/cluster_results/test", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run computation on")
    parser.add_argument("--n_haps", type=int, default=2, help="Number of haplotypes per scaffold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Visualize clusters")
    parser.add_argument("--embeddings_path", type=str, help="Path to pre-computed embeddings file")
    parser.add_argument("--save_embeddings", action="store_true", help="Save computed embeddings to file")
    parser.add_argument("--fuzziness", type=float, default=1.9, help="Fuzziness parameter for scaffold clustering")
    parser.add_argument("--evaluate",  action="store_true", help="Evaluate clusters")
    parser.add_argument("--inference",  action="store_true", help="Inference with clusters")
    parser.add_argument("--n2s_path", type=str, help="Path to reads file")
    parser.add_argument("--nn2on_path", type=str, help="Path to reads file")
    parser.add_argument("--mat_yak", type=str, help="Path to maternal YAK index")
    parser.add_argument("--pat_yak", type=str, help="Path to paternal YAK index")
    parser.add_argument("--ref", type=str, help="Path to reference genome")
    args = parser.parse_args()

    # Initialize embeddings_path
    embeddings_path = args.embeddings_path
    
    if not args.embeddings_path:
        embeddings_file = os.path.join(args.output_dir, "embeddings.pt")
        if os.path.exists(embeddings_file):
            embeddings_path = embeddings_file

    phasing_embs, chr_labels, hap_labels, original_id = get_embeddings(args.model_path, args.graph_path, args.output_dir, embeddings_path, args.save_embeddings, args.device)


    """fuzzy_labels, metrics = phasing_clustering(
        phasing_embs, chr_labels, original_id, n_haps=args.n_haps, 
        fuzzy=True, phasing_overlap_threshold=0.3, phasing_fuzziness=args.fuzziness)
    
    if args.evaluate:
        evaluate_phasing.main(
            phasing_embs,
            fuzzy_labels,
            metrics,
            chr_labels,
            hap_labels,
            args.n_haps,
            args.output_dir,
            args.visualize,
        )"""
    
    fuzzy_labels, metrics = phasing_clustering_from_gt(chr_labels, hap_labels, args.graph_path, args.device)
    if args.inference:
        inference_phasing.main(
            fuzzy_labels,
            args.n2s_path,
            args.output_dir,
            ref = args.ref,
            mat_yak = args.mat_yak,
            pat_yak = args.pat_yak,
            )
if __name__ == "__main__":
    main()