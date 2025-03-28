import argparse
import torch
import numpy as np
import os
import json
from collections import defaultdict
from cluster_utils import (
    set_seed, load_model_and_graph, compute_embeddings, 
    two_step_clustering, visualize_clusters, analyze_clusters, save_results
)

def load_reads_file(reads_file):
    """
    Load reads from a file.
    
    Args:
        reads_file: Path to reads file
        
    Returns:
        reads: List of reads
    """
    reads = []
    with open(reads_file, 'r') as f:
        for line in f:
            reads.append(line.strip())
    return reads

def load_node_to_read_mapping(mapping_file):
    """
    Load node to read mapping from a file.
    
    Args:
        mapping_file: Path to mapping file
        
    Returns:
        n2r: Dictionary mapping node IDs to read IDs
    """
    n2r = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                read_id = parts[1]
                n2r[node_id] = read_id
    return n2r

def split_reads_by_cluster(final_labels, n2r, reads, output_dir):
    """
    Split reads by cluster assignment.
    
    Args:
        final_labels: Final cluster assignments
        n2r: Dictionary mapping node IDs to read IDs
        reads: List of reads
        output_dir: Directory to save split reads
        
    Returns:
        cluster_to_reads: Dictionary mapping cluster IDs to lists of reads
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a mapping from read IDs to read content
    read_id_to_content = {}
    for read in reads:
        # Assuming read ID is the first field in the read
        read_id = read.split()[0]
        read_id_to_content[read_id] = read
    
    # Group reads by cluster
    cluster_to_reads = defaultdict(list)
    
    for node_id, cluster_id in enumerate(final_labels):
        if node_id in n2r:
            read_id = n2r[node_id]
            if read_id in read_id_to_content:
                cluster_to_reads[int(cluster_id)].append(read_id_to_content[read_id])
    
    # Write reads for each cluster to separate files
    for cluster_id, cluster_reads in cluster_to_reads.items():
        output_file = os.path.join(output_dir, f"cluster_{cluster_id}_reads.txt")
        with open(output_file, 'w') as f:
            for read in cluster_reads:
                f.write(f"{read}\n")
        print(f"Wrote {len(cluster_reads)} reads to {output_file}")
    
    # Write summary
    summary = {
        'total_reads': len(reads),
        'total_nodes': len(final_labels),
        'total_clusters': len(cluster_to_reads),
        'reads_per_cluster': {k: len(v) for k, v in cluster_to_reads.items()}
    }
    
    with open(os.path.join(output_dir, 'split_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return cluster_to_reads

def process_real_data(model_path, graph_path, reads_file, mapping_file, output_dir, 
                     device, n_chromosomes=None, n_haps=2, seed=42, visualize=False):
    """
    Process real data to split reads by cluster.
    
    Args:
        model_path: Path to trained model
        graph_path: Path to graph file
        reads_file: Path to reads file
        mapping_file: Path to node-to-read mapping file
        output_dir: Directory to save results
        device: Device to run computation on
        n_scaffold_clusters: Number of scaffold clusters (if None, auto-detect)
        seed: Random seed
        
    Returns:
        cluster_to_reads: Dictionary mapping cluster IDs to lists of reads
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
    
    # Perform hierarchical clustering
    print("Performing two step  clustering...")
    final_labels, metrics = two_step_clustering(
        scaffolding_embs, phasing_embs, n_scaffold_clusters=n_chromosomes,
        n_haps=n_haps, scaffold_fuzzy=True, phasing_fuzzy=True, 
        scaffold_overlap_threshold=0.3, scaffold_fuzziness=2,
        phasing_overlap_threshold=0.3, phasing_fuzziness=2
    )
     
    
    # Analyze clusters
    print("Analyzing cluster composition...")
    cluster_stats = analyze_clusters(final_labels)
    
    # Print metrics
    print("\nClustering Results:")
    print(f"Number of final clusters: {metrics['n_final_clusters']}")
    
    # Print cluster statistics
    print("\nCluster Statistics:")
    for cluster_id, stats in sorted(cluster_stats.items()):
        print(f"Cluster {cluster_id} (size: {stats['size']} nodes)")
    
    # Visualize clusters
    if visualize:
        print("\nVisualizing clusters...")
        visualize_clusters(
            scaffolding_embs, phasing_embs, final_labels, output_dir
        )
    
    # Save clustering results
    save_results(final_labels, metrics, cluster_stats, output_dir)
    
    # Load reads and node-to-read mapping
    print("Loading reads and node-to-read mapping...")
    reads = load_reads_file(reads_file)
    n2r = load_node_to_read_mapping(mapping_file)
    
    # Split reads by cluster
    print("Splitting reads by cluster...")
    cluster_to_reads = split_reads_by_cluster(final_labels, n2r, reads, output_dir)
    
    print(f"\nResults saved to {output_dir}")
    
    return cluster_to_reads

def main():
    parser = argparse.ArgumentParser(description="Split reads by clustering using trained embeddings")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to graph file")
    parser.add_argument("--reads_file", type=str, required=True, help="Path to reads file")
    parser.add_argument("--mapping_file", type=str, required=True, help="Path to node-to-read mapping file")
    parser.add_argument("--output_dir", type=str, default="split_reads_results", help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run computation on")
    parser.add_argument("--n_chromosomes", type=int, default=None, 
                        help="Number of chromosomes (default: auto-detect)")
    parser.add_argument("--n_haps", type=int, default=2, help="Number of haplotypes per scaffold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Visualize clusters")
    
    args = parser.parse_args()
    
    process_real_data(
        args.model_path, 
        args.graph_path, 
        args.reads_file,
        args.mapping_file,
        args.output_dir, 
        args.device, 
        args.n_chromosomes, 
        args.n_haps,
        args.seed,
        args.visualize
    )

if __name__ == "__main__":
    main()