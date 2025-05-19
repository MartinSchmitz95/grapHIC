import torch
import argparse
import yaml
from skfuzzy.cluster import cmeans
from sklearn.cluster import KMeans
import eval
import baselines

def get_fuzzy_clustering_from_predictions(preds, fuzzy_thr=0.3):
    # Initialize cluster sets
    cluster_sets = {
        0: [],  # Positive cluster
        1: []   # Negative cluster
    }
    
    # Assign elements to clusters
    for i, pred in enumerate(preds):
        # If prediction is positive or within threshold of zero, add to positive cluster
        if pred >= -fuzzy_thr:
            cluster_sets[0].append(i)
        
        # If prediction is negative or within threshold of zero, add to negative cluster
        if pred <= fuzzy_thr:
            cluster_sets[1].append(i)
    
    return cluster_sets

def get_hard_clustering_from_predictions(preds, threshold=0.0):
    # Initialize empty list for cluster assignments
    cluster_labels = []
    
    # Assign elements to clusters
    for pred in preds:
        # If prediction is greater than or equal to threshold, assign to cluster 1 (positive)
        # Otherwise, assign to cluster 0 (negative)
        if pred >= threshold:
            cluster_labels.append(1)
        else:
            cluster_labels.append(0)
    
    return cluster_labels

def get_fuzzy_clustering_from_embeddings(embs, n_haps=2, fuzziness=2, overlap_threshold=0.3, margin=0.25, use_margin=True):
    # Perform fuzzy c-means clustering
    # Note: cmeans expects data in shape (features, samples) so we transpose
    cntr, u, u0, d, jm, p, fpc = cmeans(
        embs.T,  # Transpose for skfuzzy format
        c=n_haps,                # Number of clusters
        m=fuzziness,             # Fuzziness parameter
        error=0.005,             # Stopping criterion
        maxiter=1000,            # Maximum number of iterations
        init=None                # Initial fuzzy c-partitioned matrix
    )
    
    # Create cluster sets based on membership values and threshold
    cluster_sets = {}
    # For each cluster
    for i in range(n_haps):
        cluster_sets[i] = []
    
    # For each data point
    for j in range(len(embs)):
        if use_margin:
            # Use margin-based approach
            if n_haps == 2:
                # Calculate difference between membership values
                diff = abs(u[0, j] - u[1, j])
                
                if diff <= margin:
                    # If difference is small (within margin), assign to both clusters
                    # but only if membership exceeds threshold
                    for i in range(n_haps):
                        if u[i, j] >= overlap_threshold:
                            cluster_sets[i].append(j)
                else:
                    # If difference is large, assign only to the cluster with higher membership
                    # but only if that membership exceeds threshold
                    max_cluster = u[:, j].argmax()
                    if u[max_cluster, j] >= overlap_threshold:
                        cluster_sets[max_cluster].append(j)
            else:
                # For more than 2 clusters, use a different approach
                # Find the two highest membership values
                sorted_memberships = sorted([(i, u[i, j]) for i in range(n_haps)], 
                                           key=lambda x: x[1], reverse=True)
                
                # If difference between top two memberships is small
                if len(sorted_memberships) >= 2 and (sorted_memberships[0][1] - sorted_memberships[1][1]) <= margin:
                    # Assign to all clusters with membership above threshold
                    for i in range(n_haps):
                        if u[i, j] >= overlap_threshold:
                            cluster_sets[i].append(j)
                else:
                    # Otherwise assign only to highest membership cluster if above threshold
                    max_cluster = sorted_memberships[0][0]
                    if u[max_cluster, j] >= overlap_threshold:
                        cluster_sets[max_cluster].append(j)
        else:
            # Use original approach - check if all memberships are above/below threshold
            if all(u[i, j] >= overlap_threshold for i in range(n_haps)) or all(u[i, j] < overlap_threshold for i in range(n_haps)):
                # If all memberships are above threshold or all are below, assign to all clusters
                for i in range(n_haps):
                    if u[i, j] >= overlap_threshold:
                        cluster_sets[i].append(j)
            else:
                # Otherwise, only assign to clusters where membership exceeds threshold
                for i in range(n_haps):
                    if u[i, j] >= overlap_threshold:
                        cluster_sets[i].append(j)
    
    return cluster_sets

def get_hard_clustering_from_embeddings(embs, n_haps=2, random_state=42):
    
    # Perform hard KMeans clustering
    kmeans = KMeans(n_clusters=n_haps, random_state=random_state)
    pred_labels = kmeans.fit_predict(embs)
    
    # Return list of cluster assignments
    return pred_labels.tolist()

def run_model(model_type, model_path, graph, config):
    
    if model_type == "pred":
        from SGformer import SGFormer
        model = SGFormer(in_channels=config['node_features'], 
                         hidden_channels=config['hidden_features'], 
                         out_channels=1, 
                         trans_num_layers=config['num_trans_layers'], 
                         gnn_num_layers=config['num_gnn_layers_overlap'], 
                         gnn_dropout=0,
                         layer_norm=config['layer_norm'])

    elif model_type == "emb":
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
            layer_norm=config['layer_norm'])

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Prepare graph features
    x_addon = torch.abs(graph.y).float().unsqueeze(1)
    graph.x = torch.cat([graph.x, x_addon], dim=1)
    with torch.no_grad():
        output = model(graph)

    return output

def get_fuzzy_labels_from_graph(graph_y):
    """
    Create fuzzy labels from graph.y tensor.
    - Entries with 1 go to cluster 0
    - Entries with -1 go to cluster 1
    - Entries with 0 go to both clusters
    
    Returns a dictionary with cluster indices as keys and lists of node indices as values.
    """
    # Initialize cluster sets
    fuzzy_labels = {
        0: [],  # Positive cluster (1)
        1: []   # Negative cluster (-1)
    }
    
    # Assign elements to clusters
    for i, val in enumerate(graph_y):
        if val >= 0:  # If value is 0 or 1, add to cluster 0
            fuzzy_labels[0].append(i)
        
        if val <= 0:  # If value is 0 or -1, add to cluster 1
            fuzzy_labels[1].append(i)
    
    return fuzzy_labels

def hard_to_fuzzy(hard_clusters):
    fuzzy_clusters = {0: [], 1: []}
    for i, cluster in enumerate(hard_clusters):
        fuzzy_clusters[cluster].append(i)
    return fuzzy_clusters

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate phasing clustering on synthetic data")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    parser.add_argument("--graph_path", type=str, help="Path to graph file")
    parser.add_argument("--output_dir", type=str, default="/home/schmitzmf/scratch/cluster_results/test", help="Output directory for results")
    parser.add_argument("--mode", type=str, default="mrl", help="mode")
    parser.add_argument("--config_path", type=str, default="train_config.yml", help="Path to configuration file")
    args = parser.parse_args()
    return args

def main(args):

    print(f"Loading graph from {args.graph_path}")

    graph = torch.load(args.graph_path)
        
    # Create binary mask (1 for valid nodes, 0 for masked nodes)
    mask = (graph.y != 0).int().bool()
    # Create hetero_labels where 0s are masked out
    hetero_labels = graph.y[mask].clone()
    # Convert labels to increasing integers (0 and 1)
    # Map 1 to 0 and -1 to 1
    hetero_labels = list((hetero_labels == -1).int().cpu().numpy())
    # Create fuzzy labels from graph.y
    fuzzy_labels = get_fuzzy_labels_from_graph(graph.y)
    
    # Load model configuration
    with open(args.config_path) as file:
        config = yaml.safe_load(file)['training']

    print("Computing predictions and clusters...")
    if  args.mode == "pred":
        preds = run_model("pred", args.model_path, graph, config)
        # Print statistics of predictions
        print(f"Predictions statistics:")
        print(f"  Mean: {preds.mean().item():.4f}")
        print(f"  Std: {preds.std().item():.4f}")
        print(f"  Min: {preds.min().item():.4f}")
        print(f"  Max: {preds.max().item():.4f}")
        print(f"  Median: {torch.median(preds).item():.4f}")
        
        hetero_preds = preds[mask]
        hard_clusters = get_hard_clustering_from_predictions(hetero_preds)
        fuzzy_clusters = get_fuzzy_clustering_from_predictions(preds)
    elif args.mode == "emb":
        embeddings = run_model('emb', args.model_path, graph, config)
        hetero_embeddings = embeddings[mask]
        hard_clusters = get_hard_clustering_from_embeddings(hetero_embeddings, n_haps=2)
        fuzzy_clusters = get_fuzzy_clustering_from_embeddings(embeddings, n_haps=2)
    elif args.mode == "spectral":
        cluster_result = baselines.spectral_clustering(graph)
        # Convert list to tensor, apply mask, then convert back to list
        cluster_result_tensor = torch.tensor(cluster_result)
        hard_clusters = cluster_result_tensor[mask].tolist()
        fuzzy_clusters = hard_to_fuzzy(cluster_result)
    elif args.mode == "louvain":
        cluster_result = baselines.louvain_clustering(graph)
        # Convert list to tensor, apply mask, then convert back to list
        cluster_result_tensor = torch.tensor(cluster_result)
        hard_clusters = cluster_result_tensor[mask].tolist()
        fuzzy_clusters = hard_to_fuzzy(cluster_result)
    elif args.mode == "lp":
        cluster_result = baselines.label_propagation(graph)
        # Convert list to tensor, apply mask, then convert back to list
        cluster_result_tensor = torch.tensor(cluster_result)
        hard_clusters = cluster_result_tensor[mask].tolist()
        fuzzy_clusters = hard_to_fuzzy(cluster_result)

    print("Computing metrics...")
    # Print distribution of labels and clusters
    print("Label distribution:")
    label_counts = {}
    for l in hetero_labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    print(f"  {label_counts}")
    
    print("Cluster distribution:")
    cluster_counts = {}
    for c in hard_clusters:
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
    print(f"  {cluster_counts}")
    
    metrics = eval.clustering_metrics(hetero_labels, hard_clusters)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ARI: {metrics['ARI']:.3f}")
    print(f"NMI: {metrics['NMI']:.3f}")

    # Print fuzzy clustering statistics
    print("\nFuzzy clustering statistics:")
    
    # Count total nodes in graph
    total_nodes = graph.y.size(0)
    print(f"Total nodes in graph: {total_nodes}, Edges: {graph.edge_index.size(1)}")
    
    # Count elements in fuzzy labels
    fuzzy_label_counts = {k: len(v) for k, v in fuzzy_labels.items()}
    fuzzy_label_total = sum(fuzzy_label_counts.values())
    fuzzy_label_overlap = fuzzy_label_total - total_nodes
    #print(f"Fuzzy label counts: {fuzzy_label_counts}")
    #print(f"Total elements in fuzzy labels: {fuzzy_label_total}")
    print(f"Overlap in fuzzy labels: {fuzzy_label_overlap} nodes ({fuzzy_label_overlap/total_nodes:.2%} of graph)")
    
    # Count elements in fuzzy clusters
    fuzzy_cluster_counts = {k: len(v) for k, v in fuzzy_clusters.items()}
    fuzzy_cluster_total = sum(fuzzy_cluster_counts.values())
    fuzzy_cluster_overlap = fuzzy_cluster_total - total_nodes
    #print(f"Fuzzy cluster counts: {fuzzy_cluster_counts}")
    #print(f"Total elements in fuzzy clusters: {fuzzy_cluster_total}")
    print(f"Overlap in fuzzy clusters: {fuzzy_cluster_overlap} nodes ({fuzzy_cluster_overlap/total_nodes:.2%} of graph)")
    
    fuzzy_metrics = eval.fuzzy_clustering_metrics(fuzzy_labels, fuzzy_clusters)
    omega_index = fuzzy_metrics['omega_index'] if fuzzy_metrics['omega_index'] is not None else None
    print(f"Omega Index: {omega_index:.3f}" if omega_index is not None else "Omega Index: Not available")
    
    # Return a dictionary with all metrics instead of printing
    results = {
        "accuracy": metrics['accuracy'],
        "ARI": metrics['ARI'],
        "NMI": metrics['NMI'],
        "omega": omega_index
    }
    
    return results

if __name__ == "__main__":
    import os
    import numpy as np
    from collections import defaultdict
    
    # Hardcoded graph files and directory
    GRAPH_DIR = "path_to_dataset"
    GRAPH_FILES = [
        "i002c_v04_multi_21_chr10_0.pt",
        "i002c_v04_multi_21_chr10_1.pt",
        "i002c_v04_multi_21_chr10_2.pt",
        "i002c_v04_multi_21_chr19_0.pt",
        "i002c_v04_multi_21_chr19_1.pt",
        "i002c_v04_multi_21_chr19_2.pt",
        "i002c_v04_multi_21_chr15_0.pt",
        "i002c_v04_multi_21_chr15_1.pt",
        "i002c_v04_multi_21_chr15_2.pt",
        "i002c_v04_multi_21_chr22_0.pt",
        "i002c_v04_multi_21_chr22_1.pt",
        "i002c_v04_multi_21_chr22_2.pt"
    ]
    args = get_args()
    # Process all files and collect results
    results_by_chrom = defaultdict(list)
    
    for graph_file in GRAPH_FILES:
        graph_path = os.path.join(GRAPH_DIR, graph_file)
        
        # Parse chromosome and instance from filename
        parts = graph_file.split('_')
        chrom = parts[-2]  # e.g., 'chr10'
        instance = parts[-1].split('.')[0]  # e.g., '0'
        
        print(f"\nProcessing {chrom} (instance {instance})...")

        args.graph_path = graph_path
        # Run the main function and get results
        file_results = main(args)
        
        if file_results:
            # Store results by chromosome
            results_by_chrom[chrom].append(file_results)
            
            # Print individual file results
            print(f"\nResults for {chrom} (instance {instance}):")
            print(f"  Accuracy: {file_results['accuracy']:.3f}")
            print(f"  ARI: {file_results['ARI']:.3f}")
            print(f"  NMI: {file_results['NMI']:.3f}")
            print(f"  Omega Index: {file_results['omega']:.3f}" if file_results['omega'] is not None else "  Omega Index: Not available")
    # Calculate and print averages by chromosome
    print(f"\n=== SUMMARY OF RESULTS {args.mode} ===")
    print("\nAverages by chromosome:")
    
    # Lists to store all results in chromosome order
    all_accuracies = []
    all_aris = []
    all_nmis = []
    all_omegas = []
    
    for chrom, chrom_results in results_by_chrom.items():
        avg_accuracy = np.mean([r["accuracy"] for r in chrom_results])
        avg_ari = np.mean([r["ARI"] for r in chrom_results])
        avg_nmi = np.mean([r["NMI"] for r in chrom_results])
        
        # Handle None values in omega_index
        omega_values = [r["omega"] for r in chrom_results if r["omega"] is not None]
        avg_omega = np.mean(omega_values) if omega_values else None
        
        print(f"\n{chrom}:")
        print(f"  Accuracy: {avg_accuracy:.3f}")
        print(f"  ARI: {avg_ari:.3f}")
        print(f"  NMI: {avg_nmi:.3f}")
        print(f"  Omega Index: {avg_omega:.3f}" if avg_omega is not None else "  Omega Index: Not available")
        
        # Collect all individual results for this chromosome, rounded to 4 decimal places
        all_accuracies.extend([round(float(r["accuracy"]), 4) for r in chrom_results])
        all_aris.extend([round(float(r["ARI"]), 4) for r in chrom_results])
        all_nmis.extend([round(float(r["NMI"]), 4) for r in chrom_results])
        all_omegas.extend([round(float(r["omega"]), 4) if r["omega"] is not None else float('nan') for r in chrom_results])
    
    # Print lists of all results
    print("\n=== ALL RESULTS BY METRIC ===")
    print(f"All Accuracies: {all_accuracies}")
    print(f"All ARIs: {all_aris}")
    print(f"All NMIs: {all_nmis}")
    print(f"All Omega Indices: {all_omegas}")