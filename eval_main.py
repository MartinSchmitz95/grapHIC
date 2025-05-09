import torch
import argparse
import yaml
from skfuzzy.cluster import cmeans
from sklearn.cluster import KMeans
import eval
import baselines

def get_fuzzy_clustering_from_predictions(preds, fuzzy_thr=0.2):
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

def get_fuzzy_clustering_from_embeddings(embs, n_haps=2, fuzziness=2, overlap_threshold=0.5):

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
        # Get indices where membership value exceeds threshold
        cluster_members = [j for j in range(len(embs)) if u[i, j] >= overlap_threshold]
        cluster_sets[i] = cluster_members
    
    return cluster_sets

def get_hard_clustering_from_embeddings(embs, n_haps=2, random_state=42):
    
    # Perform hard KMeans clustering
    kmeans = KMeans(n_clusters=n_haps, random_state=random_state)
    pred_labels = kmeans.fit_predict(embs)
    
    # Return list of cluster assignments
    return pred_labels.tolist()

def run_model(model_type, model_path, graph, config):
    
    if model_type == "mrl":
        from SGformer import SGFormer
        model = SGFormer(in_channels=config['node_features'], 
                         hidden_channels=config['hidden_features'], 
                         out_channels=1, 
                         trans_num_layers=config['num_trans_layers'], 
                         gnn_num_layers=config['num_gnn_layers_overlap'], 
                         gnn_dropout=config['gnn_dropout'],
                         layer_norm=config['layer_norm'])

    elif model_type == "contrastive":
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
        )

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Prepare graph features
    graph.x = torch.abs(graph.y).float().unsqueeze(1)
    
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate phasing clustering on synthetic data")
    parser.add_argument("--model_path", required=True, type=str, help="Path to trained model")
    parser.add_argument("--graph_path", required=True, type=str, help="Path to graph file")
    parser.add_argument("--output_dir", type=str, default="/home/schmitzmf/scratch/cluster_results/test", help="Output directory for results")
    parser.add_argument("--mode", type=str, default="mrl", help="mode")
    args = parser.parse_args()

    print(f"Loading graph from {args.graph_path}")

    graph = torch.load(args.graph_path)
    
    # Divide edge weight by 10 for edges with type==1
    if hasattr(graph, 'edge_type') and hasattr(graph, 'edge_weight'):
        # Count occurrences of each edge type
        edge_type_counts = {}
        for edge_type in graph.edge_type:
            edge_type_int = edge_type.item()
            edge_type_counts[edge_type_int] = edge_type_counts.get(edge_type_int, 0) + 1
        print(f"Edge type counts: {edge_type_counts}")
        mask = (graph.edge_type == 0)
        graph.edge_weight[mask] = graph.edge_weight[mask] / 10.0
    else:
        print("No edge type or edge attribute found in graph")
        exit()
        
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
    with open('train_config.yml') as file:
        config = yaml.safe_load(file)['training']

    print("Computing predictions and clusters...")
    if args.mode == "mrl":
        preds = run_model('mrl',args.model_path, graph, config)
        
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
    elif args.mode == "contrastive":
        embeddings = run_model('contrastive', args.model_path, graph, config)
        hetero_embeddings = embeddings[mask]
        hard_clusters = get_hard_clustering_from_embeddings(hetero_embeddings, n_haps=2)
        fuzzy_clusters = get_fuzzy_clustering_from_embeddings(embeddings, n_haps=2)
    elif args.mode == "spectral":
        cluster_result = baselines.spectral_clustering(graph)
        print(cluster_result)
        # Convert list to tensor, apply mask, then convert back to list
        cluster_result_tensor = torch.tensor(cluster_result)
        hard_clusters = cluster_result_tensor[mask].tolist()
        fuzzy_clusters = hard_to_fuzzy(cluster_result)
    elif args.mode == "louvain":
        cluster_result = baselines.louvain_clustering(graph)
        print(cluster_result)
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
    print(f"Total nodes in graph: {total_nodes}")
    
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
    print(f"Omega Index: {fuzzy_metrics['omega_index']:.3f}" if fuzzy_metrics['omega_index'] is not None else "Omega Index: Not available")

if __name__ == "__main__":
    main()