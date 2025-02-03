import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pickle
from Bio import SeqIO
from SGformer_HG import SGFormer
import gzip
def load_model(checkpoint_path, config):
    """Load the SGFormer model with specified configuration."""
    model = SGFormer(
        in_channels=config['node_features'],
        hidden_channels=config['hidden_features'],
        out_channels=1,
        trans_num_layers=config['num_trans_layers'],
        gnn_num_layers=config['num_gnn_layers'],
        gnn_dropout=config['gnn_dropout']
    ).to('cpu')
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model

def analyze_predictions(predictions, bins=None):
    """Create histogram of predictions and print bin information."""
    if bins is None:
        bins = np.arange(-2, 2.1, 0.1)  # Create bins from -2 to 2 with 0.1 steps
    
    # Create histogram
    hist, bin_edges = np.histogram(predictions, bins=bins)
    
    # Print bin information
    print("\nPrediction distribution:")
    print("------------------------")
    for i in range(len(hist)):
        bin_start = f"{bin_edges[i]:6.1f}"
        bin_end = f"{bin_edges[i+1]:6.1f}"
        count = hist[i]
        percentage = (count / len(predictions)) * 100
        print(f"Bin [{bin_start}, {bin_end}): {count:5d} samples ({percentage:5.1f}%)")
    
    # Create and save histogram plot
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=bins, edgecolor='black')
    plt.title('Distribution of Predictions')
    plt.xlabel('Prediction Value')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_histogram.png')
    plt.close()

def create_unitig_info_dict(graph, predictions, utg_dict):
    """
    Create a dictionary mapping unitig IDs to their information tuples.
    
    Args:
        graph: PyG graph object containing node features
        predictions: Model predictions for each node
        utg_dict: Dictionary mapping unitig IDs to node pairs
    
    Returns:
        Dictionary with unitig IDs as keys and (new_node_id, prediction, chr, pog_median) as values
    """
    unitig_info = {}
    
    for utg_id, (node1, node2) in utg_dict.items():
        # Calculate new_node_id as node_id[1]//2
        new_node_id = node2 // 2
        
        # Get prediction for this node
        pred = predictions[new_node_id]
        
        # Get chromosome and pog_median from node features
        chr_val = graph.chr[new_node_id].item()
        pog_median = graph.x[new_node_id, 2].item()  # Assuming pog_median is first feature
        
        # Store tuple in dictionary
        unitig_info[utg_id] = (new_node_id, pred, chr_val, pog_median)
    
    return unitig_info

def create_target_fastas(unitig_info, unitig_seqs):
    """
    Create two gzipped FASTA files (pos.fasta.gz and neg.fasta.gz) based on predictions.
    
    Args:
        unitig_info: Dictionary with unitig IDs mapping to (new_node_id, pred, chr, pog_median)
        unitig_seqs: Dictionary with unitig IDs mapping to sequences
    """
    pos_records = []
    neg_records = []
    
    for utg_id, (_, pred, _, pog_median) in unitig_info.items():
        if utg_id not in unitig_seqs:
            continue
            
        # Create SeqRecord object
        record = SeqIO.SeqRecord(
            seq=SeqIO.Seq(unitig_seqs[utg_id]),
            id=utg_id,
            description=f"pred={pred:.3f} pog_median={pog_median:.3f}"
        )
        
        # Add to both files if pog_median < 1.5
        if pog_median < 1.5:
            pos_records.append(record)
            neg_records.append(record)
        # Otherwise split based on prediction
        elif pred > 0:
            pos_records.append(record)
        else:  # pred <= 0
            neg_records.append(record)
    
    # Write to gzipped FASTA files
    with gzip.open("pos.fasta.gz", "wt") as handle:
        SeqIO.write(pos_records, handle, "fasta")
    
    with gzip.open("neg.fasta.gz", "wt") as handle:
        SeqIO.write(neg_records, handle, "fasta")
    
    # Print statistics
    print("\nFASTA Creation Statistics:")
    print("-------------------------")
    print(f"Sequences in pos.fasta.gz: {len(pos_records)}")
    print(f"Sequences in neg.fasta.gz: {len(neg_records)}")

def check_threshold(graph):
    """
    Analyze pile-o-gram features and find optimal threshold between O and E nodes.
    
    Args:
        graph: PyG graph object containing node features
    
    Returns:
        float: optimal threshold value
    """
    # Convert to numpy for easier analysis
    pog_medians = graph.x[:, 2].numpy()  # Assuming pog_median is third feature
    true_labels = (graph.y != 0).numpy().astype(int)  # 0 for O-nodes (y=0), 1 for E-nodes (y=±1)
    
    # Print basic statistics
    print("\nPile-O-Gram Analysis:")
    print("---------------------")
    
    # Check if we have both types of nodes
    o_nodes = pog_medians[true_labels == 0]
    e_nodes = pog_medians[true_labels == 1]
    
    if len(o_nodes) == 0:
        print("Warning: No O-nodes found in the graph")
        print(f"E-nodes range: {np.min(e_nodes):.3f} to {np.max(e_nodes):.3f}")
        exit()
    elif len(e_nodes) == 0:
        print(f"O-nodes range: {np.min(o_nodes):.3f} to {np.max(o_nodes):.3f}")
        print("Warning: No E-nodes found in the graph")
        exit()
    else:
        print(f"O-nodes range: {np.min(o_nodes):.3f} to {np.max(o_nodes):.3f}")
        print(f"E-nodes range: {np.min(e_nodes):.3f} to {np.max(e_nodes):.3f}")
    
    # Test different thresholds
    thresholds = np.linspace(0.1, 2.1, 100)
    best_accuracy = 0
    best_threshold = 1.5  # Default threshold
    
    for threshold in thresholds:
        predicted_labels = (pog_medians > threshold).astype(int)
        accuracy = np.mean(predicted_labels == true_labels)
        
        # Calculate precision and recall
        true_positives = np.sum((predicted_labels == 1) & (true_labels == 1))
        false_positives = np.sum((predicted_labels == 1) & (true_labels == 0))
        false_negatives = np.sum((predicted_labels == 0) & (true_labels == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    print(f"\nThreshold Analysis Results:")
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Precision at best threshold: {best_precision:.4f}")
    print(f"Recall at best threshold: {best_recall:.4f}")
    
    return best_threshold

def main():
    parser = argparse.ArgumentParser(description="Inference script for SGFormer model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--pyg_path", type=str, required=True, help="Path to the PYG graph")
    parser.add_argument("--utg_dict_path", type=str, required=True, help="Path to the UTG to node dictionary")
    parser.add_argument("--config_path", type=str, default='train_config.yml', help="Path to model config file")
    parser.add_argument("--fasta_path", type=str, required=True, help="Path to the FASTA file")
    
    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config_path) as file:
        full_config = yaml.safe_load(file)
        config = full_config['training']

    # Load model
    model = load_model(args.model_path, config)
    
    # Load graph
    graph = torch.load(args.pyg_path)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(graph).squeeze().numpy()
    
    # Load UTG to node dictionary
    with open(args.utg_dict_path, 'rb') as f:
        utg_dict = pickle.load(f)
    
    # Create unitig info dictionary
    unitig_info = create_unitig_info_dict(graph, predictions, utg_dict)
    
    # Check threshold for pile-o-gram features
    threshold = check_threshold(graph)
    
    # Load FASTA sequences
    unitig_seqs = {record.name: str(record.seq) for record in SeqIO.parse(gzip.open(args.fasta_path, "rt"), "fasta")}
    
    # Create target FASTA files
    #create_target_fastas(unitig_info, unitig_seqs)
    
    # Check for missing unitigs
    missing_unitigs = []
    for unitig_id in unitig_seqs.keys():
        if unitig_id not in utg_dict:
            missing_unitigs.append(unitig_id)
    
    # Print results
    print("\nFASTA and UTG Dictionary Comparison:")
    print("-----------------------------------")
    print(f"Total unitigs in FASTA: {len(unitig_seqs)}")
    print(f"Total unitigs in UTG dict: {len(utg_dict)}")
    
    if missing_unitigs:
        print(f"\nWARNING: Found {len(missing_unitigs)} unitigs in FASTA that are missing from UTG dictionary:")
        for unitig in missing_unitigs[:10]:  # Print first 10 missing unitigs
            print(f"  - {unitig}")
        if len(missing_unitigs) > 10:
            print(f"  ... and {len(missing_unitigs) - 10} more")
    else:
        print("\nAll FASTA unitigs are present in the UTG dictionary")
    
    # Analyze and visualize predictions
    analyze_predictions(predictions)
    
    # Print some basic statistics
    print("\nPrediction Statistics:")
    print("---------------------")
    print(f"Mean: {predictions.mean():.3f}")
    print(f"Std: {predictions.std():.3f}")
    print(f"Min: {predictions.min():.3f}")
    print(f"Max: {predictions.max():.3f}")
    print(f"Total nodes: {len(predictions)}")

if __name__ == "__main__":
    main()
