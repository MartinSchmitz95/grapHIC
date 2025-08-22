import torch
import argparse
import yaml
import os
import pickle
import gzip
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from SGformer import SGFormer, SGFormerMulti, SGFormer_NoGate, SGFormerGINEdgeEmbs, SGFormerEdgeEmbs
from GIN import HeteroGIN


def load_model(model_path, config, model_type):
    """
    Load a trained model based on the specified type and configuration.
    
    Args:
        model_path: Path to the saved model state dict
        config: Model configuration dictionary
        model_type: Type of model ('sgformer', 'sgformer_multi', 'sgformer_nogate', 'heterogin')
    
    Returns:
        Loaded model (on CPU)
    """
    device = 'cpu'  # Force CPU usage
    
    if model_type == 'sgformer':
        print("hiii")
        model = SGFormerGINEdgeEmbs(in_channels=config['node_features'], hidden_channels=config['hidden_features'],
                          out_channels=1, trans_num_layers=config['num_trans_layers'],
                          gnn_num_layers=config['num_gnn_layers'], gnn_dropout=config['gnn_dropout'],
                            norm=config['norm'], direct_ftrs=config['direct_ftrs']).to(device)
        
    elif model_type == 'sgformer_multi':
        model = SGFormerMulti(
            in_channels=config['node_features'],
            hidden_channels=config['hidden_features'],
            out_channels=1,
            trans_num_layers=config['num_trans_layers'],
            gnn_num_layers=config['num_gnn_layers'],
            gnn_dropout=config['gnn_dropout'],
            norm=config['norm'],
            num_blocks=config['num_blocks'],
            direct_ftrs=config['direct_ftrs']
        )
    elif model_type == 'sgformer_nogate':
        model = SGFormer_NoGate(
            in_channels=config['node_features'],
            hidden_channels=config['hidden_features'],
            out_channels=1,
            trans_num_layers=config['num_trans_layers'],
            trans_dropout=0,
            gnn_num_layers=config['num_gnn_layers'],
            gnn_dropout=config['gnn_dropout'],
            norm=config['norm'],
            direct_ftrs=config['direct_ftrs']
        )
    elif model_type == 'heterogin':
        model = HeteroGIN(
            in_channels=config['node_features'],
            hidden_channels=config['hidden_features'],
            out_channels=1,
            num_layers=config['num_gnn_layers'],
            dropout=config['gnn_dropout'],
            direct_ftrs=config['direct_ftrs']
        )
    elif model_type == 'sg_gin':
        model = SGFormerGINEdgeEmbs(in_channels=config['node_features'], hidden_channels=config['hidden_features'],
                          out_channels=1, trans_num_layers=config['num_trans_layers'],
                          gnn_num_layers=config['num_gnn_layers'], gnn_dropout=config['gnn_dropout'],
                            norm=config['norm'], direct_ftrs=config['direct_ftrs']).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def predict_node_scores(graph_path, model_path, config_path, model_type):
    """
    Load a PyG graph and trained model, make predictions, and return node scores.
    All computation is performed on CPU.
    
    Args:
        graph_path: Path to the PyG graph file (.pt)
        model_path: Path to the trained model state dict
        config_path: Path to the model configuration YAML file
        model_type: Type of model to load
    
    Returns:
        dict: Dictionary with node indices as keys and prediction scores as values
    """
    device = 'cpu'  # Force CPU usage
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)['training']
    
    # Load the trained model
    print(f"Loading model from {model_path} (CPU)")
    model = load_model(model_path, config, model_type)
    
    # Load the graph
    print(f"Loading graph from {graph_path} (CPU)")
    graph = torch.load(graph_path, map_location=device)
    
    # Add chromosome info if missing (as done in train_diploid.py)
    if not hasattr(graph, 'chr'):
        graph.chr = torch.ones(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
    
    # Make predictions
    print("Making predictions on CPU...")
    with torch.no_grad():
        predictions = model(graph)
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
    
    # Convert to dictionary
    node_scores = {}
    for node_id in range(len(predictions)):
        node_scores[node_id] = predictions[node_id].item()
    
    print(f"Generated predictions for {len(node_scores)} nodes")
    print(f"Prediction range: [{min(node_scores.values()):.4f}, {max(node_scores.values()):.4f}]")
    
    return node_scores


def create_haplotype_fastas(reads_path, predictions_dict, epsilon=0.1, output_dir=None):
    """
    Create haplotype-specific FASTA files from model predictions.
    
    Args:
        reads_path: Path to the reads FASTA file (can be .gz)
        predictions_dict: Dictionary mapping node IDs to prediction scores
        epsilon: Threshold for haplotype assignment (default: 0.1)
        output_dir: Directory to save output files (defaults to same directory as reads_path)
    
    Returns:
        Dictionary with haplotype file paths and statistics
    """
    
    if output_dir is None:
        output_dir = os.path.dirname(reads_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Creating haplotype FASTA files from {reads_path} with epsilon={epsilon}")
    
    # Check if reads file exists
    if os.path.exists(reads_path + '.gz') and not os.path.exists(reads_path):
        reads_path += '.gz'
    elif not os.path.exists(reads_path):
        raise FileNotFoundError(f"Could not find reads file at {reads_path} or {reads_path}.gz")
    
    # First, compute statistics based on predictions only (no sequence loading needed)
    print("Computing haplotype assignment statistics...")
    
    stats = {
        'total_nodes': len(predictions_dict),
        'hap1_only': 0,
        'hap2_only': 0,
        'both_haps': 0
    }
    
    # Compute statistics from predictions
    for node_id, score in predictions_dict.items():
        # Determine which haplotype(s) this node belongs to based on epsilon threshold
        in_hap1 = score > -epsilon
        in_hap2 = score < epsilon
        
        if in_hap1 and in_hap2:
            # In both (ambiguous region)
            stats['both_haps'] += 1
        elif in_hap1:
            # Only in hap1
            stats['hap1_only'] += 1
        elif in_hap2:
            # Only in hap2
            stats['hap2_only'] += 1
    
    # Print statistics first
    print(f"\nHaplotype Assignment Statistics (based on predictions):")
    print(f"======================================================")
    print(f"Epsilon threshold: {epsilon}")
    print(f"Total nodes with predictions: {stats['total_nodes']}")
    print(f"")
    print(f"Haplotype assignments:")
    print(f"  Hap1 only (score > {epsilon}): {stats['hap1_only']}")
    print(f"  Hap2 only (score < -{epsilon}): {stats['hap2_only']}")
    print(f"  Both haps (|score| <= {epsilon}): {stats['both_haps']}")
    print(f"")
    
    # Now load read sequences for file creation
    print(f"Loading read sequences from {reads_path}")
    read_seqs = {}
    
    if reads_path.endswith('.gz'):
        with gzip.open(reads_path, 'rt') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                read_seqs[record.id] = str(record.seq)
    else:
        for record in SeqIO.parse(reads_path, 'fasta'):
            read_seqs[record.id] = str(record.seq)
    
    print(f"Loaded {len(read_seqs)} read sequences")
    
    # Create records for each haplotype and track mapping statistics
    hap1_records = []  # scores > -epsilon
    hap2_records = []  # scores < epsilon
    
    # Add mapping statistics to track which reads have corresponding predictions
    stats.update({
        'total_reads': len(read_seqs),
        'mapped_reads': 0,
        'unmapped_reads': 0
    })
    
    for read_id, sequence in read_seqs.items():
        # Read ID is directly the node ID
        try:
            node_id = int(read_id)
        except ValueError:
            # If read_id is not a simple integer, try to extract node ID from it
            stats['unmapped_reads'] += 1
            continue
        
        if node_id not in predictions_dict:
            stats['unmapped_reads'] += 1
            continue
        
        score = predictions_dict[node_id]
        stats['mapped_reads'] += 1
        
        # Create description with score information
        desc = f"score={score:.4f} node_id={node_id} epsilon={epsilon}"
        
        # Create SeqRecord
        record = SeqRecord(
            seq=Seq(sequence),
            id=read_id,
            description=desc
        )
        
        # Determine which haplotype files to add to based on epsilon threshold
        in_hap1 = score > -epsilon
        in_hap2 = score < epsilon
        
        if in_hap1:
            hap1_records.append(record)
        if in_hap2:
            hap2_records.append(record)
    
    # Print mapping statistics
    print(f"\nRead Mapping Statistics:")
    print(f"========================")
    print(f"Total reads in FASTA: {stats['total_reads']}")
    print(f"Reads with mapped predictions: {stats['mapped_reads']}")
    print(f"Reads without mapped predictions: {stats['unmapped_reads']}")
    print(f"")
    
    # Generate output file names based on input file
    base_name = os.path.splitext(os.path.splitext(os.path.basename(reads_path))[0])[0]  # Remove .fasta.gz or .fasta
    hap1_file = os.path.join(output_dir, f'{base_name}_hap1.fasta.gz')
    hap2_file = os.path.join(output_dir, f'{base_name}_hap2.fasta.gz')
    
    print(f"Writing output files:")
    print(f"  Hap1: {hap1_file} ({len(hap1_records)} sequences)")
    print(f"  Hap2: {hap2_file} ({len(hap2_records)} sequences)")
    
    with gzip.open(hap1_file, 'wt') as handle:
        SeqIO.write(hap1_records, handle, 'fasta')
    
    with gzip.open(hap2_file, 'wt') as handle:
        SeqIO.write(hap2_records, handle, 'fasta')
    
    return {
        'hap1_file': hap1_file,
        'hap2_file': hap2_file,
        'stats': stats,
        'epsilon': epsilon
    }


def analyze_prediction_distribution(node_scores):
    """
    Analyze the distribution of node prediction scores.
    
    Args:
        node_scores: Dictionary with node indices as keys and prediction scores as values
    
    Returns:
        Dictionary with analysis results
    """
    scores = list(node_scores.values())
    
    print(f"\nPrediction Distribution Analysis:")
    print(f"================================")
    print(f"Total nodes: {len(scores)}")
    print(f"Score range: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"Mean: {np.mean(scores):.4f}")
    print(f"Std: {np.std(scores):.4f}")
    print(f"Median: {np.median(scores):.4f}")
    
    # Create 0.1 buckets from -1 to 1
    buckets = np.arange(-1.0, 1.1, 0.1)
    bucket_counts = np.zeros(len(buckets) - 1)
    bucket_labels = []
    
    # Count nodes in each bucket
    for i in range(len(buckets) - 1):
        lower = buckets[i]
        upper = buckets[i + 1]
        count = sum(1 for score in scores if lower <= score < upper)
        bucket_counts[i] = count
        bucket_labels.append(f"[{lower:.1f}, {upper:.1f})")
    
    # Print distribution
    print(f"\nDistribution in 0.1 buckets:")
    print(f"{'Bucket':<15} {'Count':<8} {'Percentage':<12}")
    print("-" * 35)
    
    total_nodes = len(scores)
    for i, (label, count) in enumerate(zip(bucket_labels, bucket_counts)):
        percentage = (count / total_nodes) * 100
        print(f"{label:<15} {int(count):<8} {percentage:>8.2f}%")
    
    # Additional insights
    print(f"\nKey Insights:")
    print(f"=============")
    
    # Count nodes in different regions
    hap1_only = sum(1 for score in scores if score > 0.1)
    hap2_only = sum(1 for score in scores if score < -0.1)
    ambiguous = sum(1 for score in scores if -0.1 <= score <= 0.1)
    
    print(f"Nodes with score > 0.1 (likely Hap1): {hap1_only} ({hap1_only/total_nodes*100:.1f}%)")
    print(f"Nodes with score < -0.1 (likely Hap2): {hap2_only} ({hap2_only/total_nodes*100:.1f}%)")
    print(f"Nodes with |score| <= 0.1 (ambiguous): {ambiguous} ({ambiguous/total_nodes*100:.1f}%)")
    
    # Find the bucket with most nodes
    max_bucket_idx = np.argmax(bucket_counts)
    print(f"Most populated bucket: {bucket_labels[max_bucket_idx]} with {int(bucket_counts[max_bucket_idx])} nodes")
    
    # Check for bimodality
    positive_scores = [s for s in scores if s > 0]
    negative_scores = [s for s in scores if s < 0]
    
    print(f"Positive scores: {len(positive_scores)} ({len(positive_scores)/total_nodes*100:.1f}%)")
    print(f"Negative scores: {len(negative_scores)} ({len(negative_scores)/total_nodes*100:.1f}%)")
    print(f"Zero scores: {sum(1 for s in scores if s == 0)}")
    
    return {
        'scores': scores,
        'bucket_counts': bucket_counts,
        'bucket_labels': bucket_labels,
        'hap1_only': hap1_only,
        'hap2_only': hap2_only,
        'ambiguous': ambiguous
    }


def main():
    parser = argparse.ArgumentParser(description="Generate node predictions from trained diploid phasing model (CPU only)")
    parser.add_argument("--graph_path", type=str,
                       help="Path to the PyG graph file (.pt). Can be auto-constructed from --dataset_path and --genome_str")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model state dict")
    parser.add_argument("--config_path", type=str, default='train_config.yml',
                       help="Path to the model configuration YAML file")
    parser.add_argument("--model_type", type=str, default='sgformer',
                       choices=['sgformer', 'sgformer_multi', 'sgformer_nogate', 'heterogin', 'sg_gin'],
                       help="Type of model to load")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Optional path to save predictions as a pickle file (.pkl)")
    
    # Haplotype FASTA creation arguments
    parser.add_argument("--create_haplotype_fastas", action='store_true',
                       help="Create haplotype-specific FASTA files from predictions")
    parser.add_argument("--reads_path", type=str,
                       help="Path to the reads FASTA file (can be .gz)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                       help="Epsilon threshold for haplotype assignment (default: 0.1)")
    parser.add_argument("--haplotype_output_dir", type=str,
                       help="Output directory for haplotype FASTA files (default: same directory as reads file)")
    
    # Analysis arguments
    parser.add_argument("--no_distribution_analysis", action='store_true',
                       help="Skip the prediction distribution analysis")
    
    args = parser.parse_args()
    
    # Validate that we have a graph_path
    if not args.graph_path:
        print("Error: Must specify --graph_path")
        return None
    
    if not os.path.exists(args.graph_path):
        print(f"Error: Graph file not found at {args.graph_path}")
        return None
    
    # Generate predictions (always on CPU)
    node_scores = predict_node_scores(
        args.graph_path,
        args.model_path, 
        args.config_path,
        args.model_type
    )
    
    # Optionally save predictions to file
    if args.output_path:
        with open(args.output_path, 'wb') as f:
            pickle.dump(node_scores, f)
        print(f"Saved predictions dictionary to {args.output_path}")
    
    # Optionally create haplotype FASTA files
    if args.create_haplotype_fastas:
        if not args.reads_path:
            print("Error: --reads_path is required when --create_haplotype_fastas is specified")
            return None
        
        try:
            result = create_haplotype_fastas(
                reads_path=args.reads_path,
                predictions_dict=node_scores,
                epsilon=args.epsilon,
                output_dir=args.haplotype_output_dir
            )
            print(f"\nSuccess! Created haplotype FASTA files:")
            print(f"  Hap1: {result['hap1_file']}")
            print(f"  Hap2: {result['hap2_file']}")
        except Exception as e:
            print(f"Error creating haplotype FASTA files: {e}")
            print("Continuing with predictions only...")
    
    # Analyze prediction distribution
    if not args.no_distribution_analysis and node_scores:
        analyze_prediction_distribution(node_scores)
    
    return node_scores


if __name__ == "__main__":
    predictions = main()
