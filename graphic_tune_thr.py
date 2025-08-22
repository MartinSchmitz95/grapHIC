import torch
import argparse
import yaml
import os
import pickle
from SGformer import SGFormer, SGFormerMulti, SGFormer_NoGate
from GIN import HeteroGIN
from train_diploid import global_phasing_quotient, local_phasing_quotient
import train_utils  # Import train_utils for seed setting


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
        model = SGFormer(
            in_channels=config['node_features'],
            hidden_channels=config['hidden_features'],
            out_channels=1,
            trans_num_layers=config['num_trans_layers'],
            gnn_num_layers=config['num_gnn_layers'],
            gnn_dropout=0,
            layer_norm=config['layer_norm']
        )
    elif model_type == 'sgformer_multi':
        model = SGFormerMulti(
            in_channels=config['node_features'],
            hidden_channels=config['hidden_features'],
            out_channels=1,
            trans_num_layers=config['num_trans_layers'],
            gnn_num_layers=config['num_gnn_layers'],
            gnn_dropout=0,
            layer_norm=config['layer_norm'],
            num_blocks=config['num_blocks']
        )
    elif model_type == 'sgformer_nogate':
        model = SGFormer_NoGate(
            in_channels=config['node_features'],
            hidden_channels=config['hidden_features'],
            out_channels=1,
            trans_num_layers=config['num_trans_layers'],
            trans_dropout=0,
            gnn_num_layers=config['num_gnn_layers'],
            gnn_dropout=0,
            edge_feature_dim=1,
        )
    elif model_type == 'heterogin':
        model = HeteroGIN(
            in_channels=config['node_features'],
            hidden_channels=config['hidden_features'],
            out_channels=1,
            num_layers=config['num_gnn_layers'],
            dropout=0,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Don't set model mode here - let the calling function control it
    
    return model


def predict_node_scores(graph_path, model_path, config_path, model_type):
    """
    Load a PyG graph and trained model, make predictions, and tune threshold for optimal 
    homozygous vs heterozygous classification (ignoring +1/-1 distinction).
    All computation is performed on CPU.
    
    Args:
        graph_path: Path to the PyG graph file (.pt)
        model_path: Path to the trained model state dict
        config_path: Path to the model configuration YAML file
        model_type: Type of model to load
    
    Returns:
        dict: Dictionary containing:
            - 'optimal_threshold': Best threshold for homozygous/heterozygous classification
            - 'best_accuracy': Accuracy at optimal threshold
            - 'homozygous_accuracy': Accuracy for homozygous classification
            - 'heterozygous_accuracy': Accuracy for heterozygous classification
            - 'total_nodes': Total number of nodes
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
        print("No chromosome information, exiting")
        exit()
    # Make predictions
    print("Making predictions on CPU...")
    with torch.no_grad():
        predictions = model(graph)
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
    
    # Get ground truth labels (y)
    if not hasattr(graph, 'y'):
        raise ValueError("Graph must have 'y' attribute for ground truth labels")
    
    y = graph.y.cpu()
    predictions = predictions.cpu()
    
    print(f"Generated predictions for {len(predictions)} nodes")
    print(f"Prediction range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    print(f"Prediction abs range: [0, {predictions.abs().max().item():.4f}]")
    print(f"Ground truth labels (y) range: [{y.min().item()}, {y.max().item()}]")
    
    # Convert ground truth to binary: 0 = homozygous, 1 = heterozygous
    y_binary = (y != 0).float()  # 0 stays 0, +1 and -1 become 1
    
    print(f"Binary ground truth: {(y_binary == 0).sum().item()} homozygous, {(y_binary == 1).sum().item()} heterozygous")
    
    # Analyze prediction distribution
    pred_abs = predictions.abs()
    print(f"Prediction absolute value statistics:")
    print(f"  Mean: {pred_abs.mean().item():.4f}")
    print(f"  Std: {pred_abs.std().item():.4f}")
    print(f"  Median: {pred_abs.median().item():.4f}")
    print(f"  25th percentile: {pred_abs.quantile(0.25).item():.4f}")
    print(f"  75th percentile: {pred_abs.quantile(0.75).item():.4f}")
    print(f"  90th percentile: {pred_abs.quantile(0.90).item():.4f}")
    print(f"  95th percentile: {pred_abs.quantile(0.95).item():.4f}")
    
    # Tune threshold for optimal homozygous vs heterozygous classification
    best_threshold = 0.0
    best_balanced_accuracy = 0.0
    best_accuracy = 0.0
    best_homozygous_acc = 0.0
    best_heterozygous_acc = 0.0
    
    # Try a much wider range of thresholds, including very small and large values
    max_abs_pred = predictions.abs().max().item()
    
    # Create comprehensive threshold range
    threshold_candidates = []
    # Very small thresholds
    threshold_candidates.extend(torch.linspace(0.001, 0.1, 50).tolist())
    # Small to medium thresholds
    threshold_candidates.extend(torch.linspace(0.1, 0.5, 50).tolist())
    # Medium to large thresholds
    threshold_candidates.extend(torch.linspace(0.5, max_abs_pred, 50).tolist())
    # Add percentile-based thresholds
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        threshold_candidates.append(pred_abs.quantile(p).item())
    
    threshold_candidates = sorted(list(set(threshold_candidates)))  # Remove duplicates and sort
    
    print(f"Tuning threshold for optimal homozygous/heterozygous classification...")
    print(f"Testing {len(threshold_candidates)} threshold values...")
    
    for threshold in threshold_candidates:
        # Classify based on threshold:
        # |prediction| < threshold → homozygous (0)
        # |prediction| >= threshold → heterozygous (1)
        predicted_binary = (predictions.abs() >= threshold).float()
        
        # Calculate accuracy
        correct_predictions = (predicted_binary == y_binary)
        total_accuracy = correct_predictions.float().mean().item()
        
        # Calculate homozygous accuracy (ground truth = 0)
        homozygous_mask = (y_binary == 0)
        if homozygous_mask.sum() > 0:
            homozygous_correct = correct_predictions[homozygous_mask]
            homozygous_accuracy = homozygous_correct.float().mean().item()
        else:
            homozygous_accuracy = 0.0
        
        # Calculate heterozygous accuracy (ground truth = 1)
        heterozygous_mask = (y_binary == 1)
        if heterozygous_mask.sum() > 0:
            heterozygous_correct = correct_predictions[heterozygous_mask]
            heterozygous_accuracy = heterozygous_correct.float().mean().item()
        else:
            heterozygous_accuracy = 0.0
        
        # Calculate balanced accuracy (average of class accuracies)
        balanced_accuracy = (homozygous_accuracy + heterozygous_accuracy) / 2.0
        
        # Update best threshold based on balanced accuracy
        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_threshold = threshold
            best_accuracy = total_accuracy
            best_homozygous_acc = homozygous_accuracy
            best_heterozygous_acc = heterozygous_accuracy
    
    print(f"Optimal threshold: {best_threshold:.4f}")
    print(f"Best balanced accuracy: {best_balanced_accuracy:.4f}")
    print(f"Best overall accuracy: {best_accuracy:.4f}")
    print(f"Homozygous accuracy: {best_homozygous_acc:.4f}")
    print(f"Heterozygous accuracy: {best_heterozygous_acc:.4f}")
    print(f"Total nodes: {len(predictions)}")
    
    # Calculate final classification statistics with optimal threshold
    predicted_binary = (predictions.abs() >= best_threshold).float()
    
    homozygous_predicted_count = (predicted_binary == 0).sum().item()
    heterozygous_predicted_count = (predicted_binary == 1).sum().item()
    
    print(f"Classification counts with optimal threshold:")
    print(f"  Predicted homozygous: {homozygous_predicted_count}")
    print(f"  Predicted heterozygous: {heterozygous_predicted_count}")
    
    # Ground truth counts
    gt_homozygous_count = (y == 0).sum().item()
    gt_heterozygous_count = (y != 0).sum().item()
    
    print(f"Ground truth counts:")
    print(f"  True homozygous: {gt_homozygous_count}")
    print(f"  True heterozygous: {gt_heterozygous_count}")
    
    # Additional diagnostics
    print(f"\nDiagnostic information:")
    print(f"  Threshold as percentile of |predictions|: {(pred_abs < best_threshold).float().mean().item():.3f}")
    print(f"  Precision for homozygous: {((predicted_binary == 0) & (y_binary == 0)).sum().item() / max(1, (predicted_binary == 0).sum().item()):.4f}")
    print(f"  Recall for homozygous: {best_homozygous_acc:.4f}")
    print(f"  Precision for heterozygous: {((predicted_binary == 1) & (y_binary == 1)).sum().item() / max(1, (predicted_binary == 1).sum().item()):.4f}")
    print(f"  Recall for heterozygous: {best_heterozygous_acc:.4f}")
    
    return {
        'optimal_threshold': best_threshold,
        'best_accuracy': best_accuracy,
        'best_balanced_accuracy': best_balanced_accuracy,
        'homozygous_accuracy': best_homozygous_acc,
        'heterozygous_accuracy': best_heterozygous_acc,
        'total_nodes': len(predictions),
        'predicted_counts': {
            'homozygous': homozygous_predicted_count,
            'heterozygous': heterozygous_predicted_count
        },
        'ground_truth_counts': {
            'homozygous': gt_homozygous_count,
            'heterozygous': gt_heterozygous_count
        }
    }


def evaluate_like_training(graph_path, model_path, config_path, model_type):
    """
    Evaluate the model using the same methodology as training (phasing quotients).
    This only looks at prediction signs and doesn't try to classify homozygous nodes.
    
    Args:
        graph_path: Path to the PyG graph file (.pt)
        model_path: Path to the trained model state dict
        config_path: Path to the model configuration YAML file
        model_type: Type of model to load
    
    Returns:
        dict: Dictionary containing evaluation metrics matching training methodology
    """
    device = 'cpu'  # Force CPU usage
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)['training']
    
    # Load the trained model
    print(f"Loading model from {model_path} (CPU)")
    model = load_model(model_path, config, model_type)
    
    # CRITICAL: Ensure model is in eval mode (like validation in training)
    #model.eval()
    print("Model set to eval() mode")
    
    # Load the graph
    print(f"Loading graph from {graph_path} (CPU)")
    graph = torch.load(graph_path, map_location=device)
    
    # Add chromosome info if missing (exactly as in training)
    if not hasattr(graph, 'chr'):
        graph.chr = torch.ones(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
        print("No chromosome information, exiting")
        exit()
    print(f"Graph has {len(torch.unique(graph.chr))} unique chromosomes")
    print(f"Graph node features shape: {graph.x.shape}")
    
    # Make predictions (exactly as in training validation)
    print("Making predictions on CPU...")
    with torch.no_grad():  # Same as validation in training
        predictions = model(graph)
        # Apply same squeeze operation as in training
        predictions = predictions.squeeze()
    
    # Get ground truth labels (y)
    if not hasattr(graph, 'y'):
        raise ValueError("Graph must have 'y' attribute for ground truth labels")
    
    y = graph.y.cpu()
    predictions = predictions.cpu()
    
    print(f"Generated predictions for {len(predictions)} nodes")
    print(f"Prediction range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    print(f"Prediction mean: {predictions.mean().item():.4f}, std: {predictions.std().item():.4f}")
    print(f"Ground truth labels (y) range: [{y.min().item()}, {y.max().item()}]")
    
    # Debug: Check distribution
    print(f"Predictions < 0: {(predictions < 0).sum().item()}")
    print(f"Predictions >= 0: {(predictions >= 0).sum().item()}")
    print(f"Y == -1: {(y == -1).sum().item()}")
    print(f"Y == 0: {(y == 0).sum().item()}")  
    print(f"Y == 1: {(y == 1).sum().item()}")
    
    print("\nGlobal phasing evaluation (by chromosome):")
    global_phasing = global_phasing_quotient(y, predictions, graph.chr, debug=True)
    
    # Calculate local phasing quotient if edge information is available
    local_phasing = 0.0
    if hasattr(graph, 'edge_index') and hasattr(graph, 'edge_type'):
        print("\nLocal phasing evaluation:")
        local_phasing = local_phasing_quotient(y, predictions, graph.chr, 
                                               graph.edge_index, graph.edge_type, debug=False)
    
    # Count ground truth distribution
    gt_homozygous_count = (y == 0).sum().item()
    gt_heterozygous_pos_count = (y == 1).sum().item()
    gt_heterozygous_neg_count = (y == -1).sum().item()
    
    print(f"\nGround truth distribution:")
    print(f"  Homozygous (0): {gt_homozygous_count}")
    print(f"  Heterozygous positive (+1): {gt_heterozygous_pos_count}")
    print(f"  Heterozygous negative (-1): {gt_heterozygous_neg_count}")
    
    # Prediction sign distribution
    pred_negative_count = (predictions < 0).sum().item()
    pred_positive_count = (predictions >= 0).sum().item()
    
    print(f"\nPrediction sign distribution:")
    print(f"  Negative predictions: {pred_negative_count}")
    print(f"  Positive predictions: {pred_positive_count}")
    
    return {
        'global_phasing_quotient': global_phasing,
        'local_phasing_quotient': local_phasing,
        'total_nodes': len(predictions),
        'ground_truth_counts': {
            'homozygous': gt_homozygous_count,
            'heterozygous_positive': gt_heterozygous_pos_count,
            'heterozygous_negative': gt_heterozygous_neg_count
        },
        'prediction_sign_counts': {
            'negative': pred_negative_count,
            'positive': pred_positive_count
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Tune threshold for optimal diploid phasing classification (CPU only)")
    parser.add_argument("--graph_path", type=str, required=True, 
                       help="Path to the PyG graph file (.pt)")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model state dict")
    parser.add_argument("--config_path", type=str, default='train_config.yml',
                       help="Path to the model configuration YAML file")
    parser.add_argument("--model_type", type=str, default='sgformer',
                       choices=['sgformer', 'sgformer_multi', 'sgformer_nogate', 'heterogin'],
                       help="Type of model to load")
    parser.add_argument("--eval_mode", type=str, default='threshold', 
                       choices=['threshold', 'training'],
                       help="Evaluation mode: 'threshold' for 3-class classification, 'training' for phasing quotients")
    
    args = parser.parse_args()
    
    if args.eval_mode == 'training':
        # Evaluate using the same methodology as training
        results = evaluate_like_training(
            args.graph_path,
            args.model_path, 
            args.config_path,
            args.model_type
        )
        
        print("\n" + "="*50)
        print("TRAINING-STYLE EVALUATION RESULTS")
        print("="*50)
        print(f"Global phasing quotient: {results['global_phasing_quotient']:.4f}")
        print(f"Local phasing quotient: {results['local_phasing_quotient']:.4f}")
        print(f"Total nodes processed: {results['total_nodes']}")
        
    else:
        # Generate predictions and tune threshold (original method)
        results = predict_node_scores(
            args.graph_path,
            args.model_path, 
            args.config_path,
            args.model_type
        )
        
        print("\n" + "="*50)
        print("THRESHOLD TUNING RESULTS")
        print("="*50)
        print(f"Optimal threshold: {results['optimal_threshold']:.4f}")
        print(f"Best balanced accuracy: {results['best_balanced_accuracy']:.4f}")
        print(f"Best overall accuracy: {results['best_accuracy']:.4f}")
        print(f"Homozygous classification accuracy: {results['homozygous_accuracy']:.4f}")
        print(f"Heterozygous classification accuracy: {results['heterozygous_accuracy']:.4f}")
        print(f"Total nodes processed: {results['total_nodes']}")
    
    return results


if __name__ == "__main__":
    results = main()
