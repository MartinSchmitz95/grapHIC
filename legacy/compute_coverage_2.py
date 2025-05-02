import subprocess
import gzip
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import networkx as nx
import pickle

def decompress_file(input_file, output_file):
    """Decompress a gzipped file to a temporary file."""
    with gzip.open(input_file, "rt") as infile, open(output_file, "w") as outfile:
        for line in infile:
            outfile.write(line)

def run_jellyfish(reads_file, kmer_size, output_file):
    """Run Jellyfish to compute k-mer frequencies."""
    decompressed_file = "decompressed_reads.fasta"

    # Decompress the file if it is gzipped
    if reads_file.endswith(".gz"):
        print(f"Decompressing {reads_file}...")
        decompress_file(reads_file, decompressed_file)
        reads_file = decompressed_file

    jellyfish_count_cmd = [
        "jellyfish", "count",
        "-C", "-m", str(kmer_size), "-s", "1G", "-t", "8",
        reads_file, "-o", "mer_counts.jf"
    ]
    subprocess.run(jellyfish_count_cmd, check=True)

    jellyfish_dump_cmd = [
        "jellyfish", "dump", "-c", "mer_counts.jf"
    ]
    with open(output_file, "w") as out:
        subprocess.run(jellyfish_dump_cmd, check=True, stdout=out)

    # Clean up the decompressed file
    if os.path.exists(decompressed_file):
        os.remove(decompressed_file)

def calculate_read_coverage(reads_file, kmer_hist_file, output_file, kmer_size):
    """Estimate coverage per read using k-mer frequencies."""
    # Load k-mer counts into a dictionary
    kmer_counts = {}
    with open(kmer_hist_file, "r") as f:
        for line in f:
            kmer, count = line.strip().split()
            kmer_counts[kmer] = int(count)

    # Parse reads and calculate coverage statistics per read
    read_coverage = {}
    with gzip.open(reads_file, "rt") if reads_file.endswith(".gz") else open(reads_file, "r") as f:
        read_id = None
        read_sequence = []

        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Process previous read
                if read_id is not None:
                    read_seq = "".join(read_sequence)
                    coverage_stats = calculate_kmer_coverage(read_seq, kmer_counts, kmer_size)
                    read_coverage[read_id] = coverage_stats

                # Start a new read - take only the first part before space
                read_id = line[1:].split()[0]
                read_sequence = []
            else:
                read_sequence.append(line)

        # Process the last read
        if read_id is not None:
            read_seq = "".join(read_sequence)
            coverage_stats = calculate_kmer_coverage(read_seq, kmer_counts, kmer_size)
            read_coverage[read_id] = coverage_stats

    # Write read coverage statistics to output file
    with open(output_file, "w") as out:
        # Write CSV header
        out.write("read_id,median_coverage,variance,max_coverage,filtered_avg,filtered_median\n")
        for read_id, stats in read_coverage.items():
            out.write(f"{read_id},{stats['median']:.2f},{stats['variance']:.2f},{stats['max']:.2f},{stats['filtered_avg']:.2f},{stats['filtered_median']:.2f}\n")

def calculate_kmer_coverage(read_sequence, kmer_counts, kmer_size, coverage=20):
    """Calculate coverage statistics (median, variance, max) for a read."""
    if len(read_sequence) < kmer_size:
        return {'median': 0, 'variance': 0, 'max': 0, 'filtered_avg': 0}

    # Collect coverage values for all k-mers, excluding zero coverage k-mers
    kmer_coverages = []
    for i in range(len(read_sequence) - kmer_size + 1):
        kmer = read_sequence[i:i + kmer_size]
        coverage = kmer_counts.get(kmer, 0)
        if coverage > 0:  # Only include k-mers with non-zero coverage
            kmer_coverages.append(coverage)

    if not kmer_coverages:
        return {'median': 0, 'variance': 0, 'max': 0, 'filtered_avg': 0}

    # Calculate statistics
    sorted_coverages = sorted(kmer_coverages)
    n = len(sorted_coverages)
    
    # Calculate median
    if n % 2 == 0:
        median = (sorted_coverages[n//2 - 1] + sorted_coverages[n//2]) / 2
    else:
        median = sorted_coverages[n//2]

    # Calculate variance
    mean = sum(kmer_coverages) / n
    variance = sum((x - mean) ** 2 for x in kmer_coverages) / n

    # Get maximum
    max_coverage = max(kmer_coverages)

    # Calculate filtered average (excluding kmers with coverage > 3x coverage)
    coverage_threshold = 2.1 * coverage
    filtered_coverages = [cov for cov in kmer_coverages if cov <= coverage_threshold]
    filtered_avg = sum(filtered_coverages) / len(filtered_coverages) if filtered_coverages else 0

    # Calculate filtered median (using same threshold)
    filtered_median = 0
    if filtered_coverages:
        filtered_coverages.sort()
        n = len(filtered_coverages)
        if n % 2 == 0:
            filtered_median = (filtered_coverages[n//2 - 1] + filtered_coverages[n//2]) / 2
        else:
            filtered_median = filtered_coverages[n//2]
        if n ==0:
            filtered_median = 5 * coverage

    return {
        'median': median,
        'variance': variance,
        'max': max_coverage,
        'filtered_avg': filtered_avg,
        'filtered_median': filtered_median
    }

def analyze_coverage_distribution(coverage_file, graph_path, read_to_node_path, attr='median_coverage', num_thresholds=100):
    """
    Analyze the distribution of coverage values and find optimal threshold.
    
    Args:
        coverage_file: Path to the CSV file containing coverage data
        graph_path: Path to the NetworkX graph file containing node regions
        read_to_node_path: Path to pickle file containing read_id -> (node1, node2) mapping
        num_thresholds: Number of threshold values to test
    """
    # Load the read_to_node mapping
    print("Loading read to node mapping...")
    with open(read_to_node_path, 'rb') as f:
        read_to_node = pickle.load(f)
    
    # Load the graph and get node regions
    print("Loading graph...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    node_regions = nx.get_node_attributes(G, 'region')

    # Read the coverage data, converting read_id to string
    print("Reading coverage data...")
    df = pd.read_csv(coverage_file)
    df['read_id'] = df['read_id'].astype(str)
    
    # Print some debug information
    print(f"Number of reads in coverage file: {len(df)}")
    print(f"Number of reads in read_to_node mapping: {len(read_to_node)}")
    print(f"Sample read_id from coverage file: {df['read_id'].iloc[0]}")
    print(f"Sample read_id from mapping: {next(iter(read_to_node))}")
    
    # Match coverage data with node regions using the mapping
    matched_data = []
    true_labels = []
    matched_reads = []
    
    for _, row in df.iterrows():
        read_id = row['read_id']
        if read_id in read_to_node:
            node1, node2 = read_to_node[read_id]
            # Check if either node exists in the graph and has a region
            if node1 in node_regions:
                matched_data.append(row[attr])
                true_labels.append(1 if node_regions[node1] == 'E' else 0)
                matched_reads.append(read_id)
            elif node2 in node_regions:
                matched_data.append(row[attr])
                true_labels.append(1 if node_regions[node2] == 'E' else 0)
                matched_reads.append(read_id)
    
    coverage_values = np.array(matched_data)
    true_labels = np.array(true_labels)
    
    if len(matched_data) == 0:
        print("Error: No matching nodes found between coverage data and graph!")
        return

    # Add coverage threshold filter (4x coverage)
    coverage_threshold = 4 * 20  # Assuming expected coverage is 20x
    mask = coverage_values <= coverage_threshold
    coverage_values = coverage_values[mask]
    true_labels = true_labels[mask]
    
    if len(coverage_values) == 0:
        print("Error: No data points remaining after filtering high coverage values!")
        return
    
    print(f"Filtered out {np.sum(~mask)} nodes with coverage > {coverage_threshold}x")
    print(f"Remaining nodes: {len(coverage_values)}")
    
    # Count labels after filtering
    num_o_nodes = np.sum(true_labels == 0)
    num_e_nodes = np.sum(true_labels == 1)
    print(f"Number of O-nodes after filtering: {num_o_nodes}")
    print(f"Number of E-nodes after filtering: {num_e_nodes}")
    
    # Print range information with safety checks
    if num_o_nodes > 0:
        print("O-nodes range:", np.min(coverage_values[true_labels == 0]), 
              np.max(coverage_values[true_labels == 0]))
    else:
        print("Warning: No O-nodes found in the matched data")
    
    if num_e_nodes > 0:
        print("E-nodes range:", np.min(coverage_values[true_labels == 1]), 
              np.max(coverage_values[true_labels == 1]))
    else:
        print("Warning: No E-nodes found in the matched data")
    
    if num_o_nodes == 0 or num_e_nodes == 0:
        print("Error: Cannot perform threshold analysis without both O and E nodes")
        return
    
    # Test different thresholds
    min_coverage = 29 #np.min(coverage_values)
    max_coverage = 35 #np.max(coverage_values)
    thresholds = np.linspace(min_coverage, max_coverage, num_thresholds)
    
    results = []
    best_accuracy = 0
    best_threshold = None
    
    for threshold in thresholds:
        predicted_labels = (coverage_values > threshold).astype(int)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        
        results.append((threshold, accuracy, precision, recall))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    # Plot results
    thresholds, accuracies, precisions, recalls = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.axvline(x=best_threshold, color='g', linestyle='--', label='Best threshold')
    plt.xlabel('Coverage Threshold')
    plt.ylabel('Classification Metrics')
    plt.title('Coverage Classification Metrics vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('coverage_threshold_metrics.png')
    plt.close()
    
    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Precision at best threshold: {best_precision:.4f}")
    print(f"Recall at best threshold: {best_recall:.4f}")
    
    # Add confusion matrix analysis
    best_predictions = (coverage_values > best_threshold).astype(int)
    o_correct = np.sum((true_labels == 0) & (best_predictions == 0))
    o_incorrect = np.sum((true_labels == 0) & (best_predictions == 1))
    e_correct = np.sum((true_labels == 1) & (best_predictions == 1))
    e_incorrect = np.sum((true_labels == 1) & (best_predictions == 0))
    
    print("\nConfusion Matrix Analysis:")
    print(f"O-nodes classified as O (True Negative): {o_correct}")
    print(f"O-nodes classified as E (False Positive): {o_incorrect}")
    print(f"E-nodes classified as E (True Positive): {e_correct}")
    print(f"E-nodes classified as O (False Negative): {e_incorrect}")
    
    # Add predictions to original dataframe
    df['predicted_region'] = df['read_id'].map(
        lambda x: 'E' if x in read_to_node and 
        df[df['read_id'] == x][attr].iloc[0] > best_threshold 
        else 'O'
    )
    
    # Save results to CSV
    output_file = 'coverage_with_predictions.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

def main():
    reads_file = "/mnt/sod2-project/csb4/wgs/martin/diploid_datasets/master_seminar_d20/full_reads/i002c_v04_chr18_0.fasta.gz"
    kmer_size = 21
    kmer_hist_file = "kmer_histogram.txt"
    coverage_output_file = "coverage.txt"

    # Step 1: Run Jellyfish to compute k-mer frequencies
    #print("Running Jellyfish...")
    #run_jellyfish(reads_file, kmer_size, kmer_hist_file)

    # Step 2: Calculate read coverage and save to file
    #print("Calculating read coverage...")
    #calculate_read_coverage(reads_file, kmer_hist_file, coverage_output_file, kmer_size)
    print(f"Coverage estimates saved to {coverage_output_file}")

    # Step 3: Analyze coverage distribution and find optimal threshold
    print("Analyzing coverage distribution...")
    graph_path = "/mnt/sod2-project/csb4/wgs/martin/diploid_datasets/master_seminar_d20/nx_graphs/i002c_v04_chr18_0.pkl"  # Add your graph path here
    read_to_node_path = "/mnt/sod2-project/csb4/wgs/martin/diploid_datasets/master_seminar_d20/read_to_node/i002c_v04_chr18_0.pkl"  # Add your graph path here

    analyze_coverage_distribution(coverage_output_file, graph_path, read_to_node_path)



if __name__ == "__main__":
    main()
