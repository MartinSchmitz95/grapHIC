import torch
import argparse
import os
import yaml
import pickle
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip
import networkx as nx

def create_target_fastas(G, unitig_seqs, utg_to_node, gt_haps):
    """
    Create two gzipped FASTA files based on gt_hap values from NetworkX graph.
    
    Args:
        G: NetworkX graph containing gt_hap node features
        unitig_seqs: Dictionary with unitig IDs mapping to sequences
        utg_to_node: Dictionary mapping unitig IDs to node ID tuples (n1, n2)
    """
    pos_records = []  # For gt_hap 0 and 1
    neg_records = []  # For gt_hap 0 and -1
    
    for utg_id, node_tuple in utg_to_node.items():
        if utg_id not in unitig_seqs:
            print(f"Warning: unitig {utg_id} not found in sequences")
            continue
            
        for node in node_tuple:
            gt_hap = gt_haps[int(node)]
            print(f"gt_hap for node {node}: {gt_hap}")
            # Create SeqRecord object
            record = SeqRecord(
                seq=Seq(unitig_seqs[utg_id]),
                id=utg_id,
                description=f"gt_hap={gt_hap}"
            )
            
            # Add to pos_records if gt_hap is 0 or 1
            if gt_hap in [0, 1]:
                pos_records.append(record)
                
            # Add to neg_records if gt_hap is 0 or -1
            if gt_hap in [0, -1]:
                neg_records.append(record)
        
    # Write to gzipped FASTA files
    with gzip.open("hap_1.fasta.gz", "wt") as handle:
        SeqIO.write(pos_records, handle, "fasta")
    
    with gzip.open("hap_2.fasta.gz", "wt") as handle:
        SeqIO.write(neg_records, handle, "fasta")
    
    # Print statistics
    print("\nFASTA Creation Statistics:")
    print("-------------------------")
    print(f"Sequences in hap_1.fasta.gz: {len(pos_records)}")
    print(f"Sequences in hap_2.fasta.gz: {len(neg_records)}")

def main():
    parser = argparse.ArgumentParser(description="Create haplotype-specific FASTA files")
    parser.add_argument("--graph_path", type=str, required=True, help="Path to the NetworkX graph")
    parser.add_argument("--fasta_path", type=str, required=True, help="Path to the FASTA file")
    parser.add_argument("--utg_dict_path", type=str, required=True, help="Path to the UTG to node dictionary")

    args = parser.parse_args()

    # Load NetworkX graph
    with open(args.graph_path, 'rb') as f:
        G = pickle.load(f)
    
    # Load FASTA sequences
    unitig_seqs = {record.name: str(record.seq) for record in SeqIO.parse(gzip.open(args.fasta_path, "rt"), "fasta")}

    # Load UTG to node dictionary
    with open(args.utg_dict_path, 'rb') as f:
        utg_to_node = pickle.load(f)
    
    # Print UTG dictionary statistics
    print("\nUTG Dictionary Statistics:")
    print("-----------------------")
    print(f"Total UTG entries: {len(utg_to_node)}")
    
    # Print graph statistics
    print("\nGraph Statistics:")
    print("----------------")
    print(f"Total nodes in graph: {G.number_of_nodes()}")
    # Print all node attributes
    print("\nNode Attributes:")
    print("---------------")
    # Get attributes of first node to see what's available
    first_node = list(G.nodes())[0]
    attrs = list(G.nodes[first_node].keys())
    print(f"Available attributes: {', '.join(attrs)}")
    
    # Get all gt_hap attributes for counting
    gt_haps = nx.get_node_attributes(G, 'gt_hap')
    hap_counts = {-1: 0, 0: 0, 1: 0}
    for gt_hap in gt_haps.values():
        if gt_hap is not None:
            hap_counts[gt_hap] = hap_counts.get(gt_hap, 0) + 1
    
    print("\ngt_hap Distribution:")
    print("------------------")
    for hap, count in hap_counts.items():
        print(f"gt_hap = {hap:2d}: {count} nodes")

    # Create target FASTA files
    create_target_fastas(G, unitig_seqs, utg_to_node, gt_haps)
    
if __name__ == "__main__":
    main()
