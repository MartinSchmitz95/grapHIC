#!/bin/env python
"""
Hi-C Dataset Creator with Streaming Optimizations

This script processes Hi-C data to create graph datasets for machine learning.
Optimization features include:

1. Streaming Pipeline:
   - Uses pairtools streaming to avoid intermediate files
   - Pipes BWA alignment directly to sorting (no intermediate BAM)
   - Processes pairs in a single streaming pipeline

2. Memory and Disk Optimizations:
   - Avoids writing genome file to disk (uses in-memory processing)
   - Handles compressed FASTA files directly without decompression
   - Early cleanup of intermediate files
   - Caches BWA index files

3. Processing Optimizations:
   - Faster BWA parameters (-k 19, -w 100, -r 1.5)
   - Early filtering of unitig names
   - Reduced disk space requirements (5GB vs 10GB)

4. Fast Approximate Mapping (--approx flag):
   - Uses minimap2 for dramatically faster alignment
   - Trades some accuracy for significant speed improvements
   - Lower disk space requirements (2GB vs 5GB)
   - Suitable for rapid prototyping and testing

5. Hi-C Normalization:
   - Uses symmetric normalization (D^(-1/2) * A * D^(-1/2)) for Hi-C edge weights
   - More computationally efficient than ICE normalization
   - Effective for graph neural networks and spectral analysis
   - Handles isolated nodes and non-numeric data automatically

Usage:
    python dataset_adder.py --config config.yml
    python dataset_adder.py --config config.yml --approx  # Fast approximate mode

Configuration options in dataset.yml:
    gen_config:
        parallel_jobs: 32         # Number of parallel jobs for BWA/samtools
        hic_normalization:
            normalization_method: "symmetric"  # Options: "symmetric", "ice", "none"
"""

import argparse
import os
import subprocess
import pickle
import gzip
import networkx as nx
import torch
import yaml
from collections import Counter
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio import AlignIO
import re
from datetime import datetime
import pandas as pd
import random
import json


class HicDatasetCreator:
    def __init__(self, ref_path, dataset_path, data_config='dataset.yml'):
        with open(data_config) as file:
            config = yaml.safe_load(file)
        #self.full_dataset, self.val_dataset, self.train_dataset = utils.create_dataset_dicts(data_config=data_config)
        self.paths = config['paths']
        gen_config = config['gen_config']

        self.genome_str = ""
        self.gen_step_config = config['gen_steps']
        self.depth = gen_config['depth']
        self.real = gen_config['real']
        
        # Add optimization flags
        self.parallel_jobs = gen_config.get('parallel_jobs', 32)  # Number of parallel jobs

        self.root_path = dataset_path
        self.tmp_path = os.path.join(dataset_path, 'tmp')

        # HiC stuff
        self.hic_root_path = os.path.join(dataset_path, "hic")
        self.hic_readsfiles_pairs = self.paths['hic_readsfiles_pairs']
        self.hic_sample_path = os.path.join(self.hic_root_path, self.genome_str)
        
        self.load_chromosome("", "")

        self.nx_graphs_path = os.path.join(dataset_path, "nx_utg_graphs")
        self.pyg_graphs_path = os.path.join(dataset_path, "pyg_graphs")
        self.hic_graphs_path = os.path.join(dataset_path, "hic_graphs")
        self.merged_graphs_path = os.path.join(dataset_path, "merged_graphs")
        self.reduced_reads_path = os.path.join(dataset_path, "reduced_reads")
        self.unitig_2_node_path = os.path.join(dataset_path, "unitig_2_node")

        self.deadends = {}
        self.gt_rescue = {}
        self.edge_info = {}
        
        for folder in [self.nx_graphs_path, self.pyg_graphs_path, self.tmp_path,
                       self.hic_graphs_path, self.merged_graphs_path, self.reduced_reads_path, self.unitig_2_node_path]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        #self.edge_attrs = ['overlap_length', 'overlap_similarity', 'prefix_length']
        self.node_attrs = ['overlap_degree', 'hic_degree', 'read_length', 'support']#, 'cov_avg', 'cov_pct', 'cov_med', 'cov_std', 'read_gc']

    def load_chromosome(self, genome, chr_id):
        self.genome_str = f'{genome}_{chr_id}'
        self.hic_sample_path = os.path.join(self.hic_root_path, self.genome_str)
        # create subfolder for sample in hic dir
        if not os.path.exists(self.hic_sample_path):
            if not os.path.exists(self.hic_sample_path):
                os.makedirs(self.hic_sample_path)
    
    def load_nx_graph(self, multi=False):
        if multi:
            file_name = os.path.join(self.merged_graphs_path, f'{self.genome_str}.pkl')
        else:
            file_name = os.path.join(self.nx_graphs_path, f'{self.genome_str}.pkl')
        with open(file_name, 'rb') as file:
            nx_graph = pickle.load(file)
        print(f"Loaded nx graph {self.genome_str}")
        return nx_graph


    def pickle_save(self, pickle_object, path):
        # Save the graph using pickle
        file_name = os.path.join(path, f'{self.genome_str}.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump(pickle_object, f) 
        print(f"File saved successfully to {file_name}")


    def process_hic(self):
        """
        Runs Hi-C processing pipeline using minimizer-based mapping for dramatically faster processing.
        This is an approximate method that trades some accuracy for significant speed improvements.
        Uses minimap2 for fast approximate alignment instead of BWA.
        """
        
        fasta_unitig_file = f"{self.reduced_reads_path}/{self.genome_str}.fasta"
        
        # Select one random pair of Hi-C reads
        random_pair = random.choice(self.hic_readsfiles_pairs)
        r1_file, r2_file = random_pair[0], random_pair[1]
        
        # Set working directory to hic sample path
        os.makedirs(self.hic_sample_path, exist_ok=True)
        
        # Filenames for intermediate files
        output_edges = os.path.join(self.hic_sample_path, "unitig_edges.tsv")
        
        try:
            # Check available disk space (need much less for minimizer approach)
            import shutil
            total, used, free = shutil.disk_usage(self.hic_sample_path)
            free_gb = free // (1024**3)
            print(f"Available disk space: {free_gb} GB")
            if free_gb < 2:  # Much lower requirement for minimizer approach
                print("Warning: Less than 2GB available disk space. This may cause issues.")
            
            # Extract unitig names from FASTA file (same as original method)
            print("Extracting unitig names from FASTA...")
            unitig_names_set = set()
            
            # Handle both compressed and uncompressed FASTA files
            if fasta_unitig_file.endswith('.gz'):
                import gzip
                with gzip.open(fasta_unitig_file, 'rt') as f:
                    for line in f:
                        if line.startswith(">"):
                            name = line[1:].split()[0]
                            unitig_names_set.add(name)
            else:
                with open(fasta_unitig_file, 'r') as f:
                    for line in f:
                        if line.startswith(">"):
                            name = line[1:].split()[0]
                            unitig_names_set.add(name)

            print(f"Extracted {len(unitig_names_set)} unitig names")
            
            # Use minimap2 for fast approximate alignment
            print("Aligning Hi-C reads with minimap2 (fast approximate mapping)...")
            
            # Create temporary files for minimap2 output
            paf_file = os.path.join(self.hic_sample_path, "alignments.paf")
            
            # Run minimap2 with optimized parameters for Hi-C data
            # -x sr: short-read mode optimized for Hi-C
            # -N 10: output up to 10 secondary alignments
            # -p 0.8: minimum identity threshold
            # -t: number of threads
            minimap2_cmd = f"minimap2 -x sr -N 10 -p 0.8 -t {self.parallel_jobs} {fasta_unitig_file} {r1_file} {r2_file} > {paf_file}"
            subprocess.run(minimap2_cmd, shell=True, check=True, cwd=self.hic_sample_path)
            
            print("Processing minimap2 alignments...")
            
            # Process PAF output to extract contacts
            edges = {}
            total_contacts = 0
            self_contacts = 0
            inter_contacts = 0
            
            # Read PAF file and extract contacts
            with open(paf_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 12:
                        # PAF format: query_name, query_length, query_start, query_end, strand, 
                        # target_name, target_length, target_start, target_end, matches, alignment_length, mapping_quality
                        query_name = parts[0]
                        target_name = parts[5]
                        mapping_quality = int(parts[11])
                        
                        # Filter by mapping quality (minimap2 quality scores are different from BWA)
                        if mapping_quality < 10:  # Lower threshold for minimap2
                            continue
                        
                        # Early filtering: skip if unitig names not in our set
                        if target_name not in unitig_names_set:
                            continue
                        
                        # For Hi-C, we need to pair reads from the same fragment
                        # Extract read pair information from query name
                        # Assuming paired reads have similar names (e.g., read1/read2 suffixes)
                        base_read_name = query_name.rstrip('/1').rstrip('/2').rstrip('_1').rstrip('_2')
                        
                        # Store alignment for pairing
                        if not hasattr(self, '_temp_alignments'):
                            self._temp_alignments = {}
                        
                        if base_read_name not in self._temp_alignments:
                            self._temp_alignments[base_read_name] = []
                        
                        self._temp_alignments[base_read_name].append({
                            'target': target_name,
                            'quality': mapping_quality,
                            'query': query_name
                        })
            
            # Process paired alignments to create contacts
            print("Creating contacts from paired alignments...")
            for base_read_name, alignments in self._temp_alignments.items():
                if len(alignments) >= 2:  # Need at least 2 alignments for a contact
                    # Sort by quality and take best alignments
                    alignments.sort(key=lambda x: x['quality'], reverse=True)
                    
                    # Take the two best alignments (could be from same or different unitigs)
                    best_alignments = alignments[:2]
                    
                    # Create contacts between the aligned unitigs
                    for i in range(len(best_alignments)):
                        for j in range(i+1, len(best_alignments)):
                            unitig1 = best_alignments[i]['target']
                            unitig2 = best_alignments[j]['target']
                            
                            total_contacts += 1
                            
                            # Create edge between unitigs
                            edge = tuple(sorted((unitig1, unitig2)))
                            edges[edge] = edges.get(edge, 0) + 1
                            
                            if unitig1 == unitig2:
                                self_contacts += 1
                            else:
                                inter_contacts += 1
            
            # Clean up temporary alignments
            if hasattr(self, '_temp_alignments'):
                del self._temp_alignments
            
            print(f"[+] Contact statistics (minimizer approach):")
            print(f"    Total contacts: {total_contacts}")
            print(f"    Self-contacts (same unitig): {self_contacts}")
            print(f"    Inter-contacts (different unitigs): {inter_contacts}")
            print(f"    Unique edges: {len(edges)}")
            
            # Write result
            print(f"[+] Writing contact edges to {output_edges}")
            with open(output_edges, "w") as out:
                for (u, v), count in sorted(edges.items(), key=lambda x: -x[1]):
                    out.write(f"{u}\t{v}\t{count}\n")
            
            # Clean up temporary files
            if os.path.exists(paf_file):
                os.remove(paf_file)
                print("Removed PAF file to save disk space")
            
            print("[✓] Hi-C processing completed (minimizer-based fast version).")
            
        except subprocess.CalledProcessError as e:
            print(f"Error in Hi-C minimizer processing: {e}")
            print(f"Command that failed: {e.cmd}")
            print(f"Return code: {e.returncode}")
            
            # Clean up intermediate files on error
            for file_path in [paf_file]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up {file_path}")
                    except:
                        pass
            
            raise
        except Exception as e:
            print(f"Unexpected error in Hi-C minimizer processing: {e}")
            raise

    def make_hic_edges(self):
        """
        Creates a NetworkX graph from the Hi-C contact edges file.
        Looks for unitig_edges.tsv in the hic_sample_path.
        """
        edges_file = os.path.join(self.hic_sample_path, "unitig_edges.tsv")
        
        if not os.path.exists(edges_file):
            print(f"Error: Hi-C edges file {edges_file} not found.")
            print("Run process_hic() or process_hic_minimizer() first, or place your unitig_edges.tsv file in the hic_sample_path.")
            return
        
        print(f"Loading Hi-C edges from: {edges_file}")
        
        # Create NetworkX graph
        hic_graph = nx.MultiGraph()
        
        # Read edges from file
        edge_count = 0
        with open(edges_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Skip comment lines
                if line.startswith("#"):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    try:
                        u, v, weight = parts[0], parts[1], int(parts[2])
                        # Add edge with weight
                        hic_graph.add_edge(u, v, weight=weight, type='hic')
                        edge_count += 1
                    except ValueError as e:
                        print(f"Warning: Could not parse line {line_num}: {line.strip()} - {e}")
                        continue
                elif len(parts) == 2:
                    # Handle case where weight is not provided (default to 1)
                    try:
                        u, v = parts[0], parts[1]
                        hic_graph.add_edge(u, v, weight=1, type='hic')
                        edge_count += 1
                    except Exception as e:
                        print(f"Warning: Could not parse line {line_num}: {line.strip()} - {e}")
                        continue
                else:
                    print(f"Warning: Skipping malformed line {line_num}: {line.strip()}")
        
        if edge_count == 0:
            print("Warning: No Hi-C edges found in file. Creating empty graph.")
        else:
            print(f"Successfully loaded {edge_count} Hi-C edges")
        
        # Save the graph
        output_file = os.path.join(self.hic_graphs_path, f"{self.genome_str}.nx.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(hic_graph, f)
        
        print(f"Hi-C graph saved with {hic_graph.number_of_nodes()} nodes and {hic_graph.number_of_edges()} edges")
        print(f"Graph saved to: {output_file}")

    def load_hic_edges(self):
        """
        Loads the Hi-C edges graph from pickle file.
        """
        file_path = os.path.join(self.hic_graphs_path, f"{self.genome_str}.nx.pkl")
        
        if not os.path.exists(file_path):
            print(f"Error: Hi-C graph file {file_path} not found.")
            print("Run make_hic_edges() after process_hic() or process_hic_minimizer() first.")
            return nx.MultiGraph()
        
        with open(file_path, 'rb') as f:
            hic_graph = pickle.load(f)
        
        return hic_graph

    def merge_graphs(self, nx_graph, hic_graph):
        """
        Merges the original graph with Hi-C edges graph.
        Since unitig IDs are the same as node IDs, we can directly add Hi-C edges.
        """
        # Set edge type for original graph edges
        nx.set_edge_attributes(nx_graph, "overlap", "type")
        
        # Create a new graph for the result
        merged_graph = nx.MultiGraph()
        
        # Add all nodes with their attributes from original graph
        merged_graph.add_nodes_from(nx_graph.nodes(data=True))
        
        # Add all edges with their attributes from original graph
        merged_graph.add_edges_from(nx_graph.edges(data=True))
        
        # Copy graph attributes
        merged_graph.graph.update(nx_graph.graph)
        
        # Add Hi-C edges directly since unitig IDs = node IDs
        hic_edges_added = 0
        for u, v, data in hic_graph.edges(data=True):
            try:
                # Convert unitig IDs to integers (node IDs)
                u_node = int(u)
                v_node = int(v)
                
                # Only add edge if both nodes exist in the original graph
                if u_node in merged_graph and v_node in merged_graph:
                    weight = data.get('weight', 1.0)
                    merged_graph.add_edge(u_node, v_node, type="hic", weight=weight)
                    hic_edges_added += 1
                else:
                    print(f"Warning: Node {u_node} or {v_node} not found in original graph")
            except ValueError:
                print(f"Warning: Could not convert unitig ID '{u}' or '{v}' to integer")
                continue
        
        # Set graph attribute to indicate edges need normalization since we added new HiC edges
        merged_graph.graph['edges_normalized'] = False
        
        print(f"Added {hic_edges_added} Hi-C edges to merged graph")
        print(f"Total nodes in merged graph: {len(merged_graph.nodes())}")
        print(f"Total edges in merged graph: {len(merged_graph.edges())}")
        
        return merged_graph

    def convert_to_single_stranded(self, nx_graph):
        """
        Converts double-stranded graph to single-stranded by removing odd-numbered nodes
        and redirecting their edges to their even-numbered complements.
        """
        
        print(f"Initial graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")

        # Count initial edge types
        initial_edge_types = {}
        for _, _, _, data in nx_graph.edges(data=True, keys=True):
            edge_type = data['type']
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            initial_edge_types[edge_type] = initial_edge_types.get(edge_type, 0) + 1
        print(f"Initial edge type distribution: {initial_edge_types}")

        # First, ensure all reverse complement edges are present
        # For each edge (u,v), add the reverse complement edge (v^rc, u^rc)
        edges_to_add = []
        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            # Calculate reverse complement node IDs
            u_rc = u + 1 if u % 2 == 0 else u - 1
            v_rc = v + 1 if v % 2 == 0 else v - 1
            
            # Add reverse complement edge (v_rc, u_rc) with same attributes
            edges_to_add.append((v_rc, u_rc, data))
        
        # Add the reverse complement edges to the graph
        for u_rc, v_rc, data in edges_to_add:
            nx_graph.add_edge(u_rc, v_rc, **data)
        
        # Create new directed multigraph for single-stranded version
        single_stranded = nx.MultiGraph()
        
        # Create mapping from even nodes to consecutive indices
        old_to_new = {}
        new_id = 0
        for node in sorted(nx_graph.nodes()):
            if node % 2 == 0:  # Keep even nodes
                old_to_new[node] = new_id
                new_id += 1
        
        # Copy node attributes for even nodes with new IDs
        for old_node, new_node in old_to_new.items():
            single_stranded.add_node(new_node)
            single_stranded.nodes[new_node].update(nx_graph.nodes[old_node])
        
        # Process edges
        edge_weights = {}  # Dictionary to store accumulated weights
        final_edge_types = {}
        duplicate_counts = {'hic': 0, 'overlap': 0}
        
        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            edge_type = data['type']
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            weight = data.get('weight', 1.0)
            
            # Convert nodes to their even complements if they're odd
            u_even = u if u % 2 == 0 else u - 1
            v_even = v if v % 2 == 0 else v - 1
            
            # Get new IDs for the even nodes
            u_new = old_to_new[u_even]
            v_new = old_to_new[v_even]
            
            # Sort the node IDs to ensure consistent ordering
            sorted_nodes = tuple(sorted((u_new, v_new)))
            # Create unique edge identifier as a tuple
            edge_id = sorted_nodes + (str(edge_type),)

            if edge_id not in edge_weights:
                edge_weights[edge_id] = weight
                final_edge_types[edge_type] = final_edge_types.get(edge_type, 0) + 1
            else:
                if edge_type == 'hic':  # Sum weights for HiC edges
                    edge_weights[edge_id] += weight
                duplicate_counts[edge_type] += 1

        # Add edges with accumulated weights
        for edge_id, weight in edge_weights.items():
            u_new, v_new, edge_type = edge_id[0], edge_id[1], edge_id[2]
            single_stranded.add_edge(u_new, v_new, type=edge_type, weight=weight)

        print(f"Converted graph: {single_stranded.number_of_nodes()} nodes, {single_stranded.number_of_edges()} edges")
        print(f"Final edge type distribution: {final_edge_types}")
        print(f"Duplicate edges processed: {duplicate_counts}")

        return single_stranded
    
    
    def save_to_dgl_and_pyg(self, nx_graph):
        from torch_geometric.data import Data
        print()
        print(f"Total nodes in graph: {nx_graph.number_of_nodes()}")
        # Assert that nx_graph is a multigraph
        assert isinstance(nx_graph, nx.MultiGraph) or isinstance(nx_graph, nx.MultiDiGraph), "Graph must be a NetworkX MultiGraph or MultiDiGraph"
        
        # Get number of nodes
        num_nodes = nx_graph.number_of_nodes()
        
        # Initialize lists for edges, types and weights
        edge_list = []
        edge_types = []
        edge_weights = []
        
        # Process each edge type
        edge_type_map = {'overlap': 0, 'hic': 1}

        # Convert edges for each type
        for (u, v, key, data) in nx_graph.edges(data=True, keys=True):
            edge_type = data.get('type')  # Get edge type
            # Ensure edge_type is a string
            if isinstance(edge_type, list):
                edge_type = edge_type[0]  # Take first element if it's a list
            elif edge_type is None:
                print(f"Warning: Edge ({u}, {v}) has no type, defaulting to 'overlap'")
                exit()
            
            if edge_type not in edge_type_map:
                print(f"Warning: Unknown edge type {edge_type}, defaulting to 'overlap'")
                edge_type = 'overlap'
                exit()
                
            edge_list.append([u, v])
            edge_types.append(edge_type_map[edge_type])
            edge_weights.append(data.get('weight', 1.0))  # Default weight to 1.0 if not found
            
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        # Note: Node degrees should already be computed by add_node_degrees function
        
        # Create node features using all features in self.node_attrs
        features = []
        for node in range(num_nodes):
            node_features = []
            for feat in self.node_attrs:
                if feat == 'read_length':
                    # Use log of read_length
                    read_length_value = float(nx_graph.nodes[node][feat])
                    node_features.append(np.log(read_length_value)/10)
                else:
                    node_features.append(float(nx_graph.nodes[node][feat]))

            features.append(node_features)
        x = torch.tensor(features, dtype=torch.float)

        if not self.real:
            gt_hap = [nx_graph.nodes[node]['yak_m'] for node in range(num_nodes)]
            gt_tensor = torch.tensor(gt_hap, dtype=torch.long)
        else:
            gt_tensor = torch.tensor([0 for _ in range(num_nodes)], dtype=torch.long)
        
        # Extract chromosome numbers and convert to tensor
        """chr_numbers = []
        for node in range(num_nodes):
            chr_str = nx_graph.nodes[node]['read_chr']
            # Extract number from string like 'chrX' or 'X'
            chr_num = int(chr_str.replace('chr', ''))
            chr_numbers.append(chr_num)
        chr_tensor = torch.tensor(chr_numbers, dtype=torch.long)"""
        # Create PyG Data object
        chr_tensor = torch.tensor([0 for _ in range(num_nodes)], dtype=torch.long)

        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
            y=gt_tensor,
            chr=chr_tensor  # Add chromosome tensor to the data object
        )

        # Save PyG graph
        pyg_file = os.path.join(self.pyg_graphs_path, f'{self.genome_str}.pt')
        torch.save(pyg_data, pyg_file)
        print(f"Saved PyG graph of {self.genome_str}")

    def add_node_degrees(self, nx_graph):
        """
        Compute and add normalized degree features for each node based on edge types.
        This should be called in the features step.
        """
        from torch_geometric.utils import degree
        import torch
        
        print("Computing node degrees...")
        
        # Get number of nodes
        num_nodes = nx_graph.number_of_nodes()
        
        # Initialize lists for edges and types
        edge_list = []
        edge_types = []
        
        # Process each edge type
        edge_type_map = {'overlap': 0, 'hic': 1}

        # Convert edges for degree calculation
        for (u, v, key, data) in nx_graph.edges(data=True, keys=True):
            edge_type = data.get('type')
            # Ensure edge_type is a string
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            elif edge_type is None:
                continue  # Skip edges with no type
            
            if edge_type not in edge_type_map:
                continue  # Skip unknown edge types
                
            edge_list.append([u, v])
            edge_types.append(edge_type_map[edge_type])
            
        if not edge_list:
            print("No valid edges found for degree calculation")
            return nx_graph
            
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        # Compute degrees for each edge type
        overlap_degrees = degree(edge_index[0][edge_type == 0], num_nodes=num_nodes)
        hic_degrees = degree(edge_index[0][edge_type == 1], num_nodes=num_nodes)
        
        # Add normalized degree information to the graph
        for node in range(num_nodes):
            nx_graph.nodes[node]['overlap_degree'] = float(overlap_degrees[node]) / 10
            nx_graph.nodes[node]['hic_degree'] = float(hic_degrees[node]) / 100
        print("Node degrees computed and added")

        return nx_graph

    def compute_graph_statistics(self, nx_graph):
        """
        Compute mean and standard deviation for each node and edge feature in the NetworkX graph.
        """
        print("\nComputing graph statistics...")
        
        # Node feature statistics
        print("Node feature statistics:")
        node_stats = {}
        
        # Get all node attributes
        if nx_graph.nodes():
            # Get first node to determine available attributes
            first_node = list(nx_graph.nodes())[0]
            node_attributes = nx_graph.nodes[first_node].keys()
            
            for attr in node_attributes:
                values = []
                for node in nx_graph.nodes():
                    if attr in nx_graph.nodes[node]:
                        val = nx_graph.nodes[node][attr]
                        # Handle different data types
                        if isinstance(val, (int, float)):
                            values.append(float(val))
                        elif isinstance(val, str):
                            # Skip string attributes for statistical computation
                            continue
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    node_stats[attr] = {'mean': mean_val, 'std': std_val, 'count': len(values)}
                    print(f"  {attr}: mean={mean_val:.4f}, std={std_val:.4f}, count={len(values)}")
        
        # Edge feature statistics
        print("\nEdge feature statistics:")
        edge_stats = {}
        
        if nx_graph.edges():
            # Get all edge attributes
            edge_attributes = set()
            for _, _, data in nx_graph.edges(data=True):
                edge_attributes.update(data.keys())
            
            for attr in edge_attributes:
                values = []
                for _, _, data in nx_graph.edges(data=True):
                    if attr in data:
                        val = data[attr]
                        # Handle different data types
                        if isinstance(val, (int, float)):
                            values.append(float(val))
                        elif isinstance(val, str):
                            # For string attributes like 'type', count occurrences
                            continue
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    edge_stats[attr] = {'mean': mean_val, 'std': std_val, 'count': len(values)}
                    print(f"  {attr}: mean={mean_val:.4f}, std={std_val:.4f}, count={len(values)}")
            
            # Special handling for edge type distribution
            edge_types = []
            for _, _, data in nx_graph.edges(data=True):
                edge_type = data.get('type')
                if isinstance(edge_type, list):
                    edge_type = edge_type[0]
                if edge_type:
                    edge_types.append(edge_type)
            
            if edge_types:
                type_counts = Counter(edge_types)
                print(f"  Edge type distribution: {dict(type_counts)}")
                edge_stats['type_distribution'] = dict(type_counts)
        
        # Summary
        print(f"\nGraph summary:")
        print(f"  Total nodes: {nx_graph.number_of_nodes()}")
        print(f"  Total edges: {nx_graph.number_of_edges()}")
        
        return {'node_stats': node_stats, 'edge_stats': edge_stats}
    
    

    def analyze_graph_data_types(self, nx_graph):
        """
        Analyze and report on data types in the graph to help identify non-numeric data.
        """
        print("\n=== Graph Data Type Analysis ===")
        
        # Analyze node attributes
        node_attr_types = {}
        for node in nx_graph.nodes():
            for attr, value in nx_graph.nodes[node].items():
                if attr not in node_attr_types:
                    node_attr_types[attr] = set()
                node_attr_types[attr].add(type(value).__name__)
        
        print("Node attribute types:")
        for attr, types in node_attr_types.items():
            print(f"  {attr}: {sorted(types)}")
        
        # Analyze edge attributes
        edge_attr_types = {}
        edge_type_distribution = {}
        
        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            # Count edge types
            edge_type = data.get('type')
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            edge_type_distribution[edge_type] = edge_type_distribution.get(edge_type, 0) + 1
            
            # Analyze edge attribute types
            for attr, value in data.items():
                if attr not in edge_attr_types:
                    edge_attr_types[attr] = set()
                edge_attr_types[attr].add(type(value).__name__)
        
        print("\nEdge type distribution:")
        for edge_type, count in edge_type_distribution.items():
            print(f"  {edge_type}: {count} edges")
        
        print("\nEdge attribute types:")
        for attr, types in edge_attr_types.items():
            print(f"  {attr}: {sorted(types)}")
        
        # Specifically analyze Hi-C edge weights
        hic_weights = []
        non_numeric_hic_edges = []
        zero_weight_hic_edges = []
        
        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            edge_type = data.get('type')
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            
            if edge_type == 'hic':
                weight = data.get('weight')
                if isinstance(weight, (int, float, np.integer, np.floating)) and np.isfinite(weight):
                    if weight > 0:
                        hic_weights.append(weight)
                    else:
                        zero_weight_hic_edges.append((u, v, key, weight))
                else:
                    non_numeric_hic_edges.append((u, v, key, weight, type(weight).__name__))
        
        print(f"\nHi-C edge analysis:")
        print(f"  Total Hi-C edges: {len(hic_weights) + len(non_numeric_hic_edges) + len(zero_weight_hic_edges)}")
        print(f"  Positive weight Hi-C edges: {len(hic_weights)}")
        print(f"  Zero/negative weight Hi-C edges: {len(zero_weight_hic_edges)}")
        print(f"  Non-numeric Hi-C edges: {len(non_numeric_hic_edges)}")
        
        if zero_weight_hic_edges:
            print("\nZero/negative weight Hi-C edges (first 10):")
            for i, (u, v, key, weight) in enumerate(zero_weight_hic_edges[:10]):
                print(f"  Edge ({u}, {v}, {key}): weight={weight}")
            if len(zero_weight_hic_edges) > 10:
                print(f"  ... and {len(zero_weight_hic_edges) - 10} more")
        
        if non_numeric_hic_edges:
            print("\nNon-numeric Hi-C edges (first 10):")
            for i, (u, v, key, weight, weight_type) in enumerate(non_numeric_hic_edges[:10]):
                print(f"  Edge ({u}, {v}, {key}): weight={weight} (type: {weight_type})")
            if len(non_numeric_hic_edges) > 10:
                print(f"  ... and {len(non_numeric_hic_edges) - 10} more")
        
        if hic_weights:
            hic_weights = np.array(hic_weights)
            print(f"\nPositive Hi-C weight statistics:")
            print(f"  Min: {np.min(hic_weights)}")
            print(f"  Max: {np.max(hic_weights)}")
            print(f"  Mean: {np.mean(hic_weights):.4f}")
            print(f"  Median: {np.median(hic_weights):.4f}")
            print(f"  Std: {np.std(hic_weights):.4f}")
        
        print("=" * 40)
        return len(non_numeric_hic_edges) + len(zero_weight_hic_edges)

    def apply_symmetric_normalization(self, nx_graph):
        """
        Apply symmetric normalization (Laplacian normalization) to Hi-C edges.
        This is an alternative to ICE normalization that uses D^(-1/2) * A * D^(-1/2).
        Often more computationally efficient and effective for graph neural networks.
        """
        print("Applying symmetric normalization to Hi-C edges...")
        
        # Collect all Hi-C edges and their weights
        hic_edges = []
        edge_to_data = {}
        non_numeric_edges = []
        
        print("Scanning Hi-C edges for non-numeric data...")
        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            edge_type = data.get('type')
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            
            if edge_type == 'hic':
                weight = data.get('weight')
                
                # Check if weight is numeric
                if isinstance(weight, (int, float, np.integer, np.floating)):
                    # Additional check for finite values
                    if np.isfinite(weight):
                        # Check if weight is 0 or negative
                        if weight > 0:
                            hic_edges.append((u, v, weight))
                            edge_to_data[(u, v, key)] = data
                        else:
                            print(f"Warning: Zero or negative weight {weight} for edge ({u}, {v}), removing")
                            non_numeric_edges.append((u, v, key))
                    else:
                        print(f"Warning: Non-finite weight {weight} for edge ({u}, {v}), removing")
                        non_numeric_edges.append((u, v, key))
                else:
                    print(f"Warning: Non-numeric weight {weight} (type: {type(weight)}) for edge ({u}, {v}), removing")
                    non_numeric_edges.append((u, v, key))
        
        # Remove non-numeric edges from the graph
        print(f"Removing {len(non_numeric_edges)} non-numeric edges from graph...")
        for u, v, key in non_numeric_edges:
            try:
                nx_graph.remove_edge(u, v, key)
            except Exception as e:
                print(f"Warning: Could not remove edge ({u}, {v}, {key}): {e}")
        
        if not hic_edges:
            print("No valid Hi-C edges found for symmetric normalization after filtering")
            return nx_graph
        
        print(f"Found {len(hic_edges)} valid Hi-C edges for normalization")
        
        # Create a subgraph with only Hi-C nodes and edges for normalization
        hic_nodes = set([u for u, v, _ in hic_edges] + [v for u, v, _ in hic_edges])
        hic_subgraph = nx.Graph()
        
        # Add nodes to subgraph
        for node in hic_nodes:
            hic_subgraph.add_node(node)
        
        # Add Hi-C edges to subgraph
        for u, v, weight in hic_edges:
            hic_subgraph.add_edge(u, v, weight=weight)
        
        # Get adjacency matrix
        print("Computing adjacency matrix...")
        A = nx.to_numpy_array(hic_subgraph, weight='weight')
        
        # Compute degree matrix
        print("Computing degree matrix...")
        deg = np.sum(A, axis=1)
        
        # Check for isolated nodes (degree = 0)
        isolated_nodes = np.where(deg == 0)[0]
        if len(isolated_nodes) > 0:
            print(f"Warning: Found {len(isolated_nodes)} isolated nodes, removing them from normalization")
            # Remove isolated nodes from adjacency matrix
            non_isolated = np.where(deg > 0)[0]
            A = A[non_isolated][:, non_isolated]
            deg = deg[non_isolated]
        
        # Compute D^(-1/2)
        print("Computing symmetric normalization...")
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        
        # Apply symmetric normalization: D^(-1/2) * A * D^(-1/2)
        A_normalized = D_inv_sqrt @ A @ D_inv_sqrt
        
        # Create mapping from node indices back to original node IDs
        node_list = list(hic_subgraph.nodes())
        if len(isolated_nodes) > 0:
            node_list = [node_list[i] for i in non_isolated]
        
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Update edge weights with normalized values
        print("Updating edge weights with normalized values...")
        updated_edges = 0
        for edge_key, data in edge_to_data.items():
            u, v, key = edge_key
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                # Use the normalized value (symmetric matrix, so both directions are the same)
                normalized_weight = A_normalized[i, j]
                data['weight'] = normalized_weight
                updated_edges += 1
            else:
                print(f"Warning: Nodes {u} or {v} not found in node_to_idx mapping")
        
        print(f"Symmetric normalization completed for {updated_edges} Hi-C edges")
        print(f"Removed {len(non_numeric_edges)} non-numeric edges from graph")
        
        # Print some statistics about the normalization
        if len(A_normalized) > 0:
            print(f"Normalized adjacency matrix statistics:")
            print(f"  Shape: {A_normalized.shape}")
            print(f"  Min value: {np.min(A_normalized):.6f}")
            print(f"  Max value: {np.max(A_normalized):.6f}")
            print(f"  Mean value: {np.mean(A_normalized):.6f}")
            print(f"  Std value: {np.std(A_normalized):.6f}")
        
        return nx_graph

    def analyze_hic_buckets(self, nx_graph):
        """
        Simple bucket analysis of Hi-C edge weights.
        """
        print("Analyzing Hi-C edge weight distribution...")
        
        # Collect Hi-C edge weights
        hic_weights = []
        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            edge_type = data.get('type')
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            
            if edge_type == 'hic':
                weight = data.get('weight', 1.0)
                if isinstance(weight, (int, float)) and weight > 0:
                    hic_weights.append(weight)
        
        if not hic_weights:
            print("No Hi-C edges found")
            return
        
        # Define buckets
        buckets = [
            (0.0, 0.05, "0.0-0.05"),
            (0.05, 0.1, "0.05-0.1"),
            (0.1, 0.15, "0.1-0.15"),
            (0.15, 0.2, "0.15-0.2"),
            (0.2, 0.25, "0.2-0.25"),
            (0.25, 0.3, "0.25-0.3"),
            (0.3, 0.35, "0.3-0.35"),
            (0.35, 0.4, "0.35-0.4"),
            (0.4, 0.45, "0.4-0.45"),
            (0.45, 0.5, "0.45-0.5"),
            (0.5, 0.55, "0.5-0.55"),
            (0.55, 0.6, "0.55-0.6"),
            (0.6, 0.65, "0.6-0.65"),
            (0.65, 0.7, "0.65-0.7"),
            (0.7, 0.75, "0.7-0.75"),
            (0.75, 0.8, "0.75-0.8"),
            (0.8, 0.85, "0.8-0.85"),
            (0.85, 0.9, "0.85-0.9"),
            (0.9, 0.95, "0.9-0.95"),
            (0.95, 1.0, "0.95-1.0"),
            (1.0, float('inf'), "1.0+")
        ]
        
        # Count edges in each bucket
        bucket_counts = {}
        for min_w, max_w, label in buckets:
            if max_w == float('inf'):
                count = sum(1 for w in hic_weights if w >= min_w)
            else:
                count = sum(1 for w in hic_weights if min_w <= w < max_w)
            bucket_counts[label] = count
        
        # Print results
        total_edges = len(hic_weights)
        print(f"Total Hi-C edges: {total_edges}")
        print("Weight distribution:")
        for label, count in bucket_counts.items():
            if count > 0:
                percentage = (count / total_edges) * 100
                print(f"  {label}: {count} edges ({percentage:.1f}%)")

    def split_and_save_pyg(self, nx_graph, index, output_name="output"):
        """
        Splits the NetworkX graph by chromosome attribute and saves separate PyG graphs.
        
        Args:
            nx_graph: NetworkX graph with 'read_chr' node attribute
            index: Sample index for naming
            output_name: Base name for output files
            
        Returns:
            None, but saves separate PyG files for each chromosome
        """
        from torch_geometric.data import Data
        
        print(f"\nSplitting graph by chromosome and saving separate PyG graphs...")
        print(f"Total nodes in graph: {nx_graph.number_of_nodes()}")
        print(f"Total edges in graph: {nx_graph.number_of_edges()}")
        
        # Assert that nx_graph is a multigraph
        assert isinstance(nx_graph, nx.MultiGraph) or isinstance(nx_graph, nx.MultiDiGraph), "Graph must be a NetworkX MultiGraph or MultiDiGraph"
        
        # Get all unique chromosomes in the graph
        chr_values = set()
        for node in nx_graph.nodes():
            chr_values.add(nx_graph.nodes[node]['read_chr'])
        
        print(f"Found {len(chr_values)} unique chromosomes: {sorted(chr_values)}")
        
        # Count total edges before splitting
        total_original_edges = nx_graph.number_of_edges()
        total_edges_after_split = 0
        
        # Process each chromosome separately
        for chr_id in sorted(chr_values):
            # Get nodes for this chromosome
            chr_nodes = [node for node in nx_graph.nodes() if nx_graph.nodes[node]['read_chr'] == chr_id]
            
            if not chr_nodes:
                print(f"No nodes found for chromosome {chr_id}, skipping")
                continue
                
            # Create subgraph for this chromosome
            chr_subgraph = nx_graph.subgraph(chr_nodes).copy()
            
            # Get number of nodes and edges in this subgraph
            num_nodes = chr_subgraph.number_of_nodes()
            num_edges = chr_subgraph.number_of_edges()
            
            print(f"\nProcessing chromosome {chr_id}: {num_nodes} nodes, {num_edges} edges")
            
            # Create a mapping from original node IDs to consecutive indices
            node_mapping = {old_id: new_id for new_id, old_id in enumerate(chr_subgraph.nodes())}
            
            # Initialize lists for edges, types and weights
            edge_list = []
            edge_types = []
            edge_weights = []
            
            # Process each edge type
            edge_type_map = {'overlap': 0, 'hic': 1}
            
            # Convert edges for each type
            for (u, v, key, data) in chr_subgraph.edges(data=True, keys=True):
                edge_type = data.get('type')  # Get edge type
                # Ensure edge_type is a string
                if isinstance(edge_type, list):
                    edge_type = edge_type[0]  # Take first element if it's a list
                elif edge_type is None:
                    print(f"Warning: Edge ({u}, {v}) has no type, skipping")
                    continue
                
                if edge_type not in edge_type_map:
                    print(f"Warning: Unknown edge type {edge_type}, skipping")
                    continue
                    
                # Map node IDs to new consecutive indices
                new_u = node_mapping[u]
                new_v = node_mapping[v]
                
                edge_list.append([new_u, new_v])
                edge_types.append(edge_type_map[edge_type])
                edge_weights.append(data.get('weight', 1.0))  # Default weight to 1.0 if not found
            
            if not edge_list:
                print(f"No valid edges found for chromosome {chr_id}, skipping")
                continue
                
            # Convert to tensors
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_weight = torch.tensor(edge_weights, dtype=torch.float)
            
            # Create node features using all features in self.node_attrs (excluding chr)
            features = []
            for node in chr_subgraph.nodes():
                node_features = []
                for feat in self.node_attrs:
                    if feat == 'read_length':
                        # Use log of read_length
                        read_length_value = float(chr_subgraph.nodes[node][feat])
                        node_features.append(np.log(read_length_value)/10)
                    else:
                        node_features.append(float(chr_subgraph.nodes[node][feat]))
                features.append(node_features)
            x = torch.tensor(features, dtype=torch.float)
            
            # Extract ground truth haplotype labels
            if not self.real:
                gt_hap = [chr_subgraph.nodes[node]['yak_m'] for node in chr_subgraph.nodes()]
                gt_tensor = torch.tensor(gt_hap, dtype=torch.long)
            else:
                gt_tensor = torch.tensor([0 for _ in range(num_nodes)], dtype=torch.long)
            
            # Create PyG Data object (without chr attribute)
            pyg_data = Data(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                edge_weight=edge_weight,
                y=gt_tensor
            )
            
            # Format chromosome ID for filename (remove 'chr' prefix if present)
            chr_name = chr_id.replace('chr', '')
            
            # Save PyG graph with naming format: output_name_chr[n]_[m].pt
            pyg_file = os.path.join(self.pyg_graphs_path, f'{output_name}_chr{chr_name}_{index}.pt')
            torch.save(pyg_data, pyg_file)
            
            total_edges_after_split += len(edge_list)
            print(f"Saved PyG graph for chromosome {chr_id} with {num_nodes} nodes and {len(edge_list)} edges to {pyg_file}")
        
        # Report edge loss statistics
        edges_lost = total_original_edges - total_edges_after_split
        print(f"\n=== Edge Loss Report ===")
        print(f"Total edges in original graph: {total_original_edges}")
        print(f"Total edges after chromosome separation: {total_edges_after_split}")
        print(f"Edges lost during separation: {edges_lost}")
        if total_original_edges > 0:
            loss_percentage = (edges_lost / total_original_edges) * 100
            print(f"Percentage of edges lost: {loss_percentage:.2f}%")
        
        if edges_lost > 0:
            print(f"Note: {edges_lost} edges were lost because they connected nodes from different chromosomes")
        else:
            print("No edges were lost during chromosome separation")

    def remove_hic_edges_with_threshold(self, nx_graph, threshold=0.01):
        """
        Remove all Hi-C edges with weights below the specified threshold.
        
        Args:
            nx_graph: NetworkX graph containing Hi-C edges
            threshold: Minimum weight threshold for Hi-C edges (default: 0.01)
            
        Returns:
            NetworkX graph with low-weight Hi-C edges removed
        """
        print(f"Removing Hi-C edges with weight < {threshold}...")
        
        # Count edges before removal
        total_hic_edges_before = 0
        edges_to_remove = []
        
        # Find Hi-C edges below threshold
        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            edge_type = data.get('type')
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            
            if edge_type == 'hic':
                total_hic_edges_before += 1
                weight = data.get('weight', 0.0)
                
                # Check if weight is numeric and below threshold
                if isinstance(weight, (int, float, np.integer, np.floating)):
                    if weight < threshold:
                        edges_to_remove.append((u, v, key))
                else:
                    print(f"Warning: Non-numeric weight {weight} for Hi-C edge ({u}, {v}), removing")
                    edges_to_remove.append((u, v, key))
        
        # Remove edges below threshold
        removed_count = 0
        for u, v, key in edges_to_remove:
            try:
                nx_graph.remove_edge(u, v, key)
                removed_count += 1
            except Exception as e:
                print(f"Warning: Could not remove edge ({u}, {v}, {key}): {e}")
        
        # Count remaining Hi-C edges
        total_hic_edges_after = 0
        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            edge_type = data.get('type')
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            
            if edge_type == 'hic':
                total_hic_edges_after += 1
        
        print(f"Hi-C edge filtering results:")
        print(f"  Total Hi-C edges before: {total_hic_edges_before}")
        print(f"  Hi-C edges removed: {removed_count}")
        print(f"  Hi-C edges remaining: {total_hic_edges_after}")
        print(f"  Removal percentage: {(removed_count/total_hic_edges_before*100):.2f}%" if total_hic_edges_before > 0 else "  No Hi-C edges found")
        
        return nx_graph


def create_full_dataset_dict(config):

    train_dataset = config['training']
    val_dataset = config['validation']

    # Initialize the full_dataset dictionary
    full_dataset = {}
    # Add all keys and values from train_dataset to full_dataset
    for key, value in train_dataset.items():
        full_dataset[key] = value
    # Add keys from val_dataset to full_dataset, summing values if key already exists
    if val_dataset is not None:
        for key, value in val_dataset.items():
            if key in full_dataset:
                full_dataset[key] += value
            else:
                full_dataset[key] = value

    return full_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate dataset based on configuration")
    parser.add_argument('--ref', type=str, default='/mnt/sod2-project/csb4/wgs/martin/genome_references', help='Path to references root dir')
    parser.add_argument('--data_path', type=str, default='/mnt/sod2-project/csb4/wgs/lovro_interns/leon/pipeline-test/', help='Path to dataset folder')   
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    dataset_path = args.data_path
    ref_base_path = args.ref
    config_path = args.config
    # Read the configuration file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    full_dataset = create_full_dataset_dict(config)

    # Initialize the appropriate dataset creator
    dataset_object = HicDatasetCreator(ref_base_path, dataset_path, config_path)

    # Process each chromosome
    for chrN, amount in full_dataset.items():
        for i in range(amount):
            gen_steps(dataset_object, chrN, i, config['gen_steps'], ref_base_path)

def gen_steps(dataset_object, chrN_, i, gen_step_config, ref_base_path):

    split_chrN = chrN_.split(".")
    chrN = split_chrN[1]
    genome = split_chrN[0]
    chr_id = f'{chrN}_{i}'
    ref_base = (f'{ref_base_path}/{genome}')
        
    if i < 2:
       return

    dataset_object.load_chromosome(genome, chr_id)
    print(f'Processing {dataset_object.genome_str}...')

    #nx_graph = dataset_object.load_nx_graph(multi=False)
    #print(f"Graph has yak_m attribute: {'yak_m' in nx_graph.nodes[0]}")
    #return
    # run HiC pipeline
    if gen_step_config['align']:
        print("Using fast minimizer-based Hi-C processing...")
        dataset_object.process_hic()


    if gen_step_config['hic_graph']:
        dataset_object.make_hic_edges()

    # update graph to include hic edges
    if gen_step_config['merge']:
        nx_graph = dataset_object.load_nx_graph()
        hic_graph = dataset_object.load_hic_edges()

        print("loaded Hi-C graph")
        nx_graph = dataset_object.merge_graphs(nx_graph, hic_graph)

        nx_graph = dataset_object.convert_to_single_stranded(nx_graph)
        print("merged Hi-C graph")

        dataset_object.pickle_save(nx_graph, dataset_object.merged_graphs_path)
        print("saved merged Hi-C graph")

    # Add features (including edge weight normalization)
    if gen_step_config['ftrs']:
        # Load the graph
        nx_graph = dataset_object.load_nx_graph(multi=True)
        
        # Analyze Hi-C contacts before normalization
        print("\nAnalyzing Hi-C contacts before normalization...")
        dataset_object.analyze_hic_buckets(nx_graph)
        
        # Apply symmetric normalization
        nx_graph = dataset_object.apply_symmetric_normalization(nx_graph)
        
        # Analyze Hi-C contacts after normalization
        print("\nAnalyzing Hi-C contacts after normalization...")
        dataset_object.analyze_hic_buckets(nx_graph)
        
        # Add node degree features
        nx_graph = dataset_object.add_node_degrees(nx_graph)
        
        # Add HiC neighbor weights (uses already normalized weights)
        # nx_graph = dataset_object.add_hic_neighbor_weights(nx_graph) # This line is removed

        dataset_object.pickle_save(nx_graph, dataset_object.merged_graphs_path)

    # Convert to PyG format
    if gen_step_config['pyg']:
        # Load the graph if not already processed
        nx_graph = dataset_object.load_nx_graph(multi=True)

        # Remove low-weight Hi-C edges with configurable threshold
        #hic_threshold = gen_step_config.get('hic_threshold', 0.02)  # Default threshold
        #nx_graph = dataset_object.remove_hic_edges_with_threshold(nx_graph, threshold=hic_threshold)
        #exit()
        dataset_object.save_to_dgl_and_pyg(nx_graph)
        #dataset_object.split_and_save_pyg(nx_graph, index=i, output_name=genome)
        print(f"Saved DGL and PYG graphs of {chrN}_{i}")

    # Compute and display graph statistics after all processing steps
    # Only load merged graph if merge step was run or if merged graph exists
    merged_graph_file = os.path.join(dataset_object.merged_graphs_path, f'{dataset_object.genome_str}.pkl')
    if gen_step_config['merge'] or os.path.exists(merged_graph_file):
        nx_graph = dataset_object.load_nx_graph(multi=True)
        stats = dataset_object.compute_graph_statistics(nx_graph)
    else:
        # Try to load the original nx graph for statistics if it exists
        original_graph_file = os.path.join(dataset_object.nx_graphs_path, f'{dataset_object.genome_str}.pkl')
        if os.path.exists(original_graph_file):
            nx_graph = dataset_object.load_nx_graph(multi=False)
            stats = dataset_object.compute_graph_statistics(nx_graph)
        else:
            print(f"No graph files found for {dataset_object.genome_str}, skipping statistics computation")

    print("Done for one chromosome!")

if __name__ == "__main__":
    main()
