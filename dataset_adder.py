#!/bin/env python
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


class HicDatasetCreator:
    def __init__(self, ref_path, dataset_path, data_config='dataset.yml'):
        with open(data_config) as file:
            config = yaml.safe_load(file)
        #self.full_dataset, self.val_dataset, self.train_dataset = utils.create_dataset_dicts(data_config=data_config)
        self.paths = config['paths']
        gen_config = config['gen_config']

        self.nfcore_hic = config['nfcore_hic']
        self.nextflow_config = config['nextflow']
        self.genome_str = ""
        self.gen_step_config = config['gen_steps']
        self.nextflow_call = self.paths['nextflow_path']
        self.depth = gen_config['depth']
        self.real = gen_config['real']

        self.root_path = dataset_path
        self.tmp_path = os.path.join(dataset_path, 'tmp')

        # HiC stuff
        self.hic_pipeline_path = self.paths['hic_pipeline_path']
        self.hic_root_path = os.path.join(dataset_path, "hic")
        self.hic_readsfiles_pairs = self.paths['hic_readsfiles_pairs']
        self.hic_sample_path = os.path.join(self.hic_root_path, self.genome_str)
        
        self.load_chromosome("", "")

        self.nx_graphs_path = os.path.join(dataset_path, "nx_utg_graphs")
        self.pyg_graphs_path = os.path.join(dataset_path, "pyg_graphs")
        self.hic_graphs_path = os.path.join(dataset_path, "hic_graphs")
        self.merged_graphs_path = os.path.join(dataset_path, "merged_graphs")
        self.reduced_reads_path = os.path.join(dataset_path, "reduced_reads")

        self.deadends = {}
        self.gt_rescue = {}
        self.edge_info = {}
        
        for folder in [self.nx_graphs_path, self.pyg_graphs_path, self.tmp_path,
                       self.hic_graphs_path, self.merged_graphs_path, self.reduced_reads_path]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        #self.edge_attrs = ['overlap_length', 'overlap_similarity', 'prefix_length']
        self.node_attrs = ['overlap_degree', 'hic_degree', 'read_length', 'hic_neighbor_weight', 'support']#, 'cov_avg', 'cov_pct', 'cov_med', 'cov_std', 'read_gc']

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
        Runs nf-core/hic on the HiC reads provided in the data config on the unitigs fasta.
        This finds chromosomal contacts between different unitigs on the same chromosome/haplotype.
        """
        fasta_unitig_file = f"{self.reduced_reads_path}/{self.genome_str}.fasta"
        # for now nf-core/hic does not support compressed input, decompress it 
        #TODO make PR
        if not os.path.exists(fasta_unitig_file):
            subprocess.run(f"gunzip -k {fasta_unitig_file}.gz", shell=True, check=True)
        # set fasta param to what the filename out is
        self.nfcore_hic["fasta"] = fasta_unitig_file

        nf_conf = self._write_nf_config()
        nf_params = self._write_nf_params()
        samplesheet = self._write_samplesheet()

        call = ' '.join([self.nextflow_call, "-log nextflow.log run", self.hic_pipeline_path, # "-resume",
                        "-c", nf_conf,
                        "-params-file", nf_params, "--input", samplesheet,
                         "--outdir", self.hic_sample_path, "-w", self.tmp_path, "-profile docker"])

        # call nextflow, this should finish when the pipeline is done
        print(call)
        subprocess.run(call, shell=True, check=False, cwd=self.root_path)

    def _write_nf_config(self, filename="nextflow.config") -> os.PathLike:
        """
        Writes the nextflow config file for nf-core/hic to a file.
        Allows all configuration to stay in dataset_config.yml.
        """
        path = os.path.join(self.hic_sample_path, filename)
        with open(path, 'wt') as f:
            # nf config is stored as a text blurb in the yml, can just write directly
            f.write(self.nextflow_config)

        return path

    def _write_nf_params(self, filename="params.yml") -> os.PathLike:
        """
        Writes the parameters for nf-core/hic to a yml file.
        Allows all configuration to stay in dataset_config.yml.
        """
        path = os.path.join(self.hic_sample_path, filename)

        with open(path, 'wt') as f:
            yaml.safe_dump(self.nfcore_hic, f)

        return path

    def _write_samplesheet(self, filename="samplesheet.csv") -> os.PathLike:
        """
        Writes the samplesheet for nf-core/hic to a file.
        Multiple hic runs may be used per sample. They are written as individual lines to the sample sheet under the same sample name, and will be merged by nf-core/hic.
        Allows all configuration to stay in dataset_config.yml.
        """
        path = os.path.join(self.hic_sample_path, filename)

        with open(path, 'wt') as f:
            f.write("sample,fastq_1,fastq_2\n")
            f.writelines([','.join([self.genome_str, f_pair[0], f_pair[1]]) + '\n' for f_pair in self.hic_readsfiles_pairs])

        return path

    def make_hic_edges(self):
        """
        Takes the output generated by the nf-core-HiC-step, and transforms it into a networkx graph containing contact edges.
        Uses the read to node mappings to make sure node IDs are the same as in the original graphs
        """
        from descongelador import export_connection_graph

        export_connection_graph(
                os.path.join(self.hic_sample_path, "contact_maps", "cool", self.genome_str + ".1000000_balanced.cool"),
                os.path.join(self.hic_graphs_path, self.genome_str + ".nx.pkl"),
                None)

    def load_hic_edges(self):#-> nx.MultiGraph:
        ret = None
        with open(os.path.join(self.hic_graphs_path, self.genome_str + ".nx.pkl"), 'rb') as f:

            ret = pickle.load(f)
        return ret

    def merge_graphs(self, nx_graph, hic_graph):
        nx.set_edge_attributes(nx_graph, "overlap", "type")
        # Create a new graph for the result
        merged_graph = nx.MultiGraph()  # Create empty MultiGraph
        
        # Add all nodes with their attributes
        merged_graph.add_nodes_from(nx_graph.nodes(data=True))
        
        # Add all edges with their attributes
        merged_graph.add_edges_from(nx_graph.edges(data=True))
        
        # Copy graph attributes
        merged_graph.graph.update(nx_graph.graph)
        
        print(merged_graph.nodes())
        #print node attributes
        for node in merged_graph.nodes():
            print(f"Node {node} attributes: {merged_graph.nodes[node]}")
        #exit()

        # Add Hi-C edges with the mapping (a,b) -> (a*2, b*2)
        for u, v, data in hic_graph.edges(data=True):
            # Map node IDs: multiply by 2
            u_mapped = u #* 2
            v_mapped = v #* 2
            
            # Only add nodes if they don't already exist (to preserve existing attributes)
            if u_mapped not in merged_graph:
                merged_graph.add_node(u_mapped)
                print(f"Added node {u_mapped}")
                exit()
            if v_mapped not in merged_graph:
                merged_graph.add_node(v_mapped)
                print(f"Added node {v_mapped}")
                exit()
            
            # Add the edge with type "hic"
            merged_graph.add_edge(u_mapped, v_mapped, type="hic", **data)
        
        # Set graph attribute to indicate edges need normalization since we added new HiC edges
        merged_graph.graph['edges_normalized'] = False

        print(f"\nTotal nodes in merged graph: {len(merged_graph.nodes())}")
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
                node_features.append(float(nx_graph.nodes[node][feat]))

            features.append(node_features)
        x = torch.tensor(features, dtype=torch.float)

        if not self.real:
            gt_hap = [nx_graph.nodes[node]['yak_m'] for node in range(num_nodes)]
            gt_tensor = torch.tensor(gt_hap, dtype=torch.long)
        else:
            gt_tensor = torch.tensor([0 for _ in range(num_nodes)], dtype=torch.long)
        
        # Extract chromosome numbers and convert to tensor
        chr_numbers = []
        for node in range(num_nodes):
            chr_str = nx_graph.nodes[node]['read_chr']
            # Extract number from string like 'chrX' or 'X'
            chr_num = int(chr_str.replace('chr', ''))
            chr_numbers.append(chr_num)
        chr_tensor = torch.tensor(chr_numbers, dtype=torch.long)
        # Create PyG Data object

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


    def normalize_edge_weights(self, nx_graph):
        """
        Normalize edge weights for HiC edges by dividing by 10000.
        This should only be called once to avoid double normalization.
        Uses graph attribute 'edges_normalized' to prevent double normalization.
        """
        # Check if edges are already normalized using graph attribute
        if nx_graph.graph.get('edges_normalized', False):
            print("Edge weights already normalized, skipping normalization")
            return nx_graph
            
        print("Normalizing edge weights...")
        
        for u, v, key, data in nx_graph.edges(data=True, keys=True):
            edge_type = data.get('type')
            if isinstance(edge_type, list):
                edge_type = edge_type[0]
            
            # Only normalize HiC edges
            if edge_type == 'hic':
                data['weight'] = data.get('weight', 1.0) / 10000
        
        # Set graph attribute to indicate normalization is complete
        nx_graph.graph['edges_normalized'] = True
        print("Edge weight normalization completed")
        return nx_graph

    def add_hic_neighbor_weights(self, nx_graph):
        """
        Add a node feature that sums the weights of all HiC edges connected to each node.
        Note: This assumes edge weights are already normalized.
        """
        hic_weights_sum = {}
        
        # Iterate through all nodes
        for node in nx_graph.nodes():
            total_weight = 0
            # Get all edges connected to this node
            for _, neighbor, key, data in nx_graph.edges(node, data=True, keys=True):
                edge_type = data.get('type')
                # Ensure edge_type is a string
                if isinstance(edge_type, list):
                    edge_type = edge_type[0]
                # Sum weights only for HiC edges
                if edge_type == 'hic':
                    total_weight += data['weight']
            hic_weights_sum[node] = total_weight
        
        # Add the feature to the graph
        nx.set_node_attributes(nx_graph, hic_weights_sum, 'hic_neighbor_weight')
        
        print(f"Added HiC neighbor weights feature to {len(hic_weights_sum)} nodes")
        
        return nx_graph

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
            for _, _, _, data in nx_graph.edges(data=True, keys=True):
                edge_attributes.update(data.keys())
            
            for attr in edge_attributes:
                values = []
                for _, _, _, data in nx_graph.edges(data=True, keys=True):
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
            for _, _, _, data in nx_graph.edges(data=True, keys=True):
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
        
    #if i > 0:
    #   return

    dataset_object.load_chromosome(genome, chr_id)
    print(f'Processing {dataset_object.genome_str}...')

    # run HiC pipeline
    if gen_step_config['align']:
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
        
        # Normalize edge weights first (only once)
        nx_graph = dataset_object.normalize_edge_weights(nx_graph)
        
        # Add node degree features
        nx_graph = dataset_object.add_node_degrees(nx_graph)
        
        # Add HiC neighbor weights (uses already normalized weights)
        nx_graph = dataset_object.add_hic_neighbor_weights(nx_graph)

        dataset_object.pickle_save(nx_graph, dataset_object.merged_graphs_path)

    # Convert to PyG format
    if gen_step_config['pyg']:
        # Load the graph if not already processed
        nx_graph = dataset_object.load_nx_graph(multi=True)

        dataset_object.save_to_dgl_and_pyg(nx_graph)
        #dataset_object.split_and_save_pyg(nx_graph, index=i)
        print(f"Saved DGL and PYG graphs of {chrN}_{i}")

    # Compute and display graph statistics after all processing steps
    nx_graph = dataset_object.load_nx_graph(multi=True)
    stats = dataset_object.compute_graph_statistics(nx_graph)

    print("Done for one chromosome!")

if __name__ == "__main__":
    main()
