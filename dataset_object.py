import os
import subprocess
import pickle
import gzip
import networkx as nx
import torch
import yaml
import shutil
from collections import Counter
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from torch_geometric.utils import from_networkx
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
        nfcore_hic_config = config['nfcore_hic']
        nextflow_config = config['nextflow']
        self.genome_str = ""
        self.genome = "hg002"
        self.sample_name = self.genome # not sure if should always be the same
        self.gen_step_config = config['gen_steps']
        self.centromere_dict = self.load_centromere_file()
        self.load_chromosome("", "", ref_path)
        self.hifiasm_path = self.paths['hifiasm_path']
        self.hifiasm_dump = self.paths['hifiasm_dump']
        self.raft_path = self.paths['raft_path']
        self.pbsim_path = self.paths['pbsim_path']
        self.nextflow_call = self.paths['nextflow_path']
        self.hic_pipeline_path = self.paths['hic_pipeline_path']
        self.sample_profile = self.paths['sample_profile']
        self.depth = gen_config['depth']
        self.diploid = gen_config['diploid']
        self.threads = gen_config['threads']
        self.real = gen_config['real']
        self.prep_decoding = gen_config['prep_decoding']
        self.raft = gen_config['raft']

        self.root_path = dataset_path
        self.tmp_path = os.path.join(dataset_path, 'tmp')
        self.full_reads_path = os.path.join(dataset_path, "full_reads")
        self.read_descr_path = os.path.join(dataset_path, "read_descr")
        self.gfa_unitig_path = os.path.join(dataset_path, "gfa_unitig")
        self.fasta_unitig_path = os.path.join(dataset_path, "fasta_unitig")
        self.fasta_raw_path = os.path.join(dataset_path, "fasta_raw")
        self.gfa_raw_path = os.path.join(dataset_path, "gfa_raw")
        self.overlaps_path = os.path.join(dataset_path, "overlaps")
        self.pile_o_grams_path = os.path.join(dataset_path, "pile_o_grams")
        self.utg_2_reads_path = os.path.join(dataset_path, "utg_2_reads")

        # HiC stuff
        self.hic_path = os.path.join(dataset_path, "hic")
        self.hic_readsfiles_pairs = self.paths['hic_readsfiles_pairs']

        self.nx_graphs_path = os.path.join(dataset_path, "nx_graphs")
        self.pyg_graphs_path = os.path.join(dataset_path, "pyg_graphs")
        self.read_to_node_path = os.path.join(dataset_path, "read_to_node")
        self.node_to_read_path = os.path.join(dataset_path, "node_to_read")
        self.utg_to_read_path = os.path.join(dataset_path, "utg_2_reads")

        self.deadends = {}
        self.gt_rescue = {}
        self.edge_info = {}

        for folder in [self.utg_2_reads_path, self.fasta_unitig_path, self.fasta_raw_path, self.full_reads_path, self.gfa_unitig_path, self.gfa_raw_path, self.nx_graphs_path,
                       self.pyg_graphs_path, self.read_descr_path, self.tmp_path, self.read_to_node_path, self.node_to_read_path, self.overlaps_path, self.pile_o_grams_path]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        #self.edge_attrs = ['overlap_length', 'overlap_similarity', 'prefix_length']
        self.node_attrs = ['read_length']
        if not self.real:
            self.node_attrs.extend('gt_hap')

    def load_chromosome(self, genome, chr_id, ref_path):
        self.genome_str = f'{genome}_{chr_id}'
        self.genome = genome
        self.chr_id = chr_id
        self.centromere_graph = '_c' in chr_id
        self.chrN = chr_id.split('_')[0]
        self.chromosomes_path = os.path.join(ref_path, 'chromosomes')
        self.centromeres_path = os.path.join(ref_path, 'centromeres')
        self.vcf_path = os.path.join(ref_path, 'vcf')
        self.chain_path = os.path.join(ref_path, 'chain')
        self.ref_path = ref_path
        self.maternal_yak = os.path.join(ref_path, 'mat.yak')
        self.paternal_yak = os.path.join(ref_path, 'pat.yak')

    def load_centromere_file(self):
        if 'centromere_coords' in self.paths.keys():
            centromere_coords_file = self.paths['centromere_coords']
            try:
                with open(centromere_coords_file, 'r') as file:
                    centromere_data = yaml.safe_load(file)
                return centromere_data
            except Exception as e:
                print(f"Error reading centromere coordinates file: {e}")
        exit()
    
    def get_centromere_coords(self, haplotype, chr_id):
        """
        Get centromere coordinates for a specific chromosome and haplotype.
        
        Args:
            haplotype (str): 'M' for maternal or 'P' for paternal
            chr_id (str): Chromosome identifier (e.g., '1', 'chr1', etc.)
        
        Returns:
            tuple: (centromere_start, centromere_end) coordinates
        """
        try:
            genome_data = self.centromere_dict[self.genome]
            # Add 'chr' prefix if not present
            if not chr_id.startswith('chr'):
                chr_id = f'chr{chr_id}'
                
            # Construct the key using chromosome and haplotype
            key = f'{chr_id}_{haplotype}'
            
            chromosome_data = genome_data[key]
            c_start = chromosome_data['c_start']
            c_end = chromosome_data['c_end']
            
            return c_start, c_end
        except KeyError as e:
            print(f"Error: Could not find centromere coordinates for key: {key}")
            print(f"Available keys in genome data: {list(genome_data.keys())}")
            raise e

    def _get_filetype(self, file_path):
        """Determine if file is FASTA or FASTQ format based on extension."""
        if file_path.endswith(('.gz', '')):
            base_path = file_path[:-3] if file_path.endswith('.gz') else file_path
            if base_path.endswith(('fasta', 'fna', 'fa')):
                return 'fasta'
            elif base_path.endswith(('fastq', 'fnq', 'fq')):
                return 'fastq'
        return 'fasta'  # Default to fasta if unknown
    
    def get_read_headers(self, reads_path):
        """Extract read headers from FASTA/FASTQ file, handling both compressed and uncompressed files.
        Returns a dict mapping read IDs to descriptions."""
        filetype = self._get_filetype(reads_path)
        if reads_path.endswith('.gz'):
            with gzip.open(reads_path, 'rt') as handle:
                return {read.id: read.description for read in SeqIO.parse(handle, filetype)}
        else:
            return {read.id: read.description for read in SeqIO.parse(reads_path, filetype)}
    
    def load_nx_graph(self):
        file_name = os.path.join(self.nx_graphs_path, f'{self.genome_str}.pkl')
        with open(file_name, 'rb') as file:
            nx_graph = pickle.load(file)
        print(f"Loaded nx graph {self.genome_str}")
        return nx_graph
        
    def create_reads_fasta(self, read_seqs, chr_id):
        seq_records = []
        for read_id, sequence in read_seqs.items():
            seq_record = SeqRecord(Seq(sequence), id=str(read_id), description="")
            seq_records.append(seq_record)
        seq_record_path = os.path.join(self.reduced_reads_path, f'{self.genome_str}.fasta')
        SeqIO.write(seq_records, seq_record_path, "fasta")

    def pickle_save(self, pickle_object, path):
        # Save the graph using pickle
        file_name = os.path.join(path, f'{self.genome_str}.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump(pickle_object, f) 
        print(f"File saved successfully to {file_name}")

    def save_to_dgl_and_pyg(self, nx_graph):
        print()
        print(f"Total nodes in graph: {nx_graph.number_of_nodes()}")
        
        # Get list of available node attributes in the graph
        available_attrs = []
        if len(nx_graph.nodes) > 0:
            sample_node = list(nx_graph.nodes)[0]
            available_attrs = list(nx_graph.nodes[sample_node].keys())
        
        # Filter node_attrs to only include attributes that exist in the graph
        node_attrs_to_use = [attr for attr in self.node_attrs if attr in available_attrs]
        
        print(f"Using node attributes: {node_attrs_to_use}")
        
        # Create PyG graph directly from networkx
        pyg_data = from_networkx(nx_graph, group_node_attrs=node_attrs_to_use)
        
        # Save PyG graph
        pyg_file = os.path.join(self.pyg_graphs_path, f'{self.genome_str}.pt')
        torch.save(pyg_data, pyg_file)
        print(f"Saved PyG graph of {self.genome_str}")

    def simulate_pbsim_reads(self):
        if self.diploid:
            variants = ['M', 'P']
        else:
            variants = ['M']
        out_file = os.path.join(self.full_reads_path, f'{self.genome_str}.fasta')
        out_descr_file = os.path.join(self.read_descr_path, f'{self.genome_str}.fasta')
        read_id = 1
        
        chr_name_split = self.chr_id.split('_')
        centromere = 'c' in chr_name_split
        multi_mode = any('multi' in part for part in chr_name_split)
        
        # Handle multi-chromosome mode
        if multi_mode:
            # Find the number after 'multi'
            for part in chr_name_split:
                if 'multi' in part:
                    num_chrs = int(chr_name_split[chr_name_split.index(part) + 1])
                    break
            available_chrs = [i for i in range(1,23) if i != 14]
            sampled_chrs = np.random.choice(available_chrs, size=num_chrs, replace=False)
            print(f"Sampling chromosomes: {sampled_chrs}")
        else:
            chr_num = chr_name_split[0]
            if chr_num.startswith('chr'):
                chr_num = chr_num[3:]
            sampled_chrs = [int(chr_num)]

        # Create temporary files for each chromosome/variant combination
        temp_files = []
        temp_descr_files = []

        for chr_num in sampled_chrs:
            for var in variants:
                # Create temporary files for this iteration
                temp_fasta = os.path.join(self.tmp_path, f'temp_chr{chr_num}_{var}.fasta')
                temp_descr = os.path.join(self.tmp_path, f'temp_chr{chr_num}_{var}_descr.fasta')
                temp_files.append(temp_fasta)
                temp_descr_files.append(temp_descr)

                if centromere:
                    ref_path = os.path.join(self.centromeres_path, f'chr{chr_num}_{var}_c.fasta')
                else:
                    ref_path = os.path.join(self.chromosomes_path, f'chr{chr_num}_{var}.fasta')
                
                # Run PBSIM
                subprocess.run(f'./src/pbsim --strategy wgs --method sample --depth {self.depth} --genome {ref_path} --sample-profile-id {self.sample_profile}',
                               shell=True, cwd=self.pbsim_path)
                
                reads = {r.id: r for r in SeqIO.parse(f'{self.pbsim_path}/sd_0001.fastq', 'fastq')}
                reads_list = []
                
                for align in AlignIO.parse(f'{self.pbsim_path}/sd_0001.maf', 'maf'):
                    ref, read_m = align
                    start = ref.annotations['start']
                    end = start + ref.annotations['size']
                    strand = '+' if read_m.annotations['strand'] == 1 else '-'
                    description = f'strand={strand} start={start} end={end} variant={var} chr={chr_num}'
                    reads[read_m.id].description = description
                    reads[read_m.id].id = f'{read_id}'
                    read_id += 1
                    reads_list.append(reads[read_m.id])

                # Write to temporary files
                SeqIO.write(reads_list, temp_fasta, 'fasta')
                read_descr_list = [SeqRecord(Seq(""), id=record.id, description=record.description) 
                                  for record in reads_list]
                SeqIO.write(read_descr_list, temp_descr, 'fasta')

                # Clean up PBSIM files
                subprocess.run(f'rm sd_0001.fastq sd_0001.maf sd_0001.ref', shell=True, cwd=self.pbsim_path)

        try:
            # Merge all temporary FASTA files
            subprocess.run(f'cat {" ".join(temp_files)} > {out_file}', shell=True, check=True)
            
            # Compress the merged file
            subprocess.run(f'gzip -f {out_file}', shell=True, check=True)
            
            # Merge description files
            subprocess.run(f'cat {" ".join(temp_descr_files)} > {out_descr_file}', shell=True, check=True)
            
            # Verify the compressed file
            with gzip.open(f"{out_file}.gz", 'rt') as verify_file:
                verify_count = sum(1 for _ in SeqIO.parse(verify_file, 'fasta'))
                expected_count = read_id - 1
                if verify_count != expected_count:
                    raise RuntimeError(f"File verification failed: expected {expected_count} records but read {verify_count}")
                
        except Exception as e:
            raise RuntimeError(f"Error processing files: {str(e)}")
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files + temp_descr_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    def run_raft(self):
        reads_file = os.path.join(self.fasta_unitig_path, f'{self.genome_str}.fasta.gz')
        raft_depth = 2 * self.depth

        #overlaps_file = os.path.join(self.overlaps_path, f"{self.genome_str}_cis.paf.gz")

        subprocess.run(f'./hifiasm --dbg-ovec -r3 -o {self.hifiasm_dump}/tmp_asm -t{self.threads} {reads_file}', shell=True, cwd=self.hifiasm_path)
        #subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.r_utg.gfa {gfa_unitig_output}', shell=True, cwd=self.hifiasm_path)

        # Merge cis and trans overlaps
        cis_paf = f"{self.hifiasm_dump}/tmp_asm.0.ovlp.paf"
        trans_paf = f"{self.hifiasm_dump}/tmp_asm.1.ovlp.paf"
        merged_paf = os.path.join(self.overlaps_path, f"{self.genome_str}_ov.paf")  
        cis_paf_path = os.path.join(self.overlaps_path, f"{self.genome_str}_cis.paf")
        with gzip.open(merged_paf + '.gz', 'wt') as outfile:
            for paf_file in [cis_paf, trans_paf]:
                if os.path.exists(paf_file):
                    with open(paf_file, 'r') as infile:
                        outfile.write(infile.read())
        print(f"Merged overlaps saved to: {merged_paf}.gz")
        # Move and compress cis_paf
        with gzip.open(cis_paf_path + '.gz', 'wt') as outfile:
            with open(cis_paf, 'r') as infile:
                outfile.write(infile.read())
        print(f"Cis overlaps saved to: {cis_paf_path}.gz")

        # Step 2: Run RAFT to create pil-o-gram
        frag_prefix = os.path.join(self.tmp_path, 'raft_out')
        pil_o_gram_path = os.path.join(self.pile_o_grams_path, f'{self.genome_str}.coverage.txt')
        subprocess.run(f"{self.raft_path}/raft -e {raft_depth} -o {frag_prefix} -l 100000 {reads_file} {merged_paf}.gz", shell=True, check=True)
        shutil.move(frag_prefix + ".coverage.txt", pil_o_gram_path)

    def create_graphs(self):
        if self.real:
            full_fasta = os.path.join(self.full_reads_path, f'{self.genome_str}.fastq.gz')
        else:
            full_fasta = os.path.join(self.full_reads_path, f'{self.genome_str}.fasta.gz')

        gfa_unitig_output = os.path.join(self.gfa_unitig_path, f'{self.genome_str}.gfa')
        gfa_raw_output = os.path.join(self.gfa_raw_path, f'{self.genome_str}.gfa')

        subprocess.run(f'./hifiasm --prt-raw -r3 -o {self.hifiasm_dump}/tmp_asm -t{self.threads} {full_fasta}', shell=True, cwd=self.hifiasm_path)
        subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.raw.r_utg.gfa {gfa_raw_output}', shell=True, cwd=self.hifiasm_path)
        subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.r_utg.gfa {gfa_unitig_output}', shell=True, cwd=self.hifiasm_path)

        # Convert GFA to FASTA and gzip
        awk_command = "awk '/^S/{print \">\"$2;print $3}'"
        fasta_unitig_file = f"{self.fasta_unitig_path}/{self.genome_str}.fasta"
        fasta_raw_file = f"{self.fasta_raw_path}/{self.genome_str}.fasta"
        
        # Create and gzip unitig FASTA
        subprocess.run(f"{awk_command} {gfa_unitig_output} > {fasta_unitig_file}", shell=True, check=True)
        subprocess.run(f"gzip -f {fasta_unitig_file}", shell=True, check=True)
        print(f"Converted and moved {gfa_unitig_output} to {fasta_unitig_file}.gz")
        
        # Create and gzip raw FASTA
        subprocess.run(f"{awk_command} {gfa_raw_output} > {fasta_raw_file}", shell=True, check=True)
        subprocess.run(f"gzip -f {fasta_raw_file}", shell=True, check=True)
        print(f"Converted and moved {gfa_raw_output} to {fasta_raw_file}.gz")

        subprocess.run(f'rm {self.hifiasm_dump}/tmp_asm*', shell=True, cwd=self.hifiasm_path)

    def process_hic(self):
        """
        Runs nf-core/hic on the HiC reads provided in the data config on the unitigs fasta.
        This finds chromosomal contacts between different unitigs on the same chromosome/haplotype.
        TODO specify config for pipeline from here or as separate config ymls?
        """
        # set fasta param to what the filename out is
        self.nfcore_hic_config["fasta"] = self.fasta_unitig_path

        nf_conf = self._write_nf_config()
        nf_params = self._write_nf_params()
        samplesheet = self._write_samplesheet()

        call = ' '.join([self.nextflow_call, "-log nextflow.log run", self.hic_pipeline_path,
                        "-c", nf_conf,
                        "-params-file", nf_params, "-i", samplesheet,
                         "-o", self.hic_path, "-work", self.tmp_path, "-profile docker"])

        # call nextflow, this should finish when the pipeline is done
        subprocess.run(call, shell=True, cwd=self.dataset_path)

    def _write_nf_config(self, filename="nextflow.config") -> os.PathLike:
        """
        Writes the nextflow config file for nf-core/hic to a file.
        Allows all configuration to stay in dataset_config.yml.
        """
        with open(os.path.join(self.hic_path, filename), 'wt') as f:
            # nf config is stored as a text blurb in the yml, can just write directly
            f.write(self.nextflow_config)

        return filename

    def _write_nf_params(self, filename="params.yml") -> os.PathLike:
        """
        Writes the parameters for nf-core/hic to a yml file.
        Allows all configuration to stay in dataset_config.yml.
        """
        with open(os.path.join(self.hic_path, filename), 'wt') as f:
            yml.safe_dump(self.nfcore_hic_config)
        return filename

    def _write_samplesheet(self, filename="samplesheet.csv") -> os.PathLike:
        """
        Writes the samplesheet for nf-core/hic to a file.
        Multiple hic runs may be used per sample. They are written as individual lines to the sample sheet under the same sample name, and will be merged by nf-core/hic.
        Allows all configuration to stay in dataset_config.yml.
        """
        with open(os.path.join(self.hic_path, filename), 'wt') as f:
            f.write("sample,fastq_1,fastq_2\n")
            f.writelines([','.join([self.sample_name, f_pair[0], f_pair[1]]) for f_pair in self.hic_readsfiles_pairs])
        return filename

    def make_hic_edges(self):
        """
        Takes the output generated by the nf-core-HiC-step, and transforms it into a networkx graph containing contact edges.
        Uses the read to node mappings to make sure node IDs are the same as in the original graphs
        """
        from descongelador import export_connection_graph

        export_connection_graph(
                os.path.join(self.hic_path, "contact_maps", "cool", self.sample_name + ".1000000_balanced.cool"),
                os.path.join(self.hic_path, self.sample_name + "_hic.nx.pickle"),
                self.utg_to_read_path,
                self.read_to_node_path)

    def load_hic_edges(self):#-> nx.MultiGraph:
        ret = None
        with open(os.path.join(self.hic_path, self.sample_name + "_hic.nx.pickle")), 'rb') as f:
            ret = pickle.load(f)
        return ret

    def parse_gfa(self):
        nx_graph, read_seqs, node_to_read, read_to_node, utg_2_reads = self.only_from_gfa()
        # Save data
        self.pickle_save(node_to_read, self.node_to_read_path)
        self.pickle_save(read_to_node, self.read_to_node_path)
        self.pickle_save(utg_2_reads, self.utg_2_reads_path)

        
        """
        self.create_reads_fasta(read_seqs, self.chr_id)  # Add self.chr_id as an argument

        with open(os.path.join(self.node_to_read_path, f'{self.genome_str}.pkl'), 'rb') as f:
            node_to_read = pickle.load(f)
        with open(os.path.join(self.successor_dict_path, f'{self.genome_str}.pkl'), 'rb') as f:
            successor_dict = pickle.load(f)
        with open(os.path.join(self.reduced_reads_path, f'{self.genome_str}.pkl'), 'rb') as f:
            read_seqs = pickle.load(f)
        with open(os.path.join(self.read_to_node_path, f'{self.genome_str}.pkl'), 'rb') as f:
            read_to_node = pickle.load(f)
        """

        return nx_graph  
    
    def only_from_gfa(self):
        self.assembler = "hifiasm"
        training = not self.real
        gfa_path = os.path.join(self.gfa_unitig_path, f'{self.genome_str}.gfa')
        if self.real:
            reads_path = os.path.join(self.full_reads_path, f'{self.genome_str}.fastq.gz')
        else:
            reads_path = os.path.join(self.full_reads_path, f'{self.genome_str}.fasta.gz')
        read_headers = self.get_read_headers(reads_path)

        graph_nx = nx.DiGraph()
        read_to_node, node_to_read, old_read_to_utg = {}, {}, {}  ##########
        read_to_node2 = {}
        utg_2_reads = {}
        #edges_dict = {}
        read_lengths, read_seqs = {}, {}  # Obtained from the GFA
        read_idxs, read_strands, read_starts, read_ends, read_chrs, read_variants, variant_class = {}, {}, {}, {}, {}, {}, {}  # Obtained from the FASTA/Q headers
        edge_ids, prefix_lengths, overlap_lengths, overlap_similarities = {}, {}, {}, {}
        no_seqs_flag = self.assembler == 'raven'


        time_start = datetime.now()
        print(f'Starting to loop over GFA')
        with open(gfa_path) as ff:
            node_idx = 0
            edge_idx = 0
            # -------------------------------------------------
            # We assume that the first N lines start with "S"
            # And next M lines start with "L"
            # -------------------------------------------------
            all_lines = ff.readlines()
            line_idx = 0
            while line_idx < len(all_lines):
                line = all_lines[line_idx]
                line_idx += 1
                line = line.strip().split()
                # print(line)
                if line[0] == 'A':
                    # print(line)
                    old_read_to_utg[line[4]] = line[1]

                if line[0] == 'S':
                    if len(line) == 6:
                        tag, id, sequence, length, count, color = line
                    if len(line) == 5:
                        tag, id, sequence, length, color = line 
                    if len(line) == 4:
                        tag, id, sequence, length = line
                    if sequence == '*':
                        no_seqs_flag = True
                        sequence = '*' * int(length[5:])
                    sequence = Seq(sequence)  # This sequence is already trimmed in raven!
                    length = int(length[5:])

                    real_idx = node_idx
                    virt_idx = node_idx + 1
                    read_to_node[id] = (real_idx, virt_idx)
                    node_to_read[real_idx] = id
                    node_to_read[virt_idx] = id

                    graph_nx.add_node(real_idx)  # real node = original sequence
                    graph_nx.add_node(virt_idx)  # virtual node = rev-comp sequence

                    read_seqs[real_idx] = str(sequence)
                    read_seqs[virt_idx] = str(sequence.reverse_complement())

                    read_lengths[real_idx] = length
                    read_lengths[virt_idx] = length

                    if id.startswith('utg'):
                        # Store the original unitig ID before it gets modified
                        utg_id = id
                        ids = []
                        utg_2_reads[utg_id] = []  # Use original utg_id instead of id
                        while True:
                            line = all_lines[line_idx]
                            line = line.strip().split()
                            if line[0] != 'A':
                                break
                            line_idx += 1
                            tag = line[0]
                            utg_id_line = line[1]
                            read_orientation = line[3]
                            utg_to_read = line[4]
                            ids.append((utg_to_read, read_orientation))
                            utg_2_reads[utg_id].append(utg_to_read)  # Use original utg_id
                            read_to_node2[utg_to_read] = (real_idx, virt_idx)

                            id = ids
                            node_to_read[real_idx] = id
                            node_to_read[virt_idx] = id

                    if training:
                        #print(f"id: {id}")
                        if type(id) != list:
                            description = read_headers[id]
                            # desc_id, strand, start, end = description.split()
                            strand = re.findall(r'strand=(\+|\-)', description)[0]
                            strand = 1 if strand == '+' else -1
                            start = int(re.findall(r'start=(\d+)', description)[0])  # untrimmed
                            end = int(re.findall(r'end=(\d+)', description)[0])  # untrimmed
                            #chromosome = int(re.findall(r'chr=(\d+)', description)[0])
                            chromosome = re.findall(r'chr=([^\s]+)', description)[0]
                        else:
                            strands = []
                            starts = []
                            ends = []
                            chromosomes = []
                            for id_r, id_o in id:
                                description = read_headers[id_r]
                                # desc_id, strand, start, end = description.split()
                                strand_fasta = re.findall(r'strand=(\+|\-)', description)[0]
                                strand_fasta = 1 if strand_fasta == '+' else -1
                                strand_gfa = 1 if id_o == '+' else -1
                                strand = strand_fasta * strand_gfa

                                strands.append(strand)
                                start = int(re.findall(r'start=(\d+)', description)[0])  # untrimmed
                                starts.append(start)
                                end = int(re.findall(r'end=(\d+)', description)[0])  # untrimmed
                                ends.append(end)
                                #chromosome = int(re.findall(r'chr=(\d+)', description)[0])
                                chromosome = re.findall(r'chr=([^\s]+)', description)[0]
                                chromosomes.append(chromosome)

                            # What if they come from different strands but are all merged in a single unitig?
                            # Or even worse, different chromosomes? How do you handle that?
                            # I don't think you can. It's an error in the graph
                            strand = 1 if sum(strands) >= 0 else -1
                            start = min(starts)
                            end = max(ends)
                            chromosome = Counter(chromosomes).most_common()[0][0]

                        if self.diploid:
                            variant = re.findall(r'variant=([P|M])', description)[0]
                        else:
                            variant = '0'
                        read_strands[real_idx], read_strands[virt_idx] = strand, -strand
                        read_starts[real_idx] = read_starts[virt_idx] = start
                        read_ends[real_idx] = read_ends[virt_idx] = end
                        read_variants[real_idx] = read_variants[virt_idx] = variant
                        read_chrs[real_idx] = read_chrs[virt_idx] = chromosome

                    node_idx += 2

                if line[0] == 'L':
                    if len(line) == 6:
                        # raven, normal GFA 1 standard
                        tag, id1, orient1, id2, orient2, cigar = line
                    elif len(line) == 7:
                        # hifiasm GFA
                        tag, id1, orient1, id2, orient2, cigar, _ = line
                        id1 = re.findall(r'(.*):\d-\d*', id1)[0]
                        id2 = re.findall(r'(.*):\d-\d*', id2)[0]
                    elif len(line) == 8:
                        # hifiasm GFA newer
                        tag, id1, orient1, id2, orient2, cigar, _, _ = line
                    else:
                        raise Exception("Unknown GFA format!")

                    if orient1 == '+' and orient2 == '+':
                        src_real = read_to_node[id1][0]
                        dst_real = read_to_node[id2][0]
                        src_virt = read_to_node[id2][1]
                        dst_virt = read_to_node[id1][1]
                    if orient1 == '+' and orient2 == '-':
                        src_real = read_to_node[id1][0]
                        dst_real = read_to_node[id2][1]
                        src_virt = read_to_node[id2][0]
                        dst_virt = read_to_node[id1][1]
                    if orient1 == '-' and orient2 == '+':
                        src_real = read_to_node[id1][1]
                        dst_real = read_to_node[id2][0]
                        src_virt = read_to_node[id2][1]
                        dst_virt = read_to_node[id1][0]
                    if orient1 == '-' and orient2 == '-':
                        src_real = read_to_node[id1][1]
                        dst_real = read_to_node[id2][1]
                        src_virt = read_to_node[id2][0]
                        dst_virt = read_to_node[id1][0]

                    graph_nx.add_edge(src_real, dst_real)
                    graph_nx.add_edge(src_virt,
                                    dst_virt)  # In hifiasm GFA this might be redundant, but it is necessary for raven GFA

                    edge_ids[(src_real, dst_real)] = edge_idx
                    edge_ids[(src_virt, dst_virt)] = edge_idx + 1
                    edge_idx += 2

                    # -----------------------------------------------------------------------------------
                    # This enforces similarity between the edge and its "virtual pair"
                    # Meaning if there is A -> B and B^rc -> A^rc they will have the same overlap_length
                    # When parsing CSV that was not necessarily so:
                    # Sometimes reads would be slightly differently aligned from their RC pairs
                    # Thus resulting in different overlap lengths
                    # -----------------------------------------------------------------------------------

                    try:
                        ol_length = int(cigar[:-1])  # Assumption: this is overlap length and not a CIGAR string
                    except ValueError:
                        print('Cannot convert CIGAR string into overlap length!')
                        raise ValueError

                    overlap_lengths[(src_real, dst_real)] = ol_length
                    overlap_lengths[(src_virt, dst_virt)] = ol_length

                    prefix_lengths[(src_real, dst_real)] = read_lengths[src_real] - ol_length
                    prefix_lengths[(src_virt, dst_virt)] = read_lengths[src_virt] - ol_length
        
        elapsed = (datetime.now() - time_start).seconds
        print(f'Elapsed time: {elapsed}s')

        '''print(f'Calculating similarities...')
        overlap_similarities = self.calculate_similarities(edge_ids, read_seqs, overlap_lengths)
        print(f'Done!')
        elapsed = (datetime.now() - time_start).seconds
        print(f'Elapsed time: {elapsed}s')'''

        nx.set_node_attributes(graph_nx, read_lengths, 'read_length')
        nx.set_node_attributes(graph_nx, variant_class, 'variant_class')
        node_attrs = ['read_length', 'variant_class']

        #nx.set_edge_attributes(graph_nx, prefix_lengths, 'prefix_length')
        #nx.set_edge_attributes(graph_nx, overlap_lengths, 'overlap_length')
        #edge_attrs = ['prefix_length', 'overlap_length']

        if training:
            nx.set_node_attributes(graph_nx, read_strands, 'read_strand')
            nx.set_node_attributes(graph_nx, read_starts, 'read_start')
            nx.set_node_attributes(graph_nx, read_ends, 'read_end')
            nx.set_node_attributes(graph_nx, read_variants, 'read_variant')
            nx.set_node_attributes(graph_nx, read_chrs, 'read_chr')
            node_attrs.extend(['read_strand', 'read_start', 'read_end', 'read_variant', 'read_chr'])

        #nx.set_edge_attributes(graph_nx, overlap_similarities, 'overlap_similarity')
        #edge_attrs.append('overlap_similarity')

        # Create a dictionary of nodes and their direct successors
        successor_dict = {node: list(graph_nx.successors(node)) for node in graph_nx.nodes()}

        # Why is this the case? Is it because if there is even a single 'A' file in the .gfa, means the format is all 'S' to 'A' lines?
        if len(read_to_node2) != 0:
            read_to_node = read_to_node2

        # Print number of nodes and edges in graph
    
        return graph_nx, read_seqs, node_to_read, read_to_node, utg_2_reads

    def nx_utg_ftrs(self, nx_graph):
        strand = nx.get_node_attributes(nx_graph, 'read_strand')
        start = nx.get_node_attributes(nx_graph, 'read_start')
        end = nx.get_node_attributes(nx_graph, 'read_end')
        variant = nx.get_node_attributes(nx_graph, 'read_variant')
        chr = nx.get_node_attributes(nx_graph, 'read_chr')

        print(strand)
        
        exit()
    def create_pog_features(self, nx_graph):
        """
        Create pog_median, pog_min, and pog_max features for each node and edge based on pile o gram data
        from both full and cis coverage files
        """
        # Load the pile o gram files
        pog_file = os.path.join(self.pile_o_grams_path, f'{self.genome_str}.coverage.txt')

        # Load the read_to_node_id mapping
        read_to_node_path = os.path.join(self.read_to_node_path, f'{self.genome_str}.pkl')
        with open(read_to_node_path, 'rb') as f:
            read_to_node = pickle.load(f)

        # Initialize all nodes with default values of 1
        pog_median = {node: 1 for node in nx_graph.nodes()}
        pog_min = {node: 1 for node in nx_graph.nodes()}
        pog_max = {node: 1 for node in nx_graph.nodes()}

        # Store coverage data for each read
        read_coverages = {}
        cis_read_coverages = {}

        # Process full coverage file
        with open(pog_file, 'r') as f:
            for line in f:
                if line.startswith('read'):
                    parts = line.strip().split()
                    read_id = parts[1]
                    coverages = [int(part.split(',')[1]) for part in parts[2:] if ',' in part]
                    read_coverages[read_id] = coverages
                    if read_id in read_to_node:
                        node_ids = read_to_node[read_id]
                        median = np.median(coverages)
                        min_val = np.min(coverages)
                        max_val = np.max(coverages)
                        for node_id in node_ids:
                            pog_median[node_id] = median / self.depth
                            pog_min[node_id] = min_val / self.depth
                            pog_max[node_id] = max_val / self.depth

        nx.set_node_attributes(nx_graph, pog_median, 'pog_median')
        nx.set_node_attributes(nx_graph, pog_min, 'pog_min')
        nx.set_node_attributes(nx_graph, pog_max, 'pog_max')
        
        print(f"Added pile-o-gram features to {len(pog_median)} nodes.")

    def create_hh_features(self, nx_graph):
        """
        load reads. check starting position of read and compare with variant_starts list.
        once you reach the position on variant_start list that is higher then the read_start, you go one entry back.
        now check if the variant_ends entry with the same index is larger then the read start. If yes: there is a variation
        now check the read end position: it's the same but swapping ends and starts.
        """
        read_start = nx.get_node_attributes(nx_graph, 'read_start')
        read_end = nx.get_node_attributes(nx_graph, 'read_end')
        read_variant = nx.get_node_attributes(nx_graph, 'read_variant')
        read_chr = nx.get_node_attributes(nx_graph, 'read_chr')  # Get read_chr attribute

        if self.centromere_graph:
            for read in read_start.keys():
                chr_id = read_chr[read]  # Get chromosome from read_chr
                if read_variant[read] == 'M':
                    c_start, c_end = self.get_centromere_coords('M', chr_id)
                    read_start[read] = c_start + read_start[read]
                    read_end[read] = c_end + read_end[read]
                elif read_variant[read] == 'P':
                    c_start, c_end = self.get_centromere_coords('P', chr_id)
                    read_start[read] = c_start + read_start[read]
                    read_end[read] = c_end + read_end[read]
                else:
                    print(f"Wrong Read Variant detected {read_variant[read]}")
                    exit()

        # Create dictionaries to store variant data for each chromosome
        M_variants = {}
        P_variants = {}
        processed_chrs = set()

        hh_dict = {}
        hetero_nodes = 0

        # Process nodes first to identify which chromosomes we need to load
        for node in nx_graph.nodes():
            chr_id = read_chr[node]
            if chr_id not in processed_chrs:
                processed_chrs.add(chr_id)
                # Load VCF files for this chromosome
                M_path = os.path.join(self.vcf_path, f'chr{chr_id}_M.vcf.gz')
                P_path = os.path.join(self.vcf_path, f'chr{chr_id}_P.vcf.gz')
                
                print(f'Process Maternal reference variation file for chromosome {chr_id}')
                M_variants[chr_id] = self.process_vcf_file(M_path)
                print(f'Process Paternal reference variation file for chromosome {chr_id}')
                P_variants[chr_id] = self.process_vcf_file(P_path)

        print(f'Process graph nodes')
        for node in nx_graph.nodes():
            chr_id = read_chr[node]
            hh_dict[node] = "O"  # Homozygous Region
            
            if read_variant[node] == 'M':
                var_starts, var_ends = M_variants[chr_id]
            elif read_variant[node] == 'P':
                var_starts, var_ends = P_variants[chr_id]
            else:
                print('wrong read_variant')
                exit()
            
            variant_count = 0
            threshold = 1  # Fixed threshold 
            
            for i in range(len(var_starts)):
                if var_starts[i] > read_end[node]:
                    break
                if var_ends[i] >= read_start[node]:
                    # Calculate overlap between variant and read
                    overlap_start = max(var_starts[i], read_start[node])
                    overlap_end = min(var_ends[i], read_end[node])
                    variant_length = overlap_end - overlap_start + 1
                    
                    # Each base of the variant counts as one variant
                    variant_count += variant_length
                    
                    # Early exit if threshold is reached
                    if variant_count >= threshold:
                        hh_dict[node] = "E"  # Heterozygous Region
                        hetero_nodes += 1
                        break
                        
        print(f'{hetero_nodes} / {nx_graph.number_of_nodes()} nodes are in Heterozygous regions')

        nx.set_node_attributes(nx_graph, hh_dict, 'region')

        gt_hap = {}
        for node in nx_graph.nodes():
            if hh_dict[node] == 'O':
                gt_hap[node] = 0
            else:
                if read_variant[node] == 'M':
                    gt_hap[node] = 1
                if read_variant[node] == 'P':
                    gt_hap[node] = -1
        nx.set_node_attributes(nx_graph, gt_hap, 'gt_hap')

    def process_vcf_file(self, file_path):
        # Open the file (handle .gz compression if necessary)
        with gzip.open(file_path, 'rt') as f:
            # Iterate through the file until the header line starting with '#' is found
            vcf_df = pd.read_csv(f, sep='\t', comment='#', header=None,
                                 names=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT',
                                        'SAMPLE'])
        variant_starts = []
        variant_ends = []
        for index, row in vcf_df.iterrows():
            variant_length = len(row['REF'])
            variant_starts.append(row['POS'])
            variant_ends.append(row['POS'] + variant_length - 1)  # Adjusted to calculate the end position accurately
        print(len(variant_starts), len(variant_ends))
        return variant_starts, variant_ends
