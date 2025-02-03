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
from torch_geometric.data import Data
from torch_geometric.utils import degree
import scipy.stats
import scipy.signal

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
        #self.pile_o_grams_path = os.path.join(dataset_path, "pile_o_grams")
        self.utg_2_reads_path = os.path.join(dataset_path, "utg_2_reads")
        self.jellyfish_path = os.path.join(dataset_path, "jellyfish")

        # HiC stuff
        self.hic_pipeline_path = self.paths['hic_pipeline_path']
        self.hic_root_path = os.path.join(dataset_path, "hic")
        self.hic_readsfiles_pairs = self.paths['hic_readsfiles_pairs']

        self.nx_graphs_path = os.path.join(dataset_path, "nx_graphs")
        self.pyg_graphs_path = os.path.join(dataset_path, "pyg_graphs")
        self.utg_to_read_path = os.path.join(dataset_path, "utg_2_reads")
        self.unitig_2_node_path = os.path.join(dataset_path, "unitig_2_node")
        self.hic_graphs_path = os.path.join(dataset_path, "hic_graphs")
        self.merged_graphs_path = os.path.join(dataset_path, "merged_graphs")
        self.coverage_path = os.path.join(dataset_path, "coverage")

        self.deadends = {}
        self.gt_rescue = {}
        self.edge_info = {}

        for folder in [self.coverage_path, self.jellyfish_path, self.utg_2_reads_path, self.fasta_unitig_path, self.fasta_raw_path, self.full_reads_path, self.gfa_unitig_path, self.gfa_raw_path, self.nx_graphs_path,
                       self.pyg_graphs_path, self.read_descr_path, self.tmp_path, self.overlaps_path,
                       self.hic_graphs_path, self.merged_graphs_path, self.unitig_2_node_path]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        #self.edge_attrs = ['overlap_length', 'overlap_similarity', 'prefix_length']
        self.node_attrs = ['in_degree', 'read_length', 'cov_avg', 'cov_pct', 'cov_med', 'cov_std', 'read_gc']

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
    
    def load_nx_graph(self, multi=False):
        if multi:
            file_name = os.path.join(self.merged_graphs_path, f'{self.genome_str}.pkl')
        else:
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
        """
        fasta_unitig_file = f"{self.fasta_unitig_path}/{self.genome_str}.fasta"
        # for now nf-core/hic does not support compressed input, decompress it 
        #TODO make PR
        if not os.path.exists(fasta_unitig_file):
            subprocess.run(f"gunzip -k {fasta_unitig_file}.gz", shell=True, check=True)
        # set fasta param to what the filename out is
        self.nfcore_hic["fasta"] = fasta_unitig_file

        # create subfolder for sample in hic dir
        self.hic_sample_path = os.path.join(self.hic_root_path, self.genome_str)
        if not os.path.exists(self.hic_sample_path):
            if not os.path.exists(self.hic_sample_path):
                os.makedirs(self.hic_sample_path)


        nf_conf = self._write_nf_config()
        nf_params = self._write_nf_params()
        samplesheet = self._write_samplesheet()

        call = ' '.join([self.nextflow_call, "-log nextflow.log run", self.hic_pipeline_path,
                        "-c", nf_conf,
                        "-params-file", nf_params, "--input", samplesheet,
                         "--outdir", self.hic_sample_path, "-w", self.tmp_path, "-profile docker"])

        # call nextflow, this should finish when the pipeline is done
        #subprocess.run(call, shell=True, check=False, cwd=self.root_path)

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
            f.writelines([','.join([self.sample_name, f_pair[0], f_pair[1]]) + '\n' for f_pair in self.hic_readsfiles_pairs])

        return path

    def make_hic_edges(self):
        """
        Takes the output generated by the nf-core-HiC-step, and transforms it into a networkx graph containing contact edges.
        Uses the read to node mappings to make sure node IDs are the same as in the original graphs
        """
        from descongelador import export_connection_graph

        export_connection_graph(
                os.path.join(self.hic_sample_path, "contact_maps", "cool", self.sample_name + ".1000000_balanced.cool"),
                os.path.join(self.hic_sample_path, self.sample_name + "_hic.nx.pickle"),
                os.path.join(self.unitig_2_node_path, self.genome_str + '.pkl'))

    def load_hic_edges(self):#-> nx.MultiGraph:
        ret = None
        with open(os.path.join(self.hic_sample_path, self.sample_name + "_hic.nx.pickle"), 'rb') as f:
            ret = pickle.load(f)
        return ret

    def merge_graphs(self, nx_graph, hic_graph):
        nx.set_edge_attributes(hic_graph, "hic", "type")
        nx.set_edge_attributes(nx_graph, "overlap", "type")
        return nx.compose(nx.MultiGraph(nx_graph), nx.MultiGraph(hic_graph))

    def parse_gfa(self):
        nx_graph, read_seqs, unitig_2_node, utg_2_reads = self.only_from_gfa()
        # Save data
        self.pickle_save(unitig_2_node, self.unitig_to_node_path)
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
        unitig_2_node = {}
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
                        unitig_2_node[utg_id] = (real_idx, virt_idx)

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

                            id = ids
                            node_to_read[real_idx] = id
                            node_to_read[virt_idx] = id
                    else:
                        print(f"Unknown line type: {line}")
                        exit()
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


        # Print number of nodes and edges in graph
    
        return graph_nx, read_seqs, unitig_2_node, utg_2_reads

    def nx_utg_ftrs(self, nx_graph):
        strand = nx.get_node_attributes(nx_graph, 'read_strand')
        start = nx.get_node_attributes(nx_graph, 'read_start')
        end = nx.get_node_attributes(nx_graph, 'read_end')
        variant = nx.get_node_attributes(nx_graph, 'read_variant')
        chr = nx.get_node_attributes(nx_graph, 'read_chr')
        print(strand)
        exit()

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

    def create_jellyfish_features(self, nx_graph):
        """
        Create k-mer coverage features for each node using Jellyfish and add them as node attributes.
        Uses unitig fasta file as input.
        """
        print("Creating jellyfish features...")
        def decompress_file(input_file, output_file):
            """Decompress a gzipped file to a temporary file."""
            with gzip.open(input_file, "rt") as infile, open(output_file, "w") as outfile:
                for line in infile:
                    outfile.write(line)
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
        
        # Define paths
        kmer_size = 31  # Standard k-mer size
        unitig_fasta = os.path.join(self.fasta_unitig_path, f'{self.genome_str}.fasta.gz')
        kmer_hist_file = os.path.join(self.jellyfish_path, f'{self.genome_str}_kmers.txt')
        coverage_output = os.path.join(self.jellyfish_path, f'{self.genome_str}_coverage.csv')
        
        # Run jellyfish to get k-mer frequencies
        print("Running jellyfish...")
        run_jellyfish(unitig_fasta, kmer_size, kmer_hist_file)
        
        # Calculate coverage statistics for each read
        print("Calculating coverage statistics...")
        calculate_read_coverage(unitig_fasta, kmer_hist_file, coverage_output, kmer_size)
        
        # Read coverage statistics and add to graph
        coverage_stats = {}
        with open(coverage_output, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                read_id, median_cov, variance, max_cov, filtered_avg, filtered_median = line.strip().split(',')
                coverage_stats[read_id] = {
                    'kmer_median': float(median_cov),
                    'kmer_variance': float(variance),
                    'kmer_max': float(max_cov),
                    'kmer_filtered_avg': float(filtered_avg),
                    'kmer_filtered_median': float(filtered_median)
                }
        
        # Add coverage statistics as node attributes
        print("Adding coverage statistics as node attributes...")
        for node in nx_graph.nodes():
            node_data = nx_graph.nodes[node]
            node_id = str(node)  # Convert node ID to string to match read IDs
            
            if node_id in coverage_stats:
                for stat_name, stat_value in coverage_stats[node_id].items():
                    node_data[stat_name] = stat_value
            else:
                # Set default values if node not found in coverage stats
                for stat_name in ['kmer_median', 'kmer_variance', 'kmer_max', 'kmer_filtered_avg', 'kmer_filtered_median']:
                    node_data[stat_name] = 0.0
        
        # Add new attributes to node_attrs list for normalization later
        self.node_attrs.extend(['kmer_median', 'kmer_variance', 'kmer_max', 'kmer_filtered_avg', 'kmer_filtered_median'])
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for temp_file in [kmer_hist_file, coverage_output]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        print("Jellyfish features created successfully")

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

    def calculate_sequence_features(self, nx_graph):
        """
        Calculate sequence-based features for each node
        Returns dict of feature dictionaries
        """
        features = {
            'gc_content': {},
            'seq_complexity': {},
            'homopolymer_runs': {},
            'palindrome_count': {},
            'kmer_entropy': {}
        }
        
        def calculate_gc_content(seq):
            """Calculate GC content of sequence"""
            gc_count = seq.count('G') + seq.count('C')
            return gc_count / len(seq) if len(seq) > 0 else 0
        
        def calculate_sequence_complexity(seq):
            """Calculate sequence complexity using k-mer entropy"""
            k = 3  # Use 3-mers
            kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
            kmer_freq = Counter(kmers)
            total_kmers = len(kmers)
            entropy = 0
            for count in kmer_freq.values():
                prob = count / total_kmers
                entropy -= prob * np.log2(prob)
            return entropy
        
        def find_homopolymer_runs(seq):
            """Count homopolymer runs longer than 3 bases"""
            pattern = r'([ACGT])\1{2,}'
            runs = re.findall(pattern, seq)
            return len(runs)
        
        def count_palindromes(seq, min_len=4):
            """Count palindromic sequences of minimum length"""
            count = 0
            for i in range(len(seq)-min_len+1):
                for j in range(i+min_len, len(seq)+1):
                    subseq = seq[i:j]
                    if subseq == subseq[::-1]:
                        count += 1
            return count
        
        def calculate_kmer_entropy(seq, k=2):
            """Calculate entropy of k-mer distribution"""
            kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
            kmer_freq = Counter(kmers)
            total_kmers = len(kmers)
            entropy = 0
            for count in kmer_freq.values():
                prob = count / total_kmers
                entropy -= prob * np.log2(prob)
            return entropy
        
        # Get sequences for each node
        fasta_path = os.path.join(self.fasta_unitig_path, f'{self.genome_str}.fasta.gz')
        with gzip.open(fasta_path, 'rt') as f:
            for record in SeqIO.parse(f, 'fasta'):
                node_id = int(record.id)
                seq = str(record.seq)
                
                # Calculate features
                features['gc_content'][node_id] = calculate_gc_content(seq)
                features['seq_complexity'][node_id] = calculate_sequence_complexity(seq)
                features['homopolymer_runs'][node_id] = find_homopolymer_runs(seq)
                features['palindrome_count'][node_id] = count_palindromes(seq)
                features['kmer_entropy'][node_id] = calculate_kmer_entropy(seq)
        
        # Set default values for nodes without sequence data
        for node in nx_graph.nodes():
            for feat_dict in features.values():
                if node not in feat_dict:
                    feat_dict[node] = 0.0
                
        return features

    def calculate_topological_features(self, nx_graph):
        """
        Calculate graph topology-based features for each node
        Returns dict of feature dictionaries
        """
        features = {
            'clustering_coeff': {},
            'betweenness_cent': {},
            'closeness_cent': {},
            'eigenvector_cent': {},
            'pagerank': {},
            'triangle_count': {},
            'avg_neighbor_degree': {},
            'local_efficiency': {}
        }
        
        # Convert multigraph to simple graph for some calculations
        simple_graph = nx.Graph()
        for u, v, data in nx_graph.edges(data=True):
            if simple_graph.has_edge(u, v):
                # If edge exists, update weight
                simple_graph[u][v]['weight'] += data.get('weight', 1.0)
            else:
                # Add new edge with weight
                simple_graph.add_edge(u, v, weight=data.get('weight', 1.0))
        
        print("Calculating clustering coefficients...")
        clustering = nx.clustering(simple_graph)
        features['clustering_coeff'] = clustering
        
        print("Calculating betweenness centrality...")
        betweenness = nx.betweenness_centrality(simple_graph, weight='weight')
        features['betweenness_cent'] = betweenness
        
        print("Calculating closeness centrality...")
        closeness = nx.closeness_centrality(simple_graph, distance='weight')
        features['closeness_cent'] = closeness
        
        print("Calculating eigenvector centrality...")
        try:
            eigenvector = nx.eigenvector_centrality(simple_graph, weight='weight')
            features['eigenvector_cent'] = eigenvector
        except:
            print("Warning: Eigenvector centrality calculation failed, using default values")
            features['eigenvector_cent'] = {node: 0.0 for node in nx_graph.nodes()}
        
        print("Calculating PageRank...")
        pagerank = nx.pagerank(simple_graph, weight='weight')
        features['pagerank'] = pagerank
        
        print("Calculating triangle counts...")
        triangles = nx.triangles(simple_graph)
        features['triangle_count'] = triangles
        
        print("Calculating average neighbor degree...")
        avg_neighbor_degree = nx.average_neighbor_degree(simple_graph, weight='weight')
        features['avg_neighbor_degree'] = avg_neighbor_degree
        
        print("Calculating local efficiency...")
        # Local efficiency is the average efficiency of local subgraphs
        local_efficiency = {}
        for node in nx_graph.nodes():
            neighbors = list(nx_graph.neighbors(node))
            if len(neighbors) > 1:
                subgraph = nx_graph.subgraph(neighbors)
                try:
                    efficiency = nx.global_efficiency(subgraph)
                except:
                    efficiency = 0.0
            else:
                efficiency = 0.0
            local_efficiency[node] = efficiency
        features['local_efficiency'] = local_efficiency
        
        # Normalize features to [0,1] range
        for feat_name, feat_dict in features.items():
            values = np.array(list(feat_dict.values()))
            if len(values) > 0:
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val > min_val:
                    for node in feat_dict:
                        feat_dict[node] = (feat_dict[node] - min_val) / (max_val - min_val)
        
        return features

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

        """# Create subgraph with nodes where read_chr==1
        nodes_to_keep = [n for n, attr in single_stranded.nodes(data=True) if attr['read_chr'] == "1"]
        chr1_subgraph = single_stranded.subgraph(nodes_to_keep).copy()
        
        # Get connected components
        components = list(nx.connected_components(chr1_subgraph))
        
        # Count components by size
        size_counts = {}
        for comp in components:
            size = len(comp)
            size_counts[size] = size_counts.get(size, 0) + 1
            
        print(f"\nTotal number of components: {len(components)}")
        print("\nComponent size distribution:")
        for size in sorted(size_counts.keys()):
            print(f"{size_counts[size]}x length {size}")
            
        # Check if chr1 is fully connected
        if len(components) > 1:
            print(f"\nWarning: Chromosome 1 is split into {len(components)} components")
            print("Component sizes:", [len(comp) for comp in components])
        else:
            print("\nChromosome 1 is fully connected")
        
        single_stranded = chr1_subgraph  # Replace original graph with chr1 subgraph
        exit()"""
        return single_stranded
    
    
    def save_to_dgl_and_pyg(self, nx_graph):
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
        
        # Z-score normalize edge weights separately for each type
        for type_idx in range(len(edge_type_map)):
            
            # Create mask for current edge type
            type_mask = edge_type == type_idx
            
            # Get weights for current type
            type_weights = edge_weight[type_mask]
            
            # Compute mean and std for current type
            type_mean = torch.mean(type_weights)
            type_std = torch.std(type_weights)
            
            # Normalize weights for current type if std is not 0
            if type_std != 0:
                edge_weight[type_mask] = (type_weights - type_mean) / type_std
                
            print(f"\nEdge weight statistics after normalization for type {list(edge_type_map.keys())[type_idx]}:")
            print(f"Mean: {torch.mean(edge_weight[type_mask])}")
            print(f"Std:  {torch.std(edge_weight[type_mask])}")
        
        # Now compute degrees after edge_index is created
        out_degrees = degree(edge_index[0], num_nodes=num_nodes)
        in_degrees = degree(edge_index[1], num_nodes=num_nodes)
        
        # Add degree information to the graph
        for node in range(num_nodes):
            nx_graph.nodes[node]['in_degree'] = float(in_degrees[node])
            nx_graph.nodes[node]['out_degree'] = float(out_degrees[node])
        
        # Create node features using all features in self.node_attrs
        features = []
        for node in range(num_nodes):
            node_features = []
            for feat in self.node_attrs:
                if feat in nx_graph.nodes[node]:
                    node_features.append(float(nx_graph.nodes[node][feat]))
                else:
                    print(f"Warning: Feature {feat} not found for node {node}")
                    node_features.append(0.0)  # Default value if feature is missing
            features.append(node_features)
        x = torch.tensor(features, dtype=torch.float)

        gt_hap = [nx_graph.nodes[node]['gt_hap'] for node in range(num_nodes)]
        gt_tensor = torch.tensor(gt_hap, dtype=torch.long)
        
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

        # Normalize features
        self.normalize_ftrs(pyg_data, ['in_degree', 'read_length', 'cov_std', 'read_gc'],
                            method='zscore')
        self.normalize_ftrs(pyg_data, ['cov_avg', 'cov_med'],
                            method='mean')
        # leave cov_pct unnormalized

        # Save PyG graph
        pyg_file = os.path.join(self.pyg_graphs_path, f'{self.genome_str}.pt')
        torch.save(pyg_data, pyg_file)
        print(f"Saved PyG graph of {self.genome_str}")

    def normalize_ftrs(self, pyg_data, normalize_attrs, method='zscore'):
        """
        Z-score normalize specified node features
        
        Args:
            pyg_data: PyG data object
            normalize_attrs: List of attribute names to normalize
        """
        print("\nFeature names in order:")
        print(self.node_attrs)
        
        print("\nFeature statistics before normalization:")
        print(f"Features to normalize: {normalize_attrs}")
        print(f"Data tensor shape: {pyg_data.x.shape}")
        
        # Get indices of features to normalize
        feature_indices = [self.node_attrs.index(attr) for attr in normalize_attrs]
        print(f"Feature indices to normalize: {feature_indices}")
        print(f"Mean: {torch.mean(pyg_data.x, dim=0)}")
        print(f"Std:  {torch.std(pyg_data.x, dim=0)}")
        
        # Z-score normalize specified features
        for i in feature_indices:
            if i < pyg_data.x.shape[1]:  # Check if index is valid
                mean = torch.mean(pyg_data.x[:, i])
                std = torch.std(pyg_data.x[:, i])
                if method == 'zscore' and std != 0:  # Avoid division by zero
                    pyg_data.x[:, i] = (pyg_data.x[:, i] - mean) / std
                elif method == 'mean' and mean != 0:
                    pyg_data.x[:, i] = pyg_data.x[:, i] / mean
        
        print("\nFeature statistics after normalization:")
        print(f"Mean: {torch.mean(pyg_data.x, dim=0)}")
        print(f"Std:  {torch.std(pyg_data.x, dim=0)}")
        
        return pyg_data

    def add_hic_neighbor_weights(self, nx_graph):
        """
        Add a node feature that sums the weights of all HiC edges connected to each node
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
                    total_weight += data.get('weight', 1.0)
            hic_weights_sum[node] = total_weight
        
        # Add the feature to the graph
        nx.set_node_attributes(nx_graph, hic_weights_sum, 'hic_neighbor_weight')
        
        # Add to node_attrs list if not already present
        if 'hic_neighbor_weight' not in self.node_attrs:
            self.node_attrs.append('hic_neighbor_weight')
        
        print(f"Added HiC neighbor weights feature to {len(hic_weights_sum)} nodes")
        
        return nx_graph

    def create_pileup_features(self, nx_graph):
        """
        Create coverage features using BBMap's pileup.sh script and add them to the NetworkX graph.
        Uses the unitig FASTA file as reference and maps the full reads against it.
        
        Features added:
        - cov_avg: Average fold coverage
        - cov_pct: Percent of bases covered
        - cov_med: Median fold coverage
        - coverage_stdev: Standard deviation of coverage
        - read_gc: GC content percentage
        """
        print("Creating pileup coverage features...")
        
        # Define input/output paths
        unitig_fasta = os.path.join(self.fasta_unitig_path, f'{self.genome_str}.fasta.gz')
        if self.real:
            reads_file = os.path.join(self.full_reads_path, f'{self.genome_str}.fastq.gz')
        else:
            reads_file = os.path.join(self.full_reads_path, f'{self.genome_str}.fasta.gz')
        coverage_stats = os.path.join(self.coverage_path, f'{self.genome_str}_coverage_stats.txt')
        
        # First map reads to unitigs using BBMap
        # Then pipe the output to pileup.sh
        #TODO run bbmap per hand first
        cmd = (f"bbmap.sh -Xmx20g ref={unitig_fasta} in={reads_file} nodisk=t "
               f"| pileup.sh -Xmx20g in=stdin.sam out={coverage_stats} stdev=t secondary=f samstreamer=t")
        
        try:
            # Run the pipeline
            subprocess.run(cmd, shell=True, check=True)
            
            # Read the coverage statistics file
            coverage_data = {}
            with open(coverage_stats, 'r') as f:
                # Skip header line (starts with #)
                header = f.readline().strip('#').strip().split('\t')
                
                # Create index mapping for each column
                col_idx = {
                    'id': header.index('ID'),
                    'cov_avg': header.index('Avg_fold'),
                    'cov_pct': header.index('Covered_percent'),
                    'cov_med': header.index('Median_fold'),
                    'cov_std': header.index('Std_Dev'),
                    'read_gc': header.index('Read_GC')
                }
                
                # Parse data lines
                for line in f:
                    fields = line.strip().split('\t')
                    node_id = int(fields[col_idx['id']])
                    
                    coverage_data[node_id] = {
                        'cov_avg': float(fields[col_idx['cov_avg']]) / self.depth,  # Normalize by expected depth
                        'cov_pct': float(fields[col_idx['cov_pct']]),
                        'cov_med': float(fields[col_idx['cov_med']]) / self.depth,  # Normalize by expected depth
                        'cov_stdev': float(fields[col_idx['cov_std']]) / self.depth,  # Normalize by expected depth
                        'read_gc': float(fields[col_idx['read_gc']])
                    }
            
            # Add features to graph
            for feature_name in ['cov_avg', 'cov_pct', 'cov_med', 'cov_std', 'read_gc']:
                feature_dict = {node: coverage_data.get(node, {}).get(feature_name, 0.0) 
                              for node in nx_graph.nodes()}
                nx.set_node_attributes(nx_graph, feature_dict, feature_name)
                
                # Add to node_attrs list for normalization
                if feature_name not in self.node_attrs:
                    self.node_attrs.append(feature_name)
            
            print(f"Added pileup coverage features to {len(coverage_data)} nodes")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running pileup.sh: {e}")
            raise

#################################################
####### Welcome to the Code Graveyard X.X #######
#################################################


"""
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
        subprocess.run(f"{self.raft_path}/raft -e {raft_depth} -o {frag_prefix} -l 1000000 {reads_file} {merged_paf}.gz", shell=True, check=True)
        shutil.move(frag_prefix + ".coverage.txt", pil_o_gram_path)
        # Gzip the fragmented reads file
        subprocess.run(f"gzip -f {frag_prefix}.reads.fasta", shell=True, check=True)



    def create_pog_features(self, nx_graph):
        " " "
        Create pog_median, pog_min, and pog_max features for each node and edge based on pile o gram data
        from both full and cis coverage files
        " " "
        # Load the pile o gram files
        pog_file = os.path.join(self.pile_o_grams_path, f'{self.genome_str}.coverage.txt')

        # Load the read_to_node_id mapping
        read_to_node_path = os.path.join(self.unitig_to_node_path, f'{self.genome_str}.pkl')
        with open(read_to_node_path, 'rb') as f:
            read_to_node = pickle.load(f)

        # Initialize all nodes with default values of 1
        pog_median = {node: 1 for node in nx_graph.nodes()}
        pog_min = {node: 1 for node in nx_graph.nodes()}
        pog_max = {node: 1 for node in nx_graph.nodes()}

        # Store coverage data for each read
        read_coverages = {}
        fasta_path = os.path.join(self.fasta_unitig_path, f'{self.genome_str}.fasta.gz')

        # Load FASTA IDs in order
        fasta_ids = []
        with gzip.open(fasta_path, 'rt') as f:
            for record in SeqIO.parse(f, 'fasta'):
                fasta_ids.append(record.id)
        
        # Count reads in pile-o-gram file
        pog_count = 0
        with open(pog_file, 'r') as f:
            for line in f:
                if line.startswith('read'):
                    pog_count += 1
                    
        print(f"\nNumber of reads in fasta file: {len(fasta_ids)}")
        print(f"Number of reads in pile-o-gram file: {pog_count}")
        if len(fasta_ids) != pog_count:
            print("Warning: Mismatch between number of reads in fasta and pile-o-gram files")
            exit()
        
        # Process full coverage file
        pog_line_count = 0
        with open(pog_file, 'r') as f:
            for line in f:
                if line.startswith('read'):
                    parts = line.strip().split()
                    # Use FASTA ID instead of POG file ID
                    read_id = fasta_ids[pog_line_count]
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
                    else:
                        print(f"Read {read_id} not found in read_to_node mapping")
                        exit()
                    pog_line_count += 1

        # Calculate statistics for pog_median
        " " "median_values = list(pog_median.values())
        print("\nPile-o-gram median statistics:")
        print(f"Mean: {np.mean(median_values):.3f}")
        print(f"Std: {np.std(median_values):.3f}")
        print(f"Min: {np.min(median_values):.3f}")
        print(f"Max: {np.max(median_values):.3f}")
        print(f"25th percentile: {np.percentile(median_values, 25):.3f}")
        print(f"50th percentile: {np.percentile(median_values, 50):.3f}")
        print(f"75th percentile: {np.percentile(median_values, 75):.3f}")
        exit()" " "
        
        nx.set_node_attributes(nx_graph, pog_median, 'pog_median')
        nx.set_node_attributes(nx_graph, pog_min, 'pog_min')
        nx.set_node_attributes(nx_graph, pog_max, 'pog_max')
        
        # Add capped version of pog_median
        pog_median_capped = {node: min(3, value) for node, value in pog_median.items()}
        nx.set_node_attributes(nx_graph, pog_median_capped, 'pog_median_capped')
        
        print(f"Added pile-o-gram features to {len(pog_median)} nodes.")

    def calculate_coverage_statistics(self, nx_graph):
        " " "
        Calculate extended coverage statistics for each node
        Returns dict of feature dictionaries
        " " "
        features = {
            'coverage_std': {},
            'coverage_skew': {},
            'coverage_kurtosis': {},
            'coverage_q1': {},
            'coverage_q3': {},
            'coverage_iqr': {},
            'coverage_cv': {},  # coefficient of variation
            'coverage_peaks': {},
            'coverage_density': {}
        }
        
        # Load coverage data
        pog_file = os.path.join(self.pile_o_grams_path, f'{self.genome_str}.coverage.txt')
        read_to_node_path = os.path.join(self.unitig_to_node_path, f'{self.genome_str}.pkl')
        
        with open(read_to_node_path, 'rb') as f:
            read_to_node = pickle.load(f)
        
        # Process coverage file
        with open(pog_file, 'r') as f:
            for line in f:
                if line.startswith('read'):
                    parts = line.strip().split()
                    read_id = parts[1]
                    coverages = np.array([int(part.split(',')[1]) for part in parts[2:] if ',' in part])
                    
                    if read_id in read_to_node:
                        node_ids = read_to_node[read_id]
                        
                        # Calculate statistics
                        std = np.std(coverages)
                        skew = scipy.stats.skew(coverages)
                        kurt = scipy.stats.kurtosis(coverages)
                        q1, q3 = np.percentile(coverages, [25, 75])
                        iqr = q3 - q1
                        cv = std / np.mean(coverages) if np.mean(coverages) != 0 else 0
                        
                        # Count peaks (local maxima)
                        peaks = len(scipy.signal.find_peaks(coverages)[0])
                        
                        # Calculate coverage density (% positions above mean)
                        density = np.mean(coverages > np.mean(coverages))
                        
                        # Assign to all associated nodes
                        for node_id in node_ids:
                            features['coverage_std'][node_id] = std / self.depth
                            features['coverage_skew'][node_id] = skew
                            features['coverage_kurtosis'][node_id] = kurt
                            features['coverage_q1'][node_id] = q1 / self.depth
                            features['coverage_q3'][node_id] = q3 / self.depth
                            features['coverage_iqr'][node_id] = iqr / self.depth
                            features['coverage_cv'][node_id] = cv
                            features['coverage_peaks'][node_id] = peaks
                            features['coverage_density'][node_id] = density
        
        # Set default values for nodes without coverage data
        for node in nx_graph.nodes():
            for feat_dict in features.values():
                if node not in feat_dict:
                    feat_dict[node] = 0.0
                
        return features

"""
