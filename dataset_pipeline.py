import os
import subprocess
import pickle
import gzip
import edlib
import networkx as nx
import torch
import yaml
from pyliftover import LiftOver
import pandas as pd
from tqdm import tqdm
import dgl
import shutil
from collections import Counter
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
#from torch_geometric.utils import from_dgl
from collections import deque, defaultdict
from tqdm import tqdm  
import random
from multiprocessing import Pool, Manager
import itertools

class HicDatasetCreator:
    def __init__(self, ref_path, dataset_path, data_config='dataset.yml'):
        with open(data_config) as file:
            config = yaml.safe_load(file)
        #self.full_dataset, self.val_dataset, self.train_dataset = utils.create_dataset_dicts(data_config=data_config)
        self.paths = config['paths']
        gen_config = config['gen_config']
        self.genome_str = ""
        self.genome = "hg002"
        self.gen_step_config = config['gen_steps']
        self.centromere_dict = self.load_centromere_file()
        self.load_chromosome("", "", ref_path)
        self.hifiasm_path = self.paths['hifiasm_path']
        self.hifiasm_dump = self.paths['hifiasm_dump']
        self.raven_path = self.paths['raven_path']
        self.raft_path = self.paths['raft_path']
        self.yak_path = self.paths['yak_path']

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
        self.gfa_graphs_path = os.path.join(dataset_path, "gfa_graphs")
        self.nx_graphs_path = os.path.join(dataset_path, "nx_graphs")
        self.pyg_graphs_path = os.path.join(dataset_path, "pyg_graphs")
        self.paf_path = os.path.join(dataset_path, "paf")
        self.read_to_node_path = os.path.join(dataset_path, "read_to_node")
        self.node_to_read_path = os.path.join(dataset_path, "node_to_read")

        self.deadends = {}
        self.gt_rescue = {}
        self.edge_info = {}

        for folder in [self.full_reads_path, self.gfa_graphs_path, self.nx_graphs_path, self.dgl_graphs_path,
                       self.pyg_graphs_path, self.read_descr_path, self.paf_path, self.tmp_path, self.yak_files_path, self.original_reads_path, self.read_to_node_path, self.node_to_read_path]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        self.edge_attrs = ['overlap_length', 'overlap_similarity', 'prefix_length']
        self.node_attrs = ['read_length']
        if not self.real:
            self.node_attrs.extend('gt_hap')
            add_edge_attrs = ['strand_change', 'skip_forward', 'skip_backward', 'cross_chr', 'gt_bin'] #'gt_17c', 
            self.edge_attrs.extend(add_edge_attrs)

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
        
    def load_fasta(self, file_path, dict=False):
        """Load sequences from a FASTA/FASTQ file, handling both compressed and uncompressed files.
        Returns a list of SeqRecord objects or dict mapping ids to sequences if dict=True."""
        filetype = self._get_filetype(file_path)
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as handle:
                records = SeqIO.parse(handle, filetype)
                if dict:
                    return {record.id: record.seq for record in records}
                return list(records)
        else:
            records = SeqIO.parse(file_path, filetype)
            if dict:
                return {record.id: record.seq for record in records}
            return list(records)
        
    def create_reads_fasta(self, read_seqs, chr_id):
        seq_records = []
        for read_id, sequence in read_seqs.items():
            seq_record = SeqRecord(Seq(sequence), id=str(read_id), description="")
            seq_records.append(seq_record)
        seq_record_path = os.path.join(self.reduced_reads_path, f'{self.genome_str}.fasta')
        SeqIO.write(seq_records, seq_record_path, "fasta")


    def save_to_dgl_and_pyg(self, nx_graph):
        #self.positional_node_ftrs(nx_graph)
        print()
        print(f"Total nodes in graph: {nx_graph.number_of_nodes()}")
        yak_labels = nx.get_node_attributes(nx_graph, 'yak_m')
        print(f"Number of nodes with yak labels: {len(yak_labels)}")
        graph_dgl = dgl.from_networkx(nx_graph, node_attrs=self.node_attrs, edge_attrs=self.edge_attrs)
        dgl.save_graphs(os.path.join(self.dgl_graphs_path, f'{self.genome_str}.dgl'), graph_dgl)
        print(f"Saved DGL graph of {self.genome_str}")
        '''pyg_data = from_dgl(graph_dgl) 
        # save pyggraph
        pyg_file = os.path.join(self.pyg_graphs_path, f'{self.genome_str}.pt')
        torch.save(pyg_data, pyg_file)'''


    def parse_read(self, read):
            if self.real:
                id = read.id
                train_desc = ""
            else:
                description = read.description.split()
                id = description[0]
                train_desc = read.description

            seqs = (str(read.seq), str(Seq(read.seq).reverse_complement()))
            return id, seqs, train_desc 
        
    def parse_fasta(self):
        print("Parsing FASTA...")
        path = os.path.join(self.raft_reads_path, f'{self.genome_str}.reads.fasta')
        data, train_data = {}, {}
        try:
            with open(path, 'rt') as handle:
                rows = list(SeqIO.parse(handle, 'fasta'))

            with Pool(15) as pool:
                results = pool.imap_unordered(self.parse_read, rows, chunksize=50)
                for id, seqs, train_desc in tqdm(results, total=len(rows), ncols=120):
                    data[id] = seqs

                    #if not self.real:
                    #    train_data[id] = train_desc

            with open(os.path.join(self.full_reads_path, f"{self.genome_str}.pkl"), "wb") as p:
                pickle.dump(data, p)

        except Exception as e:
            print(f"An error occurred while parsing the FASTA file: {str(e)}")
            raise

        #return data#, train_data

    def simulate_reads(self):
        if not self.real:
            self.simulate_pbsim_reads()
        if self.raft:
            self.run_raft()
            if not self.real:
                self.raft_process_fasta_files()
        if self.prep_decoding:
            # Save the dictionary as a pickle file
            self.parse_fasta()

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

    def parse_sequence_header(self, header):
        match = re.match(r"read=\d+,(\d+),pos_on_original_read=(\d+)-(\d+)", header)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return None, None, None

    def parse_description_header(self, header):
        match = re.match(r"(\d+) strand=(.) start=(\d+) end=(\d+) variant=(\w+) chr=(.+)", header)
        if match:
            return {
                "id": int(match.group(1)),
                "strand": match.group(2),
                "start": int(match.group(3)),
                "end": int(match.group(4)),
                "variant": match.group(5),
                "chr": match.group(6)
            }
        print(f"Failed to parse header: {header}", file=sys.stderr)
        return None

    def run_raft(self):
        if self.real:
            reads_file = os.path.join(self.full_reads_path, f'{self.genome_str}.fastq.gz')
        else:
            reads_file = os.path.join(self.full_reads_path, f'{self.genome_str}.fasta.gz')
        if self.diploid:
            raft_depth = 2 * self.depth
        else:
            raft_depth = self.depth
        # Step 1: Run hifiasm for error correction
        ec_prefix = os.path.join(self.hifiasm_dump, f"{self.genome_str}_ec")
        if self.real:
            subprocess.run(f"{self.hifiasm_path}/hifiasm -o {ec_prefix} -t{self.threads} --write-ec {reads_file}", shell=True, check=True)
            #exit()
            # Move error-corrected reads to permanent storage
            ec = os.path.join(self.full_reads_ec_path, f"{ec_prefix}_ec.fa")
            shutil.move(f"{ec_prefix}.ec.fa", ec)
            # Gzip the error-corrected reads file
            subprocess.run(f"gzip -f {ec}", shell=True, check=True)
            reads_file = ec + '.gz'

        # Step 2: Run hifiasm to obtain all-vs-all read overlaps
        ov_prefix = os.path.join(self.hifiasm_dump, f"getOverlaps")
        #ec_reads = os.path.join(self.full_reads_ec_path, f"{self.genome_str}_ec.fa")
        #subprocess.run(f"{self.hifiasm_path}/hifiasm -o {ov_prefix} -r 1 -t{self.threads} --dbg-ovec {ec_reads}", shell=True, check=True)
        subprocess.run(f"{self.hifiasm_path}/hifiasm -o {ov_prefix} -r 3 -t{self.threads} --dbg-ovec {reads_file}", shell=True, check=True)
        # Merge cis and trans overlaps
        cis_paf = f"{ov_prefix}.0.ovlp.paf"
        trans_paf = f"{ov_prefix}.1.ovlp.paf"
        merged_paf = os.path.join(self.overlaps_path, f"{self.genome_str}_ov.paf")  
        cis_paf_path = os.path.join(self.overlaps_path, f"{self.genome_str}_cis.paf")
        with gzip.open(merged_paf + '.gz', 'wt') as outfile:
            for paf_file in [cis_paf, trans_paf]:
                if os.path.exists(paf_file):
                    with open(paf_file, 'r') as infile:
                        outfile.write(infile.read())
        print(f"Merged overlaps saved to: {merged_paf}")
        shutil.move(cis_paf, cis_paf_path)
        
        # Step 3: Run RAFT to fragment the error-corrected reads
        if self.real:
            frag_prefix = os.path.join(self.raft_reads_path, f"{self.genome_str}")
        else:
            frag_prefix = os.path.join(self.raft_reads_path, f"{self.genome_str}_frag")
        ov_file = merged_paf + '.gz'
        subprocess.run(f"{self.raft_path}/raft -e {raft_depth} -o {frag_prefix} -l 50000 {reads_file} {cis_paf_path}", shell=True, check=True)
        shutil.move(frag_prefix + ".coverage.txt", frag_prefix + '_cis.coverage.txt')
        subprocess.run(f"{self.raft_path}/raft -e {raft_depth} -o {frag_prefix} {reads_file} {ov_file}", shell=True, check=True)        

        # Gzip the fragmented reads file
        subprocess.run(f"gzip -f {frag_prefix}.reads.fasta", shell=True, check=True)
        #-m 2.3 

    def raft_process_fasta_files(self):

        sequence_file = os.path.join(self.raft_reads_path, f"{self.genome_str}_frag.reads.fasta.gz")            
        description_file = os.path.join(self.read_descr_path, f'{self.genome_str}.fasta')
        output_file = os.path.join(self.raft_reads_path, f'{self.genome_str}.reads.fasta.gz')

        sequences = self.load_fasta(sequence_file)
        descriptions = self.load_fasta(description_file)
        
        description_dict = {}
        for d in descriptions:
            parsed = self.parse_description_header(d.description)
            if parsed:
                description_dict[parsed["id"]] = parsed
            else:
                print(f"Skipping description: {d.id} {d.description}", file=sys.stderr)
        
        with gzip.open(output_file, 'wt') as out_f:
            for i, seq in enumerate(sequences, start=1):
                read_id, pos_start, pos_end = self.parse_sequence_header(seq.description)
                #print(read_id, pos_start, pos_end, read_id in description_dict)
                #print(i)
                if read_id and read_id in description_dict:
                    desc = description_dict[read_id]
                    relative_start = desc["start"] + pos_start
                    relative_end = desc["start"] + pos_end
                    new_header = f">{i} strand={desc['strand']} start={relative_start} end={relative_end} variant={desc['variant']} chr={desc['chr']}"
                    out_f.write(f"{new_header}\n{seq.seq}\n")
                else:
                    print(f"No matching description found for sequence: {seq.id} {seq.description}", file=sys.stderr)
                    exit()
        #shutil.move(os.path.join(self.raft_reads_path, f'{self.genome_str}.coverage.txt'), os.path.join(self.pile_o_grams_path, f'{self.genome_str}.txt'))


    def create_graphs(self):
        if self.raft:
            full_fasta = os.path.join(self.raft_reads_path, f'{self.genome_str}.reads.fasta.gz')
        elif self.real:
            full_fasta = os.path.join(self.full_reads_path, f'{self.genome_str}.fastq.gz')
        else:
            full_fasta = os.path.join(self.full_reads_path, f'{self.genome_str}.fasta.gz')

        gfa_output = os.path.join(self.gfa_graphs_path, f'{self.genome_str}.gfa')
        

        if self.diploid:
            hifiasm_asm_output_1 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap1.gfa')
            hifiasm_asm_output_2 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap2.gfa')
            #subprocess.run(f'./hifiasm --prt-raw --write-paf -r1 -o {self.hifiasm_dump}/tmp_asm -t{self.threads} -1 {self.paternal_yak} -2 {self.maternal_yak} {full_fasta}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'./hifiasm --prt-raw --write-paf -r3 -o {self.hifiasm_dump}/tmp_asm -t{self.threads} {full_fasta}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.hap1.p_ctg.gfa {hifiasm_asm_output_1}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.hap2.p_ctg.gfa {hifiasm_asm_output_2}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.raw.r_utg.gfa {gfa_output}', shell=True, cwd=self.hifiasm_path)
        else:
            hifiasm_asm_output = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.gfa')
            subprocess.run(f'./hifiasm --prt-raw --write-paf -r3 -o {self.hifiasm_dump}/tmp_asm -t{self.threads} {full_fasta}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.p_ctg.noseq.gfa {hifiasm_asm_output}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.p_ctg.gfa {hifiasm_asm_output}', shell=True, cwd=self.hifiasm_path)
            subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.bp.raw.r_utg.gfa {gfa_output}', shell=True, cwd=self.hifiasm_path)
            # Move the PAF file to the hifiasm PAF directory
        # Move the PAF file to the hifiasm PAF directory
        paf_output = os.path.join(self.paf_path, f'{self.genome_str}.paf')
        subprocess.run(f'mv {self.hifiasm_dump}/tmp_asm.ovlp.paf {paf_output}', shell=True, cwd=self.hifiasm_path)
        self.save_hifiasm_assemblies()

        subprocess.run(f'rm {self.hifiasm_dump}/tmp_asm*', shell=True, cwd=self.hifiasm_path)

    def save_hifiasm_assemblies(self):
        #trio_mode = self.diploid
        hic_mode = False
        if  self.diploid:
            self._convert_and_move_assembly('tmp_asm.dip.hap1.p_ctg.gfa')
            self._convert_and_move_assembly('tmp_asm.dip.hap2.p_ctg.gfa')
        elif hic_mode:
            self._convert_and_move_assembly('tmp_asm.hic.hap1.p_ctg.gfa')
            self._convert_and_move_assembly('tmp_asm.hic.hap2.p_ctg.gfa')
        else:
            self._convert_and_move_assembly('tmp_asm.bp.p_ctg.gfa')

    def _convert_and_move_assembly(self, gfa_filename):
        gfa_path = os.path.join(self.hifiasm_dump, gfa_filename)
        fa_filename = gfa_filename.replace('.gfa', '.fa')
        fa_path = os.path.join(self.hifiasm_asm_path, fa_filename)
        
        if os.path.exists(gfa_path):
            # Convert GFA to FASTA
            awk_command = "awk '/^S/{print \">\"$2;print $3}'"
            subprocess.run(f"{awk_command} {gfa_path} > {fa_path}", shell=True, check=True)
            print(f"Converted and moved {gfa_filename} to {fa_filename}")
        else:
            print(f"Warning: Expected assembly file {gfa_filename} not found.")

    def add_hifiasm_final_edges(self, gfa_path):
        print(f"Loading HiFiasms final gfa...")
        with open(gfa_path) as f:
            rows = f.readlines()
            c2r = defaultdict(list)
            for row in rows:
                row = row.strip().split()
                if row[0] != "A": continue
                c2r[row[1]].append(row)

        print(f"Generating contigs...")
        edges = []
        prefixes = []
        orientations = []
        for c_id, reads in c2r.items():
            reads = sorted(reads, key=lambda x:int(x[2]))
            for i in range(len(reads)-1):
                curr_row, next_row = reads[i], reads[i+1]
                curr_prefix = int(next_row[2])-int(curr_row[2])
                edges.append((str(curr_row[4]), str(next_row[4])))
                orientations.append((0 if curr_row[3] == '+' else 1, 0 if next_row[3] == '+' else 1))
                #print(curr_row[4])
                prefixes.append(curr_prefix)
        return edges, prefixes, orientations


    def parse_gfa(self):
        nx_graph, read_seqs, node_to_read, read_to_node, successor_dict = self.only_from_gfa()
        self.create_reads_fasta(read_seqs, self.chr_id)  # Add self.chr_id as an argument
        # Save data
        self.pickle_save(node_to_read, self.node_to_read_path)
        self.pickle_save(successor_dict, self.successor_dict_path)
        if self.prep_decoding:
            self.pickle_save(read_seqs, self.reduced_reads_path)
        self.pickle_save(read_to_node, self.read_to_node_path)
        
        """with open(os.path.join(self.node_to_read_path, f'{self.genome_str}.pkl'), 'rb') as f:
            node_to_read = pickle.load(f)
        with open(os.path.join(self.successor_dict_path, f'{self.genome_str}.pkl'), 'rb') as f:
            successor_dict = pickle.load(f)
        with open(os.path.join(self.reduced_reads_path, f'{self.genome_str}.pkl'), 'rb') as f:
            read_seqs = pickle.load(f)
        with open(os.path.join(self.read_to_node_path, f'{self.genome_str}.pkl'), 'rb') as f:
            read_to_node = pickle.load(f)"""

        # Load pickled FASTA file from full_reads
        if self.prep_decoding:
            fasta_pickle_path = os.path.join(self.full_reads_path, f'{self.genome_str}.pkl')
            with open(fasta_pickle_path, 'rb') as f:
                annotated_fasta_data = pickle.load(f)
                    # Initialize reads_parsed set
            paf_path = os.path.join(self.paf_path, f'{self.genome_str}.paf')
            paf_out = os.path.join(self.paf_path, f"{self.genome_str}.pkl")
            aux = {
                'read_seqs': read_seqs,
                'read_to_node': read_to_node,
                'annotated_fasta_data': annotated_fasta_data,
                'successor_dict': successor_dict,
                'node_to_read': node_to_read
                }
            self.parse_paf(aux, paf_path, paf_out)

        return nx_graph  

    def only_from_gfa(self):
        self.assembler = "hifiasm"
        training = not self.real
        gfa_path = os.path.join(self.gfa_graphs_path, f'{self.genome_str}.gfa')
        if self.raft:
            reads_path = os.path.join(self.raft_reads_path, f'{self.genome_str}.reads.fasta.gz')
        else:
            #reads_path = os.path.join(self.full_reads_ec_path, f'{self.genome_str}_ec.fa')
            if self.real:
                reads_path = os.path.join(self.full_reads_path, f'{self.genome_str}.fastq.gz')
            else:
                reads_path = os.path.join(self.full_reads_path, f'{self.genome_str}.fasta.gz')

        read_headers = self.get_read_headers(reads_path)

        graph_nx = nx.DiGraph()
        read_to_node, node_to_read, old_read_to_utg = {}, {}, {}  ##########
        read_to_node2 = {}
        edges_dict = {}
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
                        # The issue here is that in some cases, one unitig can consist of more than one read
                        # So this is the adapted version of the code that supports that
                        # The only things of importance here are read_to_node2 dict (not overly used)
                        # And id variable which I use for obtaining positions during training (for the labels)
                        # I don't use it for anything else, which is good
                        ids = []
                        while True:
                            line = all_lines[line_idx]
                            line = line.strip().split()
                            #print(line)
                            if line[0] != 'A':
                                break
                            line_idx += 1
                            tag = line[0]
                            utg_id = line[1]
                            read_orientation = line[3]
                            utg_to_read = line[4]
                            ids.append((utg_to_read, read_orientation))
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

        ### The following code is for adding the final assembly edges to the graph.
        ## In HiFiasm the final assembly has some edges that are NOT in the full graph, 
        # so we need to add them to make the graph complete.

        
        if self.diploid:
            hifiasm_asm_output_1 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap1.gfa')
            hifiasm_asm_output_2 = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.hap2.gfa')
            final_assembly_edges_1, prefixes_1, orientations_1 = self.add_hifiasm_final_edges(hifiasm_asm_output_1)
            final_assembly_edges_2, prefixes_2, orientations_2 = self.add_hifiasm_final_edges(hifiasm_asm_output_2)
            #exit()
            final_assembly_edges = final_assembly_edges_1 + final_assembly_edges_2
            prefixes = prefixes_1 + prefixes_2
            orientations = orientations_1 + orientations_2
        else:
            hifiasm_asm_output = os.path.join(self.hifiasm_asm_path, f'{self.genome_str}.gfa')
            final_assembly_edges, prefixes, orientations = self.add_hifiasm_final_edges(hifiasm_asm_output)

        nx_edges = set(graph_nx.edges())
        nx_nodes = set(graph_nx.nodes())
        just_added_edges = set()
        new_nodes_to_read = {}
        new_edge = {edge: 0 for edge in graph_nx.edges()}
        new_node = {node: 0 for node in graph_nx.nodes()}
        fastaq_seqs = self.load_fasta(reads_path, dict=True)
        # Initialize counters
        print(f"Processing {len(final_assembly_edges)} edges")
        total_edges = len(final_assembly_edges)
        skipped_missing_read_ids = 0
        skipped_missing_nodes = 0 
        skipped_self_loops = 0
        added_forward_edges = 0
        existing_forward_edges = 0
        double_new_edges = 0
        for i, edge in enumerate(final_assembly_edges):
            # Convert read IDs to node IDs using read_to_node
            ori = orientations[i]

            if edge[0] not in read_to_node2:
                # Add node and sequence for the first read
                #continue
                if ori[0] == '+':
                    real_idx, virt_idx = node_idx, node_idx + 1 
                else:
                    real_idx, virt_idx = node_idx + 1, node_idx
                read_to_node2[edge[0]] = (real_idx, virt_idx)
                for idx in (real_idx, virt_idx):
                    nx_nodes.add(idx)
                    graph_nx.add_node(idx)
                    node_to_read[idx] = edge[0]
                    new_nodes_to_read[idx] = edge[0]
                    new_node[idx] = 1
                    read_lengths[idx] = len(fastaq_seqs[edge[0]])
                node_idx += 2
                if not self.real:
                    #### you have given here the read_ids (small read_ids) 
                    description = read_headers[edge[0]]
                    # desc_id, strand, start, end = description.split()
                    strand = re.findall(r'strand=(\+|\-)', description)[0]
                    strand = 1 if strand == '+' else -1
                    #print(f"strand: {strand}", orientations[i])
                    start = int(re.findall(r'start=(\d+)', description)[0])  # untrimmed
                    end = int(re.findall(r'end=(\d+)', description)[0])  # untrimmed
                    #chromosome = int(re.findall(r'chr=(\d+)', description)[0])
                    chromosome = re.findall(r'chr=([^\s]+)', description)[0]
                    # this should be the ones in the description header info!! So I am very close
                    # just use the descr header info and fill the dicts here???
                    if self.diploid:
                        variant = re.findall(r'variant=([P|M])', description)[0]
                        read_variants[real_idx] = read_variants[virt_idx] = variant
                    read_strands[real_idx], read_strands[virt_idx] = strand, -strand
                    read_starts[real_idx] = read_starts[virt_idx] = start
                    read_ends[real_idx] = read_ends[virt_idx] = end
                    read_chrs[real_idx] = read_chrs[virt_idx] = chromosome
                    #read_lengths[real_idx] = read_lengths[virt_idx] = abs(end-start)
                    #skipped_missing_read_ids += 1
                    #continue

            if edge[1] not in read_to_node2:
                # Add node and sequence for the second read
                #continue
                if ori[1] == '+':
                    real_idx, virt_idx = node_idx, node_idx + 1 
                else:
                    real_idx, virt_idx = node_idx + 1, node_idx

                read_to_node2[edge[1]] = (real_idx, virt_idx)
                for idx in (real_idx, virt_idx):
                    nx_nodes.add(idx)
                    graph_nx.add_node(idx)
                    node_to_read[idx] = edge[1]
                    new_nodes_to_read[idx] = edge[1]
                    new_node[idx] = 1
                    read_lengths[idx] = len(fastaq_seqs[edge[1]])
                node_idx += 2
                if not self.real:
                    #### you have given here the read_ids (small read_ids) 
                    description = read_headers[edge[1]]
                    # desc_id, strand, start, end = description.split()
                    strand = re.findall(r'strand=(\+|\-)', description)[0]
                    strand = 1 if strand == '+' else -1
                    start = int(re.findall(r'start=(\d+)', description)[0])  # untrimmed
                    end = int(re.findall(r'end=(\d+)', description)[0])  # untrimmed
                    #chromosome = int(re.findall(r'chr=(\d+)', description)[0])
                    chromosome = re.findall(r'chr=([^\s]+)', description)[0]
                    # this should be the ones in the description header info!! So I am very close
                    # just use the descr header info and fill the dicts here???
                    if self.diploid:
                        variant = re.findall(r'variant=([P|M])', description)[0]
                        read_variants[real_idx] = read_variants[virt_idx] = variant
                    read_strands[real_idx], read_strands[virt_idx] = strand, -strand
                    read_starts[real_idx] = read_starts[virt_idx] = start
                    read_ends[real_idx] = read_ends[virt_idx] = end
                    read_chrs[real_idx] = read_chrs[virt_idx] = chromosome
                    #read_lengths[real_idx] = read_lengths[virt_idx] = abs(end-start)
                #skipped_missing_read_ids += 1
                #continue
                
            # Get forward node IDs
            src_node = read_to_node2[edge[0]][ori[0]]  # Forward or Backward node ID
            dst_node = read_to_node2[edge[1]][ori[1]]  # Forward or Backward node ID
                
            if src_node == dst_node:
                skipped_self_loops += 1
                continue

            if (src_node, dst_node) in just_added_edges:
                double_new_edges += 1
                continue
                
            # Add forward edge (e1,e2)
            if (src_node, dst_node) not in nx_edges:
                #print(f"Adding forward edge ({src_node}, {dst_node})")
                graph_nx.add_edge(src_node, dst_node)
                new_edge[(src_node, dst_node)] = 1
                just_added_edges.add((src_node, dst_node))
                prefix_lengths[(src_node, dst_node)] = prefixes[i]
                ol_length = read_lengths[src_node] - prefixes[i]

                overlap_lengths[(src_node, dst_node)] = ol_length
                added_forward_edges += 1
                edge_ids[(src_node, dst_node)] = edge_idx
                edge_idx += 1
            
            # Add reverse complement edge (e2^1, e1^1)
            rc_src = dst_node ^ 1  # e2^1
            rc_dst = src_node ^ 1  # e1^1
            
            if (rc_src, rc_dst) not in nx_edges:
                #print(f"Adding RC edge ({rc_src}, {rc_dst})")
                graph_nx.add_edge(rc_src, rc_dst)
                new_edge[(rc_src, rc_dst)] = 1
                just_added_edges.add((rc_src, rc_dst))
                prefix_lengths[(rc_src, rc_dst)] = prefixes[i]
                ol_length = read_lengths[rc_src] - prefixes[i]
                overlap_lengths[(rc_src, rc_dst)] = ol_length
                edge_ids[(rc_src, rc_dst)] = edge_idx
                edge_idx += 1

        print(f"\nEdge Processing Summary:")
        print(f"Total edges checked: {total_edges}")
        print(f"Skipped edges:")
        print(f"  - Missing read IDs: {skipped_missing_read_ids}")
        print(f"  - Self loops: {skipped_self_loops}")
        print(f"Forward edges:")
        print(f"  - Added: {added_forward_edges}")
        print(f"  - Already existed: {existing_forward_edges}")
        print(f"  - Double new edges: {double_new_edges}")
        elapsed = (datetime.now() - time_start).seconds
        print(f'Elapsed time: {elapsed}s')
        
        if no_seqs_flag:
            fastaq_seqs = self.load_fasta(reads_path, dict=True)

            print(f'Sequences successfully loaded!')
            # fastaq_seqs = {read.id: read.seq for read in SeqIO.parse(reads_path, filetype)}
            for node_id in tqdm(node_to_read.keys(), ncols=120):
                    read_id = node_to_read[node_id]
                    seq = fastaq_seqs[read_id]
                    read_seqs[node_id] = str(seq if node_id % 2 == 0 else seq.reverse_complement())
                    
            print(f'Loaded DNA sequences!')
        else:
            fastaq_seqs = self.load_fasta(reads_path, dict=True)
            print(f'Sequences successfully loaded!')
            #print(new_nodes_to_read.keys())
            for node_id in tqdm(new_nodes_to_read.keys(), ncols=120):
                read_id = new_nodes_to_read[node_id]
                seq = fastaq_seqs[read_id]
                read_seqs[node_id] = str(seq if node_id % 2 == 0 else seq.reverse_complement())
            print(f'Added new DNA sequences!')
        
        elapsed = (datetime.now() - time_start).seconds
        print(f'Elapsed time: {elapsed}s')

        print(f'Calculating similarities...')
        overlap_similarities = self.calculate_similarities(edge_ids, read_seqs, overlap_lengths)
        print(f'Done!')
        elapsed = (datetime.now() - time_start).seconds
        print(f'Elapsed time: {elapsed}s')

        nx.set_node_attributes(graph_nx, read_lengths, 'read_length')
        nx.set_node_attributes(graph_nx, variant_class, 'variant_class')
        node_attrs = ['read_length', 'variant_class']

        nx.set_edge_attributes(graph_nx, prefix_lengths, 'prefix_length')
        nx.set_edge_attributes(graph_nx, overlap_lengths, 'overlap_length')
        nx.set_edge_attributes(graph_nx, new_edge, 'new_edge')
        edge_attrs = ['prefix_length', 'overlap_length', 'new_edge']

        if training:
            nx.set_node_attributes(graph_nx, read_strands, 'read_strand')
            nx.set_node_attributes(graph_nx, read_starts, 'read_start')
            nx.set_node_attributes(graph_nx, read_ends, 'read_end')
            nx.set_node_attributes(graph_nx, read_variants, 'read_variant')
            nx.set_node_attributes(graph_nx, read_chrs, 'read_chr')
            node_attrs.extend(['read_strand', 'read_start', 'read_end', 'read_variant', 'read_chr'])

        nx.set_edge_attributes(graph_nx, overlap_similarities, 'overlap_similarity')
        edge_attrs.append('overlap_similarity')

        # Create a dictionary of nodes and their direct successors
        successor_dict = {node: list(graph_nx.successors(node)) for node in graph_nx.nodes()}

        # Why is this the case? Is it because if there is even a single 'A' file in the .gfa, means the format is all 'S' to 'A' lines?
        if len(read_to_node2) != 0:
            read_to_node = read_to_node2

        # Print number of nodes and edges in graph
    
        return graph_nx, read_seqs, node_to_read, read_to_node, successor_dict

    def create_pog_features(self, nx_graph):
        """
        Create pog_median, pog_min, and pog_max features for each node and edge based on pile o gram data
        from both full and cis coverage files
        """
        # Load the pile o gram files
        if self.real:
            full_pog_file = os.path.join(self.raft_reads_path, f'{self.genome_str}.coverage.txt')
            cis_pog_file = os.path.join(self.raft_reads_path, f'{self.genome_str}_cis.coverage.txt')
        else:
            full_pog_file = os.path.join(self.raft_reads_path, f'{self.genome_str}_frag.coverage.txt')
            cis_pog_file = os.path.join(self.raft_reads_path, f'{self.genome_str}_frag_cis.coverage.txt')
        
        # Load the read_to_node_id mapping
        read_to_node_path = os.path.join(self.read_to_node_path, f'{self.genome_str}.pkl')
        with open(read_to_node_path, 'rb') as f:
            read_to_node = pickle.load(f)
        node_to_read_path = os.path.join(self.node_to_read_path, f'{self.genome_str}.pkl')
        with open(node_to_read_path, 'rb') as f:
            node_to_read = pickle.load(f)

        # Initialize all nodes with default values of 1
        pog_median = {node: 1 for node in nx_graph.nodes()}
        pog_min = {node: 1 for node in nx_graph.nodes()}
        pog_max = {node: 1 for node in nx_graph.nodes()}
        cis_pog_median = {node: 1 for node in nx_graph.nodes()}
        cis_pog_min = {node: 1 for node in nx_graph.nodes()}
        cis_pog_max = {node: 1 for node in nx_graph.nodes()}

        # Store coverage data for each read
        read_coverages = {}
        cis_read_coverages = {}

        # Process full coverage file
        with open(full_pog_file, 'r') as f:
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
        # Process cis coverage file 
        with open(cis_pog_file, 'r') as f:
            for line in f:
                if line.startswith('read'):
                    parts = line.strip().split()
                    read_id = parts[1]
                    coverages = [int(part.split(',')[1]) for part in parts[2:] if ',' in part]
                    cis_read_coverages[read_id] = coverages
                    if read_id in read_to_node:
                        node_ids = read_to_node[read_id]
                        median = np.median(coverages)
                        min_val = np.min(coverages)
                        max_val = np.max(coverages)
                        for node_id in node_ids:
                            cis_pog_median[node_id] = median / self.depth
                            cis_pog_min[node_id] = min_val / self.depth
                            cis_pog_max[node_id] = max_val / self.depth

        # Set the pog attributes for each node in the graph
        nx.set_node_attributes(nx_graph, cis_pog_median, 'cis_pog_median')
        nx.set_node_attributes(nx_graph, cis_pog_min, 'cis_pog_min')
        nx.set_node_attributes(nx_graph, cis_pog_max, 'cis_pog_max')
        print(f"Added pile-o-gram features to {len(pog_median)} nodes.")

    def create_hh_features(self, nx_graph):
        """
        load reads. check starting position of read and compare with variant_starts list.
        once you reach the position on variant_start list that is higher then the read_start, you go one entry back.
        now check if the variant_ends entry with the same index is larger then the read start. If yes: there is a variation
        now check the read end position: it's the same but swapping ends and starts.
        """
        M_path = os.path.join(self.vcf_path, f'{self.chrN}_M.vcf.gz')
        P_path = os.path.join(self.vcf_path, f'{self.chrN}_P.vcf.gz')

        read_start = nx.get_node_attributes(nx_graph, 'read_start')
        read_end = nx.get_node_attributes(nx_graph, 'read_end')
        read_variant = nx.get_node_attributes(nx_graph, 'read_variant')

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

        hh_dict = {}
        print(f'Process Maternal reference variation file')
        M_variant_starts, M_variant_ends = self.process_vcf_file(M_path)
        print(f'Process Paternal reference variation file')
        P_variant_starts, P_variant_ends = self.process_vcf_file(P_path)
        print(f'Process graph nodes')
        hetero_nodes = 0

        for node in nx_graph.nodes():
            hh_dict[node] = "O"  # Homozygous Region
            if read_variant[node] == 'M':
                var_starts = M_variant_starts
                var_ends = M_variant_ends
            elif read_variant[node] == 'P':
                var_starts = P_variant_starts
                var_ends = P_variant_ends
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



def main():
    parser = argparse.ArgumentParser(description="Generate dataset based on configuration")
    parser.add_argument('--ref', type=str, default='/mnt/sod2-project/csb4/wgs/martin/genome_references', help='Path to references root dir')
    parser.add_argument('--data_path', type=str, default='/mnt/sod2-project/csb4/wgs/martin/diploid_datasets/hifiasm_dataset', help='Path to dataset folder')   
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()

    dataset_path = args.data_path
    ref_base_path = args.ref
    # Read the configuration file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    full_dataset = create_full_dataset_dict(config)
    # Initialize the appropriate dataset creator
    if config['gen_config']['assembler'] == 'raven':
        dataset_object = RavenDatasetCreator(args.ref, args.data_path, args.config)
        stop_after_reads = True
    elif config['gen_config']['assembler'] == 'hifiasm':
        dataset_object = HifiasmDatasetCreator(args.ref, args.data_path, args.config)
        stop_after_reads = False
    else:
        raise ValueError(f"Unknown assembler: {config['gen_config']['assembler']}")

    # Process each chromosome
    for chrN, amount in full_dataset.items():
        for i in range(amount):
            gen_steps(dataset_object, chrN, i, config['gen_steps'], ref_base_path, stop_after_reads)

def gen_steps(dataset_object, chrN_, i, gen_step_config, ref_base_path, stop_after_reads=False):

    split_chrN = chrN_.split(".")
    chrN = split_chrN[1]
    genome = split_chrN[0]
    chr_id = f'{chrN}_{i}'
    ref_base = (f'{ref_base_path}/{genome}')
        
    #if i<110:
    #   return

    dataset_object.load_chromosome(genome, chr_id, ref_base)
    print(f'Processing {dataset_object.genome_str}...')

    if gen_step_config['sample_reads']:
        dataset_object.simulate_reads()
        if stop_after_reads:
            return

    if gen_step_config['create_graphs']:
        dataset_object.create_graphs()
        print(f"Created gfa graph {chrN}_{i}")

    if gen_step_config['parse_gfa']:
        nx_graph = dataset_object.parse_gfa()
        dataset_object.pickle_save(nx_graph, dataset_object.nx_graphs_path)
        print(f"Saved nx graph {chrN}_{i}")

    elif gen_step_config['ground_truth'] or gen_step_config['diploid_features'] or gen_step_config['ml_graphs'] or gen_step_config['pile-o-gram']:
        nx_graph = dataset_object.load_nx_graph()
        print(f"Loaded nx graph {chrN}_{i}")

    if dataset_object.diploid and gen_step_config['create_yak']:# and dataset_object.real:
        dataset_object.gen_yak_files()
        print(f"Done with yak files {chrN}_{i}")

    if 'pile-o-gram' in gen_step_config:
        if gen_step_config['pile-o-gram']:
            #dataset_object.add_pos_features(nx_graph)
            #print(f"Done with pile o gram {chrN}_{i}")
            #dataset_object.create_pog_features(nx_graph)
            print(f"Done with pog features {chrN}_{i}")
            #dataset_object.analyze_debug(nx_graph)
            #dataset_object.create_random_walk_features(nx_graph)
            #dataset_object.add_min_cycle_edges_length(nx_graph)
            if 'comp_dist' not in nx_graph.nodes[list(nx_graph.nodes())[0]]:
                dataset_object.add_comp_dist(nx_graph)
            #dataset_object.add_comp_dist_multi_process(nx_graph)
            #exit()
            dataset_object.pickle_save(nx_graph, dataset_object.nx_graphs_path)

    if dataset_object.diploid and gen_step_config['diploid_features']:
        #if not dataset_object.real:
        #    dataset_object.create_hh_features(nx_graph)
        #    print(f"Done with hh features {chrN}_{i}")
        dataset_object.add_trio_binning_labels(nx_graph)
        print(f"Done with trio binning {chrN}_{i}")
        dataset_object.pickle_save(nx_graph, dataset_object.nx_graphs_path)

    if not dataset_object.real and gen_step_config['ground_truth']:
        #dataset_object.get_telomere_ftrs(nx_graph)
        dataset_object.create_gt(nx_graph)
        print(f"Done with ground truth creation {chrN}_{i}")
        #dataset_object.add_decision_attr(nx_graph)
        #print(f"Done with decision node creation {chrN}_{i}")
        dataset_object.pickle_save(nx_graph, dataset_object.nx_graphs_path)

    if gen_step_config['ml_graphs']:
        dataset_object.save_to_dgl_and_pyg(nx_graph)
        print(f"Saved DGL and PYG graphs of {chrN}_{i}")

    print("Done for one chromosome!")

if __name__ == "__main__":
    main()