import pickle
from Bio import SeqIO
import gzip
import os
import subprocess
import glob
# Create a SeqRecord with bin information in the ID
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
    
# Import tqdm for progress bar
from tqdm import tqdm
import re

def _get_filetype(file_path):
    """Determine if file is FASTA or FASTQ format based on extension."""
    if file_path.endswith(('.gz', '')):
        base_path = file_path[:-3] if file_path.endswith('.gz') else file_path
        if base_path.endswith(('fasta', 'fna', 'fa')):
            return 'fasta'
        elif base_path.endswith(('fastq', 'fnq', 'fq')):
            return 'fastq'
    return 'fasta'  # Default to fasta if unknown

def load_fasta(file_path, dict=False):
    """Load sequences from a FASTA/FASTQ file, handling both compressed and uncompressed files.
    Returns a list of SeqRecord objects or dict mapping ids to sequences if dict=True."""
    filetype = _get_filetype(file_path)
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

def save_fastas(chr_labels, n2s_path, binned_dir):
    """Bin reads based on chromosome labels and save as separate FASTA files."""
    if n2s_path is None:
        print("No n2s_path path provided, skipping inference")
        return None
    
    print(f"Loading reads from {n2s_path}")
    reads = load_fasta(n2s_path)
    print(f"Loaded {len(reads)} reads")
    # Convert reads list to dictionary for faster lookup
    reads_dict = {record.id: record for record in reads}
    
    # Create a dictionary to store reads for each bin
    binned_reads = {}

    # Process each node and assign its reads to the appropriate bin
    for node_id, bin_label in tqdm(chr_labels.items(), total=len(chr_labels), desc="Binning reads"):
        # Get both the node ID and its complement (n^1)
        node_ids_to_process = [int(node_id)]
        complement_node_id = int(node_id) ^ 1  # XOR with 1 to get complement node
        node_ids_to_process.append(complement_node_id)
        
        for node_id_to_process in node_ids_to_process:
            # Skip if no read for this node
            if str(node_id_to_process) not in reads_dict:
                continue
                
            # Handle bin_label as a list - add the read to each bin in the list
            if isinstance(bin_label, list):
                for single_bin in bin_label:
                    bin_key = str(single_bin)
                    if bin_key not in binned_reads:
                        binned_reads[bin_key] = []
                    binned_reads[bin_key].append(reads_dict[str(node_id_to_process)])
            else:
                # Handle single bin label
                bin_key = str(bin_label)
                if bin_key not in binned_reads:
                    binned_reads[bin_key] = []
                binned_reads[bin_key].append(reads_dict[str(node_id_to_process)])
    
    # Create output directory for binned fastas if it doesn't exist
    os.makedirs(binned_dir, exist_ok=True)
    
    # Write each bin to a separate FASTA file with progress bar
    for bin_label, bin_reads in tqdm(binned_reads.items(), total=len(binned_reads), desc="Writing FASTA files"):
        output_file = os.path.join(binned_dir, f"bin_{bin_label}.fasta")
        SeqIO.write(bin_reads, output_file, "fasta")
        print(f"Wrote {len(bin_reads)} reads to {output_file}")
    

def run_hifiasm(binned_dir, output_dir, hifiasm_path, merged_fasta):
    """Run hifiasm on each binned FASTA file and merge results."""
    # Create output directories
    asm_dir = os.path.join(output_dir, "hifiasm_results")
    os.makedirs(asm_dir, exist_ok=True)
    
    # Get all FASTA files in the binned directory
    fasta_files = glob.glob(os.path.join(binned_dir, "*.fasta"))

    if not fasta_files:
        print("No FASTA files found in the binned directory")
        return
    
    # Process each FASTA file with hifiasm
    all_gfa_outputs = []
    for fasta_input in fasta_files:
        os.makedirs(os.path.join(asm_dir, "tmp_asms"), exist_ok=True)
        bin_name = os.path.basename(fasta_input).replace(".fasta", "")
        print(f"Running hifiasm on {bin_name}...")
        
        # Run hifiasm
        subprocess.run(f'./hifiasm -l0 -r1 -o {asm_dir}/tmp_asms/tmp_asm {fasta_input}', 
                      shell=True, cwd=hifiasm_path)
        
        # Move the output file to results directory
        gfa_output = os.path.join(asm_dir, f"{bin_name}.gfa")

        subprocess.run(f'mv {asm_dir}/tmp_asms/tmp_asm.bp.p_ctg.gfa {gfa_output}', 
                      shell=True, cwd=hifiasm_path)
        subprocess.run(f'rm -rf {asm_dir}/tmp_asms', shell=True, cwd=hifiasm_path)
        
        all_gfa_outputs.append(gfa_output)
        print(f"Completed assembly for {bin_name}")
    
    # Merge results into a single FASTA file
    merge_gfa_to_fasta(all_gfa_outputs, merged_fasta)
    
    print(f"All assemblies completed and merged into {merged_fasta}")


def merge_gfa_to_fasta(gfa_files, output_fasta):
    """Merge multiple GFA files into a single FASTA file."""
    all_sequences = []
    
    for gfa_file in gfa_files:
        bin_name = os.path.basename(gfa_file).replace(".gfa", "")
        
        # Parse GFA file to extract sequences
        with open(gfa_file, 'r') as f:
            for line in f:
                if line.startswith('S'):  # Sequence line in GFA format
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        seq_id = parts[1]
                        sequence = parts[2]

                        record = SeqRecord(
                            Seq(sequence),
                            id=f"{bin_name}_{seq_id}",
                            description=""
                        )
                        all_sequences.append(record)
    
    # Write all sequences to a single FASTA file
    SeqIO.write(all_sequences, output_fasta, "fasta")
    print(f"Merged {len(all_sequences)} sequences into {output_fasta}")


def parse_yak_result(yakres_path):
    """
    Yak triobinning result files have following info:
    C       F  seqName     type      startPos  endPos    count
    C       W  #switchErr  denominator  switchErrRate
    C       H  #hammingErr denominator  hammingErrRate
    C       N  #totPatKmer #totMatKmer  errRate
    """
    switch_err = None
    hamming_err = None

    with open(yakres_path, 'r') as file:
        # Read all the lines and reverse them
        lines = file.readlines()
        reversed_lines = reversed(lines)

        for line in reversed_lines:
            if line.startswith('W'):
                switch_err = float(line.split()[3])
            elif line.startswith('H'):
                hamming_err = float(line.split()[3])

            if switch_err is not None and hamming_err is not None:
                break

    return switch_err, hamming_err

def parse_minigraph_result(stat_path):
    nga50 = 0
    ng50 = 0
    length = 0
    rdup = 0
    with open(stat_path) as f:
        for line in f.readlines():
            if line.startswith('NG50'):
                try:
                    ng50 = int(re.findall(r'NG50\s*(\d+)', line)[0])
                except IndexError:
                    ng50 = 0
            if line.startswith('NGA50'):
                try:
                    nga50 = int(re.findall(r'NGA50\s*(\d+)', line)[0])
                except IndexError:
                    nga50 = 0
            if line.startswith('Length'):
                try:
                    length = int(re.findall(r'Length\s*(\d+)', line)[0])
                except IndexError:
                    length = 0
            if line.startswith('Rdup'):
                try:
                    rdup = float(re.findall(r'Rdup\s*(\d+\.\d+)', line)[0])
                except IndexError:
                    rdup = 0

    return ng50, nga50, length, rdup

def eval_fasta(asm_path, ref, out_dir, mat_yak, pat_yak, threads = 32, yak_path="/home/schmitzmf/yak/yak", minigraph_path="/home/schmitzmf/minigraph/minigraph"):
    """Evaluate assembly against reference and parental haplotypes."""
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup evaluation paths    
    report_path = os.path.join(out_dir, "minigraph.txt")
    paf_path = os.path.join(out_dir, "asm.paf")
    phs_path = os.path.join(out_dir, "phs.txt")
    idx = ref + '.fai'
    
    # Run minigraph for assembly evaluation
    print(f"Running minigraph alignment...")
    if minigraph_path:
        cmd = f'{minigraph_path} -t32 -xasm -g10k -r10k --show-unmap=yes {ref} {asm_path}'.split(' ')
    else:
        cmd = f'minigraph -t32 -xasm -g10k -r10k --show-unmap=yes {ref} {asm_path}'.split(' ')
    
    with open(paf_path, 'w') as f:
        subprocess.run(cmd, stdout=f, check=True)
    
    # Parse PAF file
    print(f"Parsing alignment results...")
    #cmd = f'k8 paftools.js asmstat {idx} {paf_path}'.split()
    cmd = f'paftools.js asmstat {idx} {paf_path}'.split()

    with open(report_path, 'w') as f:
        subprocess.run(cmd, stdout=f, check=True)
    
    # Parse and print minigraph results
    with open(report_path) as f:
        report_content = f.read()
        print(report_content)
    
    # Run YAK evaluation if YAK indices are provided
    print(f"Running YAK evaluation...")
    cmd = f'{yak_path} trioeval -t{threads} {pat_yak} {mat_yak} {asm_path}'.split(' ')
    with open(phs_path, 'w') as f:
        subprocess.run(cmd, stdout=f, check=True)
            
    # Parse results
    ng50_m, nga50_m, length_m, rdup_m = parse_minigraph_result(report_path)
    switch_err_m, hamming_err_m = parse_yak_result(phs_path)
    
    print(f'Results:')
    print(f'Length: {"{:,}".format(length_m)}')
    print(f'Rdup: {rdup_m:.4f}')
    print(f'NG50: {"{:,}".format(ng50_m)}')
    print(f'NGA50: {"{:,}".format(nga50_m)}')
    print(f'YAK Switch Err: {switch_err_m * 100:.4f}%')
    print(f'YAK Hamming Err: {hamming_err_m * 100:.4f}%')

def main(chr_labels, n2s_path, output_dir, hifiasm_path="/home/schmitzmf/hifiasm_023", config=None, ref=None, mat_yak=None, pat_yak=None):
    """Main function to process reads, run hifiasm, and merge results."""
    # Save binned FASTA files
    print(f"Inference: Saving binned FASTA files...")
    binned_dir = os.path.join(output_dir, "binned_fastas")
    save_fastas(chr_labels, n2s_path, binned_dir)

    merged_fasta = os.path.join(output_dir, "merged_assembly.fasta")
    print(f"Inference: Running hifiasm on each bin and merging results...")
    run_hifiasm(binned_dir, output_dir, hifiasm_path, merged_fasta)

    if mat_yak and pat_yak and ref:
        eval_fasta(merged_fasta, ref, output_dir, mat_yak, pat_yak)
    else:
        print("No YAK indices or reference provided, skipping evaluation")
