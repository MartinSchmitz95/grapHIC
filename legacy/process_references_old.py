import argparse
import os
import subprocess
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from pyliftover import LiftOver
import yaml

class ProcessRefsObject:
    def __init__(self, ref_dir, ref_name, cen_file='/mnt/sod2-project/csb4/wgs/martin/genome_references/hg002_v101/hg002v1.0.1.cenSatv1.0.bed'):
        self.ref_dir = ref_dir
        self.ref_name = ref_name
        self.vcf_path = os.path.join(self.ref_dir, 'vcf')
        self.tmp_path = os.path.join(self.ref_dir, 'tmp')
        self.liftover_path = os.path.join(self.ref_dir, 'liftover')
        self.centromeres_path = os.path.join(self.ref_dir, 'centromeres')
        self.chromosomes_path = os.path.join(self.ref_dir, 'chromosomes')
        self.chain_path = os.path.join(self.ref_dir, 'chain')
        self.cent_entries = self.read_bed(cen_file)

        for path in [self.centromeres_path, self.vcf_path, self.tmp_path, self.chromosomes_path, self.liftover_path, self.chain_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    def read_bed(self, cen_file):
        # Initialize a list to hold the bed entries
        # Open the BED file and read its contents
        bed_dict = {}

        with open(cen_file, 'r') as file:
            for line in file:
                # Strip newline characters and split the line into columns
                columns = line.strip().split('\t')
                # Only add entries that have at least 3 columns
                if len(columns) >= 3:
                    #bed_entries.append(columns)
                    chrom = columns[0]
                    start = int(columns[1])
                    end = int(columns[2])
                    loc = columns[3]
                    if chrom not in bed_dict:
                        bed_dict[chrom] = []
                    bed_dict[chrom].append((start, end, loc))
        return bed_dict
    def split_refs(self, filter=True):
        """
        Split the reference into chromosomes.
        Traverses the reference, loaded with Seqio and saves each chromosome,
        where there exists a key X_MATERNAL and X_PATERNAL as separate file in a
        folder chromosomes, which is newly created in ref_path
        """
        fasta_file = os.path.join(self.ref_dir, f'{self.ref_name}.fasta')
        fasta_sequences = SeqIO.parse(open(fasta_file), 'fasta')
        contigs = {}
        for fasta in fasta_sequences:
            name, sequence = fasta.id, fasta.seq

            # Skip sequences with less than 1 million basepairs
            if len(sequence) < 1_000_000:
                print(f"{name} is less than 1 million basepairs, skipping")
                continue

            # filter sequences with N tokens
            if filter and 'N' in sequence:
                print(name, " contains N tokens, skipping")
                continue

            # Split the name and determine if it's maternal or paternal
            split_name = name.split('_')
            if len(split_name) > 1:
                chr_num = split_name[0].replace('chr', '')
                haplotype = split_name[1].upper()
                
                if haplotype in ['MATERNAL', 'MATERNAL', 'HAP1', 'MAT', 'M']:
                    new_name = f'chr{chr_num}_M'
                elif haplotype in ['PATERNAL', 'PATERNAL', 'HAP2', 'PAT', 'P']:
                    new_name = f'chr{chr_num}_P'
                else:
                    print(f"Unexpected haplotype for {name}, skipping")
                    continue
            else:
                print(f"Unexpected name format for {name}, skipping")
                continue

            # Check if the new_name already exists
            if new_name in contigs:
                # If it exists, keep the longer sequence
                print(f"Comparing {new_name} with {len(sequence)} and {len(contigs[new_name].seq)}")
                if len(sequence) > len(contigs[new_name].seq):
                    contigs[new_name] = fasta
                    print(f"Replaced {new_name} with longer sequence")
                else:
                    print(f"Kept existing {new_name} as it's longer than the new sequence")
            else:
                contigs[new_name] = fasta

        # Write the final contigs to files
        for new_name, fasta in contigs.items():
            output_fasta = os.path.join(self.chromosomes_path, f'{new_name}.fasta')
            with open(output_fasta, "w") as output_handle:
                fasta.id = new_name
                fasta.description = new_name
                SeqIO.write(fasta, output_handle, "fasta")
                print(f'wrote {new_name} to file {output_fasta}')

            # Index the new chromosome file with samtools
            subprocess.run(f"samtools faidx {output_fasta}", shell=True, check=True)
            print(f'Created index file for {new_name}: {output_fasta}.fai')

    def find_indices_of_N(self, s):
        # Find all indices of 'N' in the string
        indices = [i for i, char in enumerate(s) if char == 'N']
        return (indices[0], indices[-1]) if indices else (None, None)

    def create_acrocentric_refs(self):
        """
        For each chromosome in the reference, saves a sequence of the computed interval into a separate file.
        """
        fasta_file = os.path.join(self.ref_dir, f'{self.ref_name}.fasta')
        fasta_sequences = SeqIO.parse(open(fasta_file), 'fasta')
        print(fasta_file)
        for fasta in fasta_sequences:
            #print(fasta.id)

            name, sequence = fasta.id, str(fasta.seq)
            first_N, last_N = self.find_indices_of_N(sequence)


            if last_N is not None:
                print(f'{fasta.id} interval from: {last_N+1}')
                # Extract the interval sequence
                interval_sequence = sequence[last_N+1:]
                # Save the interval sequence to a new file
                with open(os.path.join(self.chromosomes_path, name + "_no_N.fasta"), "w") as output_handle:
                    # Create a SeqRecord to write using SeqIO for consistency
                    interval_seq_record = SeqIO.SeqRecord(Seq(interval_sequence), id=name,
                                                          description="no N Region")
                    SeqIO.write(interval_seq_record, output_handle, "fasta")
                    print(f'wrote {name}  file {os.path.join(self.chromosomes_path, name + "_no_N.fasta")}')
        print("Done with all acrocentrics.")


    def create_centromere_refs(self):
        """
        For each chromosome in the reference, saves a sequence of the computed interval into a separate file
        using centromere coordinates from a YAML file.
        """
        # Load centromere coordinates from YAML file
        with open('../data/centromere_coordinates_same_len.yml', 'r') as yaml_file:
            centromere_coords = yaml.safe_load(yaml_file)

        # Get the coordinates for the current genome
        genome_coords = centromere_coords.get(self.ref_name)
        if not genome_coords:
            print(f"No centromere coordinates found for {self.ref_name}")
            return

        for chr_name, coords in genome_coords.items():
            chr_file = f"{chr_name}.fasta"
            chr_path = os.path.join(self.chromosomes_path, chr_file)
            
            if not os.path.exists(chr_path):
                print(f"Warning: File not found for {chr_name}: {chr_path}")
                continue

            # Read the chromosome sequence
            try:
                with open(chr_path, 'r') as fasta_file:
                    records = list(SeqIO.parse(fasta_file, 'fasta'))
                    if not records:
                        print(f"Error: No sequences found in {chr_path}")
                        continue
                    if len(records) > 1:
                        print(f"Warning: Multiple sequences found in {chr_path}. Using the first one.")
                    record = records[0]
                    sequence = str(record.seq)
            except Exception as e:
                print(f"Error reading {chr_path}: {str(e)}")
                continue

            # Get centromere coordinates
            start = coords['c_start']
            end = coords['c_end']

            # Ensure coordinates are within sequence bounds
            start = max(0, start)
            end = min(len(sequence), end)

            print(f'{chr_name} interval: {start} - {end}')

            # Extract the centromeric sequence
            centromere_sequence = sequence[start:end]

            # Save the centromeric sequence to a new file
            output_path = os.path.join(self.centromeres_path, f"{chr_name}_c.fasta")
            with open(output_path, "w") as output_handle:
                centromere_record = SeqIO.SeqRecord(Seq(centromere_sequence), id=chr_name,
                                                    description="Centromeric Region")
                SeqIO.write(centromere_record, output_handle, "fasta")
                print(f'Wrote {chr_name} centromere to file {output_path}')

            # Create .fai file using samtools
            subprocess.run(f"samtools faidx {output_path}", shell=True, check=True)
            print(f'Created index file for {chr_name} centromere: {output_path}.fai')

        print("Extracted centromeric regions for all chromosomes.")


    def liftover_convert_coordinate(self, liftover_object, chr, position):
        new_pos = liftover_object.convert_coordinate(chr, position)
        if new_pos:
            return new_pos[0][1]
        else:
            return position
        
    def get_chr_names(self):
        chr_names = []
        for filename in os.listdir(self.chromosomes_path):
            if filename.endswith('_M.fasta'):
                chr_name = filename.split('_')[0]
                if os.path.exists(os.path.join(self.chromosomes_path, f'{chr_name}_P.fasta')):
                    chr_names.append(chr_name)
        return chr_names

    def create_vcf(self):
        # Get list of chromosomes with both maternal and paternal files
        chr_names = self.get_chr_names()
        for chr in chr_names:
            print(f"Active Chromosome: {chr} create vcf file")
            mat_ref = os.path.join(self.chromosomes_path, f'{chr}_M.fasta')
            pat_ref = os.path.join(self.chromosomes_path, f'{chr}_P.fasta')

            self._create_files(mat_ref, pat_ref)
            subprocess.run(f"mv {self.tmp_path}/tmp.calls.vcf.gz {self.vcf_path}/{chr}_M.vcf.gz", shell=True, check=True)

            self._create_files(pat_ref, mat_ref)
            subprocess.run(f"mv {self.tmp_path}/tmp.calls.vcf.gz {self.vcf_path}/{chr}_P.vcf.gz", shell=True, check=True)

    def create_chain(self):
        # Get list of chromosomes with both maternal and paternal files
        chr_names = self.get_chr_names()
        for chr in chr_names:
            print(f"Active Chromosome: {chr} create chain file")
            mat_ref = os.path.join(self.chromosomes_path, f'{chr}_M.fasta')
            pat_ref = os.path.join(self.chromosomes_path, f'{chr}_P.fasta')

            self.chain_files_nf_LO(mat_ref, pat_ref, chr, "M_to_P")
            self.chain_files_nf_LO(pat_ref, mat_ref, chr, "P_to_M")

    def _create_files(self, reference, alignment, t=16):

                # Run Minimap2 alignment
        tmp_sam = f"{self.tmp_path}/tmp.sam"
        # Execute minimap2 command
        minimap2_command = f"/home/schmitzmf/minimap2/minimap2 -ax asm5 -t{t} {reference} {alignment} > {tmp_sam}"
        subprocess.run(minimap2_command, shell=True, check=True)

        bam_file = tmp_sam.replace('.sam', '.bam')
        sorted_bam_file = tmp_sam.replace('.sam', '.sorted.bam')
        output_vcf_gz = tmp_sam.replace('.sam', '.calls.vcf.gz')

        # Convert SAM to BAM
        subprocess.run(f"samtools view -bS {tmp_sam} > {bam_file}", shell=True, check=True)
        subprocess.run(f"samtools sort {bam_file} -o {sorted_bam_file}", shell=True, check=True)
        subprocess.run(f"samtools index {sorted_bam_file}", shell=True, check=True)
        # Clean up intermediate BAM file
        #subprocess.run(f"rm {bam_file}", shell=True, check=True)

        # Call variants and compress the VCF
        mpileup_command = f"bcftools mpileup -q 20 -Q 20 -f {reference} {sorted_bam_file} | bcftools call -mv -O u | bcftools norm -f {reference} -O z -o {output_vcf_gz}"       
        subprocess.run(mpileup_command, shell=True, check=True)

        # Index the compressed VCF file
        index_command = f"bcftools index {output_vcf_gz}"
        subprocess.run(index_command, shell=True, check=True)

    def chain_files_nf_LO(self, src_fasta, trg_fasta, chr, chain_direction):
        nx_command = f"nextflow run evotools/nf-LO --source {src_fasta} --target {trg_fasta} -profile local --aligner minimap2 --distance near --max_cpus 32 --outdir /home/schmitzmf/scratch/chainfiles_output"
        subprocess.run(nx_command, shell=True, check=True, cwd="/home/schmitzmf/nf-LO")
        subprocess.run(f"mv /home/schmitzmf/scratch/chainfiles_output/chainnet/liftover.chain {self.chain_path}/{chr}_{chain_direction}.chain", shell=True, check=True, cwd="/home/schmitzmf/nf-LO")

    def create_telomere_bam(self, tel_seq='TTAGGG', window_size=1000):
        chr_names = self.get_chr_names()

        # Define your variables
        tsv_file = os.path.join(self.tmp_path, f'tmp_telomeric_repeat_windows.tsv')
        len_telomere = len(tel_seq)

        name_list = []
        start_list = []
        end_list = []
        tel_list = []

        for chr in chr_names:
            for aln in ['M', 'P']:
                current_chr = f'{chr}_{aln}'
                fasta_file = os.path.join(self.chromosomes_path, f'{current_chr}.fasta')

                # Construct the command using the variables
                command = f"tidk search --string {tel_seq} --output tmp --dir {self.tmp_path} {fasta_file} --window {window_size}"
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                df = pd.read_csv(tsv_file, sep='\t')

                # Iterate over each row
                for index, row in df.iterrows():
                    # Check the condition
                    if 0.25 * window_size > (int(row[2]) + int(row[3])) * len_telomere:
                        telomere_end = row[1] - window_size
                        name_list.append(current_chr)
                        start_list.append(0)
                        end_list.append(telomere_end)
                        tel_list.append('telomere')
                        break  # Exit the loop since condition is met

                # Iterate over each row in reverse, starting from the second-to-last row
                for index in range(len(df) - 2, -1, -1):  # Start from second-to-last row, stop at first row
                    row = df.iloc[index]
                    #print(int(row[2]) + int(row[3]))
                    if 0.25 * window_size > (int(row[2]) + int(row[3])) * len_telomere:
                        telomere_end = row[1] + window_size
                        name_list.append(current_chr)
                        start_list.append(telomere_end)
                        end_list.append(df.iloc[-1][1])
                        tel_list.append('telomere')
                        break  # Exit the loop since condition is met

        telomere_df = pd.DataFrame({
            'name': name_list,
            'start': start_list,
            'end': end_list,
            'tel': tel_list
        })
        print(telomere_df)
        out_bed_path = os.path.join(self.ref_dir, 'telomere.bed')
        telomere_df.to_csv(out_bed_path, sep='\t', index=False, header=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_dir', type=str, default='/mnt/sod2-project/csb4/wgs/martin/genome_references/hg002_v101/',
                        help='Path to folder with train data tuples')
    parser.add_argument('--ref_name', type=str, default='hg002v1.0.1',
                        help='ref name')
    args = parser.parse_args()
    ref_obj = ProcessRefsObject(args.ref_dir, args.ref_name)

    #ref_obj.split_refs()
    #ref_obj.create_vcf()
    #ref_obj.create_centromere_refs()
    ref_obj.create_chain()


    #ref_obj.create_acrocentric_refs()
    #ref_obj.create_centromere_refs_nobed()
    #ref_obj.create_telomere_bam()
    print("Done. These are the Diploid chromosomes:")
    #chr, _, _ = ref_obj.get_chr_prefixes()
    #print(chr)
