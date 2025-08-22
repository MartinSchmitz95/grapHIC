import argparse
import subprocess
import os
import eval

def parse_args():
    parser = argparse.ArgumentParser(description='Run and evaluate HiFiasm assembly from haplotype FASTA files')
    
    # Haplotype FASTA inputs
    parser.add_argument('--hap1_fasta', type=str, required=True, help='Haplotype 1 FASTA file (e.g., from graphic_pred_nodes.py)')
    parser.add_argument('--hap2_fasta', type=str, required=True, help='Haplotype 2 FASTA file (e.g., from graphic_pred_nodes.py)')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--threads', type=int, default=32, help='Number of threads (default: 32)')
    parser.add_argument('--hifiasm_path', type=str, default='/home/schmitzmf/hifiasm_025/hifiasm', help='Path to hifiasm executable')
    
    # Evaluation arguments
    parser.add_argument('--ref_m', type=str, help='Maternal reference genome')
    parser.add_argument('--ref_p', type=str, help='Paternal reference genome')
    parser.add_argument('--mat_yak', type=str, help='Maternal YAK index')
    parser.add_argument('--pat_yak', type=str, help='Paternal YAK index')
    parser.add_argument('--yak_path', type=str, default='/home/schmitzmf/yak/yak', help='Path to yak executable')
    parser.add_argument('--minigraph_path', type=str, default='/home/schmitzmf/minigraph/minigraph', help='Path to minigraph executable')
    
    # Assembly options
    parser.add_argument('--chr_name', type=str, default='asm', help='Assembly name prefix')
    parser.add_argument('--skip_hifiasm', action='store_true', default=False, help='Skip hifiasm run')
    parser.add_argument('--purge_level', type=int, default=3, help='Purge level for hifiasm (default: 3)')
    parser.add_argument('--hifiasm_extra_args', type=str, default='', help='Extra arguments for hifiasm')

    return parser.parse_args()

def gfa_to_fasta(gfa_path, fasta_path):
    """Convert GFA to FASTA using awk."""
    cmd = f"awk '/^S/{{print \">\"$2;print $3}}' {gfa_path} > {fasta_path}"
    subprocess.run(cmd, shell=True)

def run_hifiasm_on_haplotype(hap_fasta, output_prefix, hifiasm_path, threads, purge_level, extra_args=""):
    """
    Run hifiasm on a single haplotype FASTA file.
    Uses purge mode to clean up the assembly.
    """
    print(f"Running hifiasm on {hap_fasta}...")
    
    # Clean up any existing files
    subprocess.run(f"rm -f {output_prefix}.*", shell=True)
    
    # Run hifiasm in purge mode
    cmd = f'{hifiasm_path} -o {output_prefix} -r{purge_level} -t{threads} {extra_args} {hap_fasta}'
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Warning: hifiasm returned non-zero exit code {result.returncode}")
    
    # Expected output files from hifiasm
    primary_gfa = f"{output_prefix}.bp.p_ctg.gfa"
    alternate_gfa = f"{output_prefix}.bp.a_ctg.gfa"
    
    # Check which files exist
    if os.path.exists(primary_gfa):
        print(f"Primary contigs: {primary_gfa}")
        return primary_gfa
    elif os.path.exists(alternate_gfa):
        print(f"Using alternate contigs: {alternate_gfa}")
        return alternate_gfa
    else:
        print(f"Warning: No expected GFA output found for {output_prefix}")
        # Look for any .gfa files
        output_dir = os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.'
        gfa_files = [f for f in os.listdir(output_dir) if f.startswith(os.path.basename(output_prefix)) and f.endswith('.gfa')]
        if gfa_files:
            gfa_path = os.path.join(output_dir, gfa_files[0])
            print(f"Using found GFA file: {gfa_path}")
            return gfa_path
        else:
            raise FileNotFoundError(f"No GFA output found for {output_prefix}")

def main():
    args = parse_args()
    
    # Set output directory
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.hap1_fasta)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Validate input files
    if not os.path.exists(args.hap1_fasta):
        raise FileNotFoundError(f"Hap1 FASTA file not found: {args.hap1_fasta}")
    if not os.path.exists(args.hap2_fasta):
        raise FileNotFoundError(f"Hap2 FASTA file not found: {args.hap2_fasta}")
    
    print(f"Processing haplotype FASTA files:")
    print(f"  Hap1: {args.hap1_fasta}")
    print(f"  Hap2: {args.hap2_fasta}")
    print(f"  Output directory: {args.out_dir}")
    
    chr_name = args.chr_name
    
    # Run HiFiasm on both haplotype FASTA files
    if not args.skip_hifiasm:
        # Run hifiasm on hap1
        hap1_prefix = os.path.join(args.out_dir, f"{chr_name}_hap1")
        hap1_gfa = run_hifiasm_on_haplotype(
            args.hap1_fasta, hap1_prefix, args.hifiasm_path, 
            args.threads, args.purge_level, args.hifiasm_extra_args
        )
        
        # Run hifiasm on hap2
        hap2_prefix = os.path.join(args.out_dir, f"{chr_name}_hap2")
        hap2_gfa = run_hifiasm_on_haplotype(
            args.hap2_fasta, hap2_prefix, args.hifiasm_path, 
            args.threads, args.purge_level, args.hifiasm_extra_args
        )
    else:
        print("Skipping hifiasm run")
        hap1_gfa = os.path.join(args.out_dir, f"{chr_name}_hap1.bp.p_ctg.gfa")
        hap2_gfa = os.path.join(args.out_dir, f"{chr_name}_hap2.bp.p_ctg.gfa")
    
    # Convert GFA to FASTA
    hap1_asm = os.path.join(args.out_dir, f"{chr_name}_hap1.fasta")
    hap2_asm = os.path.join(args.out_dir, f"{chr_name}_hap2.fasta")
    
    # Convert GFA to FASTA if files exist
    if os.path.exists(hap1_gfa):
        print(f"Converting {hap1_gfa} to FASTA...")
        gfa_to_fasta(hap1_gfa, hap1_asm)
    else:
        print(f"Warning: Hap1 GFA not found: {hap1_gfa}")
    
    if os.path.exists(hap2_gfa):
        print(f"Converting {hap2_gfa} to FASTA...")
        gfa_to_fasta(hap2_gfa, hap2_asm)
    else:
        print(f"Warning: Hap2 GFA not found: {hap2_gfa}")
    
    # Only proceed with evaluation if we have reference genomes
    if not args.ref_m or not args.ref_p:
        print("No reference genomes provided, skipping minigraph evaluation")
        print(f"Assembly complete. Output files:")
        if os.path.exists(hap1_asm):
            print(f"  Hap1 assembly: {hap1_asm}")
        if os.path.exists(hap2_asm):
            print(f"  Hap2 assembly: {hap2_asm}")
        return
    
    print("\nEvaluating haplotype assemblies...")

    # Setup evaluation paths
    hap1_report = os.path.join(args.out_dir, "hap1_minigraph.txt")
    hap1_paf = os.path.join(args.out_dir, "hap1_asm.paf")
    hap1_phs = os.path.join(args.out_dir, "hap1_phs.txt")
    
    hap2_report = os.path.join(args.out_dir, "hap2_minigraph.txt")
    hap2_paf = os.path.join(args.out_dir, "hap2_asm.paf")
    hap2_phs = os.path.join(args.out_dir, "hap2_phs.txt")

    idx_m = args.ref_m + '.fai'
    idx_p = args.ref_p + '.fai'
    
    # Evaluate hap1 assembly against maternal reference
    if os.path.exists(hap1_asm):
        print(f"Evaluating hap1 assembly against maternal reference...")
        p = eval.run_minigraph(args.ref_m, hap1_asm, hap1_paf, minigraph_path=args.minigraph_path)
        p.wait()
        p = eval.parse_pafs(idx_m, hap1_report, hap1_paf, paf_path=None)
        p.wait()
        eval.parse_minigraph_for_full(hap1_report)
    
    # Evaluate hap2 assembly against paternal reference
    if os.path.exists(hap2_asm):
        print(f"Evaluating hap2 assembly against paternal reference...")
        p = eval.run_minigraph(args.ref_p, hap2_asm, hap2_paf, minigraph_path=args.minigraph_path)
        p.wait()
        p = eval.parse_pafs(idx_p, hap2_report, hap2_paf, paf_path=None)
        p.wait()
        eval.parse_minigraph_for_full(hap2_report)
    
    # Run YAK evaluation if YAK indices are provided
    if args.mat_yak and args.pat_yak:
        print("Running YAK evaluation...")
        procs = []
        
        if os.path.exists(hap1_asm):
            p1 = eval.run_yak(args.mat_yak, args.pat_yak, hap1_asm, hap1_phs, 
                            yak_path=args.yak_path, threads=args.threads)
            procs.append(p1)
        
        if os.path.exists(hap2_asm):
            p2 = eval.run_yak(args.mat_yak, args.pat_yak, hap2_asm, hap2_phs, 
                                yak_path=args.yak_path, threads=args.threads)
            procs.append(p2)
        
        for p in procs:
            p.wait()
        
        # Parse results if both evaluations were run
        if os.path.exists(hap1_phs) and os.path.exists(hap2_phs):
            eval.parse_real_results(hap1_report, hap2_report, hap1_phs, hap2_phs)
    else:
        print("No YAK indices provided, skipping YAK evaluation")

    print(f"\nEvaluation complete. Results saved to {args.out_dir}")

if __name__ == "__main__":
    main()
