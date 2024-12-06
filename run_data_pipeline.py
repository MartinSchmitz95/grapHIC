import argparse
import yaml
from raven_dataset_creator import RavenDatasetCreator
from hifiasm_dataset_creator import HifiasmDatasetCreator
from data_gen_utils import create_full_dataset_dict

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

    elif gen_step_config['ground_truth'] or gen_step_config['diploid_features'] or gen_step_config['ml_graphs']:
        nx_graph = dataset_object.load_nx_graph()
        print(f"Loaded nx graph {chrN}_{i}")


    if dataset_object.diploid and gen_step_config['diploid_features']:
        if not dataset_object.real:
            dataset_object.create_hh_features(nx_graph)
            print(f"Done with hh features {chrN}_{i}")
        dataset_object.add_trio_binning_labels(nx_graph)
        print(f"Done with trio binning {chrN}_{i}")
        dataset_object.pickle_save(nx_graph, dataset_object.nx_graphs_path)

    if not dataset_object.real and gen_step_config['ground_truth']:
        #dataset_object.get_telomere_ftrs(nx_graph)
        dataset_object.create_gt(nx_graph)
        print(f"Done with ground truth creation {chrN}_{i}")
        dataset_object.add_decision_attr(nx_graph)
        print(f"Done with decision node creation {chrN}_{i}")
        dataset_object.pickle_save(nx_graph, dataset_object.nx_graphs_path)

    if gen_step_config['ml_graphs']:
        dataset_object.save_to_dgl_and_pyg(nx_graph)
        print(f"Saved DGL and PYG graphs of {chrN}_{i}")

    print("Done for one chromosome!")

if __name__ == "__main__":
    main()