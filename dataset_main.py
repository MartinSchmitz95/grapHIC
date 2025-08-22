#!/bin/env python
import argparse
import yaml
from dataset_object import HicDatasetCreator


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

    dataset_object.load_chromosome(genome, chr_id, ref_base)
    print(f'Processing {dataset_object.genome_str}...')

    if gen_step_config['sample_reads']:
        dataset_object.simulate_pbsim_reads()
        print(f"Done with reads simulation {chrN}_{i}")
    if gen_step_config['create_graphs']:
        dataset_object.create_graphs()
        print(f"Created gfa graph {chrN}_{i}")

    # run HiC pipeline
    if gen_step_config['hic']:
        dataset_object.process_hic()
        dataset_object.make_hic_edges()

    if gen_step_config['parse_gfa']:
        nx_graph = dataset_object.parse_gfa()
        dataset_object.pickle_save(nx_graph, dataset_object.nx_graphs_path)
        print(f"Saved nx graph {chrN}_{i}")

    elif gen_step_config['ground_truth'] or gen_step_config['diploid_features'] or gen_step_config['ml_graphs'] or gen_step_config['pile-o-gram']:
        nx_graph = dataset_object.load_nx_graph()
        print(f"Loaded nx graph {chrN}_{i}")

    # update graph to include hic edges
    if gen_step_config['hic']:
        nx_graph = dataset_object.load_nx_graph()
        # not sure if logic code in here is the prettiest, but should work
        hic_graph = dataset_object.load_hic_edges()
        print("loaded Hi-C graph")
        nx_graph = dataset_object.merge_graphs(nx_graph, hic_graph)
        print("merged Hi-C graph")
        dataset_object.pickle_save(nx_graph, dataset_object.merged_graphs_path)
        print("saved merged Hi-C graph")

    #if 'pile-o-gram' in gen_step_config:
    #    if gen_step_config['pile-o-gram']:
    #        print(f"Creating pog files with raft {chrN}_{i}")
    #        #dataset_object.run_raft()
    #        #dataset_object.nx_utg_ftrs(nx_graph)
    #        dataset_object.create_pog_features(nx_graph)
    #        print(f"Done with pog features {chrN}_{i}")
    #        dataset_object.pickle_save(nx_graph, dataset_object.nx_graphs_path)

    if dataset_object.diploid and gen_step_config['diploid_features']:
        dataset_object.pickle_save(nx_graph, dataset_object.nx_graphs_path)

    if not dataset_object.real and gen_step_config['ground_truth']:
        dataset_object.create_hh_features(nx_graph)
        print(f"Done with hh features {chrN}_{i}")
        dataset_object.pickle_save(nx_graph, dataset_object.nx_graphs_path)

    if gen_step_config['ml_graphs']:
        nx_graph = dataset_object.load_nx_graph(multi=True)
        #dataset_object.create_jellyfish_features(nx_graph)
        #dataset_object.calculate_coverage_statistics(nx_graph)
        #dataset_object.create_pog_features(nx_graph)
        #dataset_object.create_pileup_features(nx_graph)
        dataset_object.create_hh_features(nx_graph)
        dataset_object.pickle_save(nx_graph, dataset_object.merged_graphs_path)
        exit()
        nx_multi_reduced = dataset_object.convert_to_single_stranded(nx_graph)
        nx_multi_reduced = dataset_object.add_hic_neighbor_weights(nx_multi_reduced)
        dataset_object.save_to_dgl_and_pyg(nx_multi_reduced)
        dataset_object.split_and_save_pyg(nx_multi_reduced, index=i)
        print(f"Saved DGL and PYG graphs of {chrN}_{i}")

    print("Done for one chromosome!")

if __name__ == "__main__":
    main()
