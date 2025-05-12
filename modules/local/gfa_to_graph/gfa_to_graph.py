#!/bin/env python3

import networkx as nx

import pickle
import re
from collections import Counter
from datetime import datetime

complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
mapper = lambda c: complement[c]


def revc(seq) -> str:
    return ''.join(map(mapper, seq))[::-1]

def pickle_save(pickle_object, path):
    # Save the graph using pickle
    with open(path, 'wb') as f:
        pickle.dump(pickle_object, f) 
    print(f"File saved successfully to {path}")

def process_gfa_to_files(gfa_path,
              utg_to_node_path='./utg_to_node.dict.pkl',
              utg_to_reads_path='./utg_to_reads.dict.pkl',
              nx_graph_path='./utg_graph.nx.pkl'):

    nx_graph, read_seqs, utg_to_node, utg_to_reads = nx_from_gfa(gfa_path, compute_seqs=False)
    # Save data
    pickle_save(utg_to_node, utg_to_node_path)
    pickle_save(utg_to_reads, utg_to_reads_path)
    pickle_save(nx_graph, nx_graph_path)

def nx_from_gfa(gfa_path, diploid=False, compute_seqs=False):
    graph_nx = nx.DiGraph()
    read_to_node, node_to_read, old_read_to_utg = {}, {}, {}
    unitig_2_node = {}
    utg_2_reads = {}
    #edges_dict = {}
    read_lengths, read_seqs, covs = {}, {}, {}  # Obtained from the GFA
    read_idxs, read_strands, read_starts, read_ends, read_chrs, read_variants, variant_class = {}, {}, {}, {}, {}, {}, {}  # Obtained from the FASTA/Q headers
    edge_ids, prefix_lengths, overlap_lengths, overlap_similarities = {}, {}, {}, {}

    time_start = datetime.now()
    print(f'Starting to loop over GFA')
    with open(gfa_path) as ff:
        node_idx = 0
        edge_idx = 0

        ## We assume that the first N lines start with "S"
        ## And next M lines start with "L"
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
                    tag, id, sequence, length, count, cov = line
                if len(line) == 5:
                    tag, id, sequence, length, cov = line 
                if len(line) == 4:
                    tag, id, sequence, length = line
                if sequence == '*':
                    no_seqs_flag = True
                    sequence = '*' * int(length[5:])

                length = int(length[5:])

                real_idx = node_idx
                virt_idx = node_idx + 1
                read_to_node[id] = (real_idx, virt_idx)
                node_to_read[real_idx] = id
                node_to_read[virt_idx] = id

                graph_nx.add_node(real_idx)  # real node = original sequence
                graph_nx.add_node(virt_idx)  # virtual node = rev-comp sequence

                if compute_seqs:
                    read_seqs[real_idx] = sequence
                    read_seqs[virt_idx] = revc(sequence)

                read_lengths[real_idx] = length
                read_lengths[virt_idx] = length

                covs[real_idx] = cov
                covs[virt_idx] = cov

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

                ## This enforces similarity between the edge and its "virtual pair"
                ## Meaning if there is A -> B and B^rc -> A^rc they will have the same overlap_length
                ## When parsing CSV that was not necessarily so:
                ## Sometimes reads would be slightly differently aligned from their RC pairs
                ## Thus resulting in different overlap lengths

                try:
                    ol_length = int(cigar[:-1])  # Assumption: this is overlap length and not a CIGAR string
                except ValueError:
                    print('Cannot convert CIGAR string into overlap length!')
                    raise ValueError

                overlap_lengths[(src_real, dst_real)] = ol_length
                overlap_lengths[(src_virt, dst_virt)] = ol_length
    
    elapsed = (datetime.now() - time_start).seconds
    print(f'Elapsed time: {elapsed}s')

    nx.set_node_attributes(graph_nx, read_lengths, 'read_length')
    nx.set_node_attributes(graph_nx, variant_class, 'variant_class')
    nx.set_node_attributes(graph_nx, covs, 'cov')
    node_attrs = ['read_length', 'variant_class', 'cov']

    return graph_nx, read_seqs, unitig_2_node, utg_2_reads

    ## some old code graveyard? not deleting for now

    #nx.set_edge_attributes(graph_nx, prefix_lengths, 'prefix_length')
    #nx.set_edge_attributes(graph_nx, overlap_lengths, 'overlap_length')
    #edge_attrs = ['prefix_length', 'overlap_length']
    #nx.set_edge_attributes(graph_nx, overlap_similarities, 'overlap_similarity')
    #edge_attrs.append('overlap_similarity')

    # Create a dictionary of nodes and their direct successors
    #successor_dict = {node: list(graph_nx.successors(node)) for node in graph_nx.nodes()}

    # Why is this the case? Is it because if there is even a single 'A' file in the .gfa, means the format is all 'S' to 'A' lines?


def main(args):
    import argparse as ap

    parser = ap.ArgumentParser(description="constructs a networkx graph from hifiasm GFA output")

    parser.add_argument("-i", dest='infile', default='-', type=ap.FileType('r'), help="Input GFA. Default stdin.")
    parser.add_argument("--node-dict", dest='utg_to_node', default='./utg_to_node.dict.pkl', type=ap.FileType('w'), help="Where to write the unitig to node dict. Default './utg_to_node.dict.pkl'.")
    parser.add_argument("--reads-dict", dest='utg_to_reads', default='./utg_to_reads.dict.pkl', type=ap.FileType('w'), help="Where to write the unitig to reads dict. Default './utg_to_reads.dict.pkl'.")
    parser.add_argument("--utg-graph", dest='utg_graph', default='./utg_graph.nx.pkl', type=ap.FileType('w'), help="Where to write the unitig graph. Default './utg_graph.nx.pkl'.")

    args = parser.parse_args()
    process_gfa_to_files(args.infile.name, args.utg_to_node.name, args.utg_to_reads.name, args.utg_graph.name)

if __name__ == '__main__':
    import sys
    main(sys.argv)
