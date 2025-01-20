#!/bin/python

import sys
import tempfile
import math
import itertools

from typing import Tuple, Dict, Callable

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle

import cooler
from cooler import Cooler

## THOUGHTS
# file formats:
#     - support cool primarily
#     - maybe direct from .pairs/text file as well
#     - use straw for .hic support? seems messy...
# in principle this should be equivalent to `cooler coarsen` with an unlimited factor
# yes, this works
# just call 
# https://github.com/open2c/cooler/blob/cfd2c4990a2c52deb5f3d277c977daae6e59f6c4/src/cooler/cli/coarsen.py#L58
# then export as numpy matrix

def aggr_chrs(in_path: str) -> Cooler:
    """
    Takes a path to a cooler in any resolution, calls coarsen to aggregate counts by chromosome.
    :returns: A new Cooler stored in a tmpfile with counts aggregated by chromosome.
    """
    # cooler requires a URI on the filesystem, so use a tmpfile
    tmp = tempfile.NamedTemporaryFile().name

    #TODO calculate exact factor?
    cooler.coarsen_cooler(in_path, tmp, sys.maxsize, 900) # factor should be infinity

    return Cooler(tmp)


def to_np_matrix(in_cooler: Cooler, balance=False) -> Tuple[np.array, np.ndarray]:
    """
    Extracts the data in a cooler aggregated by chromosomes into a numpy matrix.
    If `balance` is set to `True`, use weights (need to be computed by 
    :returns: a tuple of a 1D numpy array containing chromosome labels and a 2D array containing counts between chromosomes.
    """
    if balance:
        cooler.balance_cooler(in_cooler, store=True)

    # this returns it in a bit of a weird format, necessitating [:]
    # could also call fetch(chr, 0, 1) in a loop, but I think this is neater
    array = np.array(in_cooler.matrix(balance=balance)[:])
    chrnames = np.array(in_cooler.chromnames)

    if len(array) != len(chrnames):
        raise ValueError("Labels not same length as array! Might not have been aggregated by Chr.")

    # using the weights seems to induce NaN values in the sex chrs, as they have 0 counts
    # set these to 0 manually
    np.nan_to_num(array, copy=False)

    return (chrnames, array)

def to_graph(in_cooler: Cooler, get_idtup: Callable, balance=False) -> nx.MultiGraph:
    """
    Takes a coarsened cooler and a lambda mapping unitig names to pairs of node ids and constructs a graph from it.
    :returns: a networkx MultiGraph with two nodes corresponding to one contig/its complement.
    The edges in the graph represent hic contacts between the two contigs.
    """
    # check cooler and nodes_dict match
    assert len(in_cooler.chromnames) == len(nodes_dict.keys())
    # get range of node ids
    max_node_id = max(max(x[0], x[1]) for x in nodes_dict.values())
    assert max_node_id == 2*len(nodes_dict) - 1 # indices start with 0
    # check it starts with 0
    assert min(min(x[0], x[1]) for x in nodes_dict.values()) == 0

    # thoughts: could also use cooler.rename()
    # but can't handle multiple substitution, and will write to disc unnecessarily
    # so do iteration ourselves
    # maybe try again if too slow
    # maybe do twice & merge the multigraphs to handle complement IDs

    # init the graph with all nodes
    ret = nx.MultiGraph()
    ret.add_nodes_from(range(max_node_id))

    mat = in_cooler.matrix(balance=balance)
    # helper lambda to generate full pairwise edges between uncomplemented and complemented node ids
    full_pairwise = lambda idtup1, idtup2, val: [
            (idtup1[0], idtup2[0], val),
            (idtup1[0], idtup2[1], val),
            (idtup1[1], idtup2[0], val),
            (idtup1[1], idtup2[1], val),
            ]
    # iterate through every two distinct nodes once
    ret.add_weighted_edges_from(
            # value needs to be coerced to single float
            itertools.chain.from_iterable( # apparently the standard way to flatMap in python
                                          full_pairwise(get_idtup(i_lab), get_idtup(j_lab), float(mat[i, j][:]))
                                          for j, j_lab in enumerate(in_cooler.chromnames)
                                          for i, i_lab in enumerate(in_cooler.chromnames)
                                          if j > i # will not generate self-edges
                                          #if i != j # will not generate self-edges
                                          ))
    return ret

def load_pickle(filepath) -> Dict:
    ret = None
    with open(filepath, 'rb') as f:
        ret = pickle.load(f)
    return ret

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        ret = pickle.dump(obj, f)

def save_np_matrix(intuple, path, sep='\t'):
    np.savetxt(path, intuple[1], fmt='%d', delimiter=sep, header=sep.join(intuple[0]))

def export_image(intuple, path, scale=id):
    """
    Plots the interchromosomal contacts given by `intuple` as an image that is saved at `path`.
    Applies a scaling function passed as `scale` to the matrix first.
    The default is the identity, but other scaling factors can be passed as a lambda (e.g. `lambda x: np.log(x+1)` for log scaling with a pseudocount of 1).
    """
    plt.imsave(path, scale(intuple[1]).astype(np.float64))

def export_connection_graph(infile, outfile, unitig_dict, read_dict):
    print("aggregating cooler")
    c = aggr_chrs(infile)
    print("loading contig dict")
    unitig_dict = load_pickle(unitig_dict)
    read_dict = load_pickle(read_dict)
    print("constructing graph")
    graph = to_graph(c, lambda x: read_dict[unitig_dict[x][0]])
    print("saving graph")
    save_pickle(graph, outfile)

def main(args):
    #TODO do proper argparsing later
    infile = args[1]
    outfile = args[2]
    unitig_dict = load_pickle(args[3])
    read_dict = load_pickle(args[4])

    c = aggr_chrs(infile)

    np_tup = to_np_matrix(c)

    #print(np_tup)

    #print("saving NP matrix")
    #save_np_matrix(np_tup, outfile + '.tsv')

    print("converting to MultiGraph")
    graph = to_graph(c, lambda x: read_dict[unitig_dict[x][0]])

    print("saving MultiGraph")
    save_pickle(c, outfile + '.nx.pickle')

    print("plotting image")
    export_image(np_tup, outfile + ".png", scale=lambda x: np.log(x+1))


if __name__ == '__main__':
    main(sys.argv)
