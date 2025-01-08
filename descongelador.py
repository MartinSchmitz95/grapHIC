#!/bin/python

import sys
import tempfile
import math

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

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

def save_np_matrix(intuple, path, sep='\t'):
    np.savetxt(path, intuple[1], fmt='%d', delimiter=sep, header=sep.join(intuple[0]))

def export_image(intuple, path, scale=id):
    """
    Plots the interchromosomal contacts given by `intuple` as an image that is saved at `path`.
    Applies a scaling function passed as `scale` to the matrix first.
    The default is the identity, but other scaling factors can be passed as a lambda (e.g. `lambda x: np.log(x+1)` for log scaling with a pseudocount of 1).
    """
    plt.imsave(path, scale(intuple[1]).astype(np.float64))



def main(args):
    #TODO do proper argparsing later
    infile = args[1]
    outfile = args[2]

    np_tup = to_np_matrix(
            aggr_chrs(infile)
            )

    #print(np_tup)

    save_np_matrix(np_tup, outfile)

    export_image(np_tup, outfile + ".png", scale=lambda x: np.log(x+1))



if __name__ == '__main__':
    main(sys.argv)
