#!/bin/python

import sys

import cooler

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

infile = sys.argv[1]
#outfile = sys.argv[2]
