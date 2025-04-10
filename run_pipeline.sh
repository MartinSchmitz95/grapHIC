#!/bin/bash

## make sure we are in an env with appropriate tools installed
## apart from nextflow, needed are (should all be on bioconda)
## - bowtie2
## - hifiasm
## - fastqc
## - multiqc
## - python
## - cooler
micromamba activate nf-core

mkdir $(basename $1 .csv) # to be sure the dir exists

nextflow run -o $(basename $1 .csv).log ~/graphic/main.nf --input $1 -bg --dnase --outdir $(basename $1 .csv)
# -profile conda # if not running in a conda env above
