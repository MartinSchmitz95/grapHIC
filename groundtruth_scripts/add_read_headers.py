#!/bin/env python3

import gzip
from collections import defaultdict
from functools import reduce

import networkx as nx
import pickle as pkl


## thoughts
## - write generator over read ids
## - add attributes for each
## - write fn to check if read is het/hom
## => VCFs as global state or fn args?
## => figure out vcf parsing => pysam?

def load_pickle(filepath) -> Dict:
    ret = None
    with open(filepath, 'rb') as f:
        ret = pickle.load(f)
    return ret

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        ret = pickle.dump(obj, f)

#TODO make call to handle het/homozygous
# set zygosity mapping as global variable?
def check_zygosity(chrid, start, end):
    return 'E' # for now, everything is hEterozygous


def iter_reads(inpath, gzipped=True):
    """
    Takes a fastq file produced by badreads and emits the headers containing the ground truth values as a dict to be added to the newtorkx graph along with each read ID
    """
    # handle gzipped/uncompressed files
    with gzip.open(inpath, "rt") if gzipped else open(inpath, "rt") as file:
        for line in file:
            # only parse headers
            if line[0] != '@':
                continue
            header = line.split(' ')[1:]
            assert len(header) == 4
            # header looks like
            # chr_id,+strand,start-end, length=x, error-free_legnth=x, read_identity=xx.yyy%
            pos = header[0].split(',')
            assert len(pos) == 3
            chr_id = header[0][0]
            strand = header[0][1][0] # should be always + or -
            assert strand == '+' or strand == '-'
            start, end = int(pos[2].split('-')[0]), int(pos[2].split('-')[1])
            assert start <= end and start >= 0 and end >= 0

            yield (idx, {'chr': chr_id, 'strand': strand, 'start': start, 'end': end})


def main(fastq_path, nx_path, read_to_node_dict_path, outpath):
    graph = load_pickle(nx_path)
    r2n = load_pickle(read_to_node_dict_path)

    ## group read headers by unitig ID
    utg_reads = defaultdict(list)
    for idx, vals in iter_reads(fastq_path):
        utg_reads[r2n[idx]].append(vals)

    ## set values for each unitig
    utg_vals = dict()
    for idx, headers in utg_reads:
        # check CHR and strand is matching
        assert reduce(lambda x, y: x == y, h['chr'] for h in headers)
        assert reduce(lambda x, y: x == y, h['strand'] for h in headers)

        # start is the earliest, end is the latest of all reads mapping to a utig
        start = min(h['start'] for h in headers)
        end = max(h['end'] for h in headers)

        zygosity = check_zygosity(h[0]['chr'], start, end)

        utg_vals[idx] = {'chr': h[0]['chr'], 'strand': h[0]['strand'], 'start': start, 'end': end, 'zygosity': zygosity}

    ## then add to graph and save to output
    nx.set_node_attributes(graph, utg_vals)

    save_pickle(graph, outpath)


if __name__ == '__main__':
    import argparse as ap


"""
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def process_fastq_gz_badread(self, input_file, all_reads, v, start_id=0):
    with gzip.open(input_file, "rt") as handle:
        for idx, record in enumerate(SeqIO.parse(handle, "fastq")):
            # Parse the header to extract relevant information
            parts = record.description.split()
            chr_info = parts[1].split(',')
            # Extract strand, start, and end from chr_info
            if len(chr_info) < 2:
                start, end = 0, 0
                strand = "+"
            else:
                strand = chr_info[1][0]  # Get the first character, which is either '-' or '+'
                start, end = map(int, chr_info[2].split('-'))

            # Create the new description
            new_description = f'strand={strand} start={start} end={end} variant={v} chr={self.chr_id[3:]}'

            # Create a new SeqRecord with updated description and a new id
            new_record = SeqRecord(
                seq=record.seq,
                id=f"{idx+start_id}",
                description=new_description,
                letter_annotations={"phred_quality": record.letter_annotations["phred_quality"]}
            )
            #new_record.letter_annotations["phred_quality"] = record.letter_annotations["phred_quality"]
            # Add the new record to the list
            all_reads.append(new_record)
    return idx
"""
