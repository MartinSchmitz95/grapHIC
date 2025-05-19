#!/bin/bash

# Check if mode parameter is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <mode>"
  echo "Example: $0 louvain"
  exit 1
fi

MODE=$1
GRAPH_DIR="/mnt/sod2-project/csb4/wgs/lovro_interns/leon/multi_chrom_dataset/pyg_edge_gates"

# List of graph files to process
GRAPH_FILES="
  i002c_v04_multi_21_chr10_0.pt
  i002c_v04_multi_21_chr10_1.pt
  i002c_v04_multi_21_chr10_2.pt
  i002c_v04_multi_21_chr19_0.pt
  i002c_v04_multi_21_chr19_1.pt
  i002c_v04_multi_21_chr19_2.pt
  i002c_v04_multi_21_chr15_0.pt
  i002c_v04_multi_21_chr15_1.pt
  i002c_v04_multi_21_chr15_2.pt
  i002c_v04_multi_21_chr22_0.pt
  i002c_v04_multi_21_chr22_1.pt
  i002c_v04_multi_21_chr22_2.pt
"

echo "Running evaluation with mode: $MODE"
echo "----------------------------------------"

# Process each graph file
for graph_file in $GRAPH_FILES; do
  graph_path="${GRAPH_DIR}/${graph_file}"
  echo "Processing graph: $graph_file"
  
  python eval_main.py --graph_path "$graph_path" --mode "$MODE"
  
  echo "----------------------------------------"
done

echo "Evaluation complete!"
