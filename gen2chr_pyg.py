import os
import torch
import argparse
from torch_geometric.data import Data
from tqdm import tqdm

def split_graph_by_chr(input_file, output_dir):
    """
    Load a PyG graph, split it by chromosome, and save the resulting graphs.
    
    Args:
        input_file: Path to the input PyG graph file
        output_dir: Directory to save the split graphs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the PyG graph with map_location to CPU
    print(f"Loading graph from {input_file}...")
    data = torch.load(input_file, map_location=torch.device('cpu'))
    
    # Check if 'chr' attribute exists
    if not hasattr(data, 'chr'):
        raise ValueError("The graph does not have a 'chr' attribute")
    
    # Get unique chromosomes
    unique_chrs = torch.unique(data.chr).tolist()
    print(f"Found {len(unique_chrs)} unique chromosomes: {unique_chrs}")
    
    # Split the graph by chromosome
    for chr_num in tqdm(unique_chrs, desc="Splitting by chromosome"):
        # Create mask for nodes in this chromosome
        chr_mask = data.chr == chr_num
        
        # Get node indices for this chromosome
        chr_node_indices = torch.nonzero(chr_mask).squeeze()
        
        if chr_node_indices.dim() == 0:  # Handle single-node case
            chr_node_indices = chr_node_indices.unsqueeze(0)
        
        # Create a mapping from old node indices to new node indices
        old_to_new = {int(old_idx): new_idx for new_idx, old_idx in enumerate(chr_node_indices)}
        
        # Get edges where both nodes are in this chromosome
        edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[0, i], data.edge_index[1, i]
            if chr_mask[src] and chr_mask[dst]:
                edge_mask[i] = True
        
        # Extract subgraph data
        sub_edge_index = data.edge_index[:, edge_mask]
        sub_edge_type = data.edge_type[edge_mask] if hasattr(data, 'edge_type') else None
        sub_edge_weight = data.edge_weight[edge_mask] if hasattr(data, 'edge_weight') else None
        
        # Remap node indices in edge_index
        for i in range(sub_edge_index.size(1)):
            sub_edge_index[0, i] = old_to_new[int(sub_edge_index[0, i])]
            sub_edge_index[1, i] = old_to_new[int(sub_edge_index[1, i])]
        
        # Extract node features
        sub_x = data.x[chr_node_indices]
        
        # Extract ground truth labels if available
        sub_y = data.y[chr_node_indices] if hasattr(data, 'y') else None
        
        # Extract original IDs if available
        sub_original_id = data.original_id[chr_node_indices] if hasattr(data, 'original_id') else None
        
        # Create new PyG Data object
        sub_data_dict = {
            'x': sub_x,
            'edge_index': sub_edge_index,
            'chr': torch.full((chr_node_indices.size(0),), chr_num, dtype=torch.long)
        }
        
        if sub_edge_type is not None:
            sub_data_dict['edge_type'] = sub_edge_type
        
        if sub_edge_weight is not None:
            sub_data_dict['edge_weight'] = sub_edge_weight
        
        if sub_y is not None:
            sub_data_dict['y'] = sub_y
        
        if sub_original_id is not None:
            sub_data_dict['original_id'] = sub_original_id
        
        sub_data = Data(**sub_data_dict)
        
        # Save the subgraph
        chr_name = f"chr{chr_num}"
        
        # Extract the base filename without extension
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Check if the filename ends with _n (where n is a number)
        if base_filename.split('_')[-1].isdigit():
            # Split the base filename to separate the numeric suffix
            parts = base_filename.rsplit('_', 1)
            main_part = parts[0]
            numeric_suffix = parts[1]
            # Format: main_part_chrX_numeric_suffix.pt
            output_file = os.path.join(output_dir, f"{main_part}_{chr_name}_{numeric_suffix}.pt")
        else:
            # No numeric suffix, use the original format
            output_file = os.path.join(output_dir, f"{base_filename}_{chr_name}.pt")
            
        torch.save(sub_data, output_file)
        print(f"Saved {chr_name} graph with {sub_data.num_nodes} nodes and {sub_data.num_edges} edges to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split PyG graph by chromosome')
    parser.add_argument('--input', type=str, required=True, help='Input PyG graph file (.pt)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for split graphs')
    
    args = parser.parse_args()
    
    split_graph_by_chr(args.input, args.output_dir)
