#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

print(*"hello  world")

# Function to create the grouped plot
def create_grouped_plot(data_lists, method_names=None, group_names=None, title="Grouped Data Plot", groups_to_plot=None, save_path=None):
    """
    Create a plot where data points are grouped by chromosomes.
    
    Parameters:
    - data_lists: List of lists, each containing 12 data points
    - method_names: Names of the methods (datasets)
    - group_names: Names of the groups (chromosomes)
    - title: Title of the plot
    - groups_to_plot: List of group indices to include in this plot
    - save_path: Path to save the figure (if None, figure is not saved)
    """
    # Number of lists (datasets/methods)
    num_datasets = len(data_lists)
    
    # Set default method names if not provided
    if method_names is None:
        method_names = [f"method{i+1}" for i in range(num_datasets)]
    
    # Set default group names if not provided
    if group_names is None:
        group_names = ["chr10", "chr19", "chr15", "chr22"]
    
    # Default to plotting all groups if not specified
    if groups_to_plot is None:
        groups_to_plot = list(range(len(group_names)))
    
    # Number of groups to plot in this figure
    num_groups = len(groups_to_plot)
    
    # Number of points per group
    points_per_group = 3
    
    # A4 width in inches (8.27) and height as 1/3 of A4 height (11.69/3 ≈ 3.9)
    fig, ax = plt.subplots(figsize=(8.27, 3.0))
    
    # Width of each bar
    bar_width = 0.8
    # Space between groups
    group_spacing = 0.5
    
    # Colors for different methods
    colors = plt.cm.viridis(np.linspace(0, 1, num_datasets))
    
    # Plot each chromosome group on x-axis, with methods side by side within each group
    for i, group_idx in enumerate(groups_to_plot):
        for method_idx in range(num_datasets):
            # Get the 3 data points for this method and chromosome
            start_idx = group_idx * points_per_group
            group_data = data_lists[method_idx][start_idx:start_idx + points_per_group]
            
            # Calculate x position for this method within the chromosome group
            x_pos = i * (num_datasets * bar_width + group_spacing) + method_idx * bar_width
            
            # Get the color for this method
            method_color = colors[method_idx]
            # Create a darker version of the same color for the lines
            darker_color = np.array(method_color) * 0.7  # Multiply RGB values by 0.7 to darken
            darker_color[3] = 1.0  # Keep alpha at 1.0
            
            # Add Tukey box plot
            bp = ax.boxplot([group_data], positions=[x_pos], widths=bar_width*0.7, 
                           patch_artist=True, showfliers=True)
            
            # Customize the box plot appearance with darker version of the method color
            for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=darker_color, linewidth=1.5)
            
            # Set the box outline to the darker color and fill to the original color
            for patch in bp['boxes']:
                patch.set(facecolor=method_color, alpha=0.7, edgecolor=darker_color, linewidth=1.0)
    
    # Set x-ticks for each method within each chromosome group
    x_ticks = []
    x_tick_labels = []
    
    for i, group_idx in enumerate(groups_to_plot):
        # Add dividing line in the center of the figure
        if i < num_groups - 1 and num_groups > 1:
            # Calculate the position between the current group and the next group
            current_group_end = i * (num_datasets * bar_width + group_spacing) + num_datasets * bar_width
            next_group_start = (i + 1) * (num_datasets * bar_width + group_spacing)
            divider_pos = (current_group_end + next_group_start) / 2 - 0.5
            # Adjust divider position to be closer to the last sample of the current group
            ax.axvline(x=divider_pos, color='gray', linestyle='--', alpha=0.7)
        
        # Add chromosome label above the method names
        middle_pos = i * (num_datasets * bar_width + group_spacing) + (num_datasets * bar_width) / 2 - bar_width/2
        ax.text(middle_pos, ax.get_ylim()[1] * 1.05, group_names[group_idx], 
                ha='center', va='bottom', fontweight='bold')
        
        # Add method ticks within each chromosome group
        for method_idx in range(num_datasets):
            x_pos = i * (num_datasets * bar_width + group_spacing) + method_idx * bar_width
            x_ticks.append(x_pos)
            x_tick_labels.append(method_names[method_idx])
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    
    # Add labels and title
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    
    # No legend
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
    return fig, ax

# Example usage
if __name__ == "__main__":
    # Example data: 3 lists, each with 12 data points
    # Each list has 4 groups (chromosomes) with 3 data points each


    accuracies_classic = [
        [0.506, 0.5009, 0.5035, 0.5244, 0.5014, 0.5188, 0.5171, 0.5079, 0.517, 0.5102, 0.5274, 0.5],  # Spectral
        [0.5124, 0.5074, 0.5178, 0.5009, 0.5207, 0.5069, 0.5171, 0.5096, 0.5066, 0.5278, 0.5115, 0.5174],  # Louvain
        [0.5097, 0.5042, 0.5072, 0.5319, 0.5159, 0.5109, 0.5171, 0.5085, 0.5088, 0.5129, 0.5, 0.5027],  # LP
        [0.8705, 0.8765, 0.8824, 0.8433, 0.8924, 0.8765, 0.8074, 0.8734, 0.874, 0.8236, 0.8718, 0.8847],  # grapHIC
    ]

    omegas_classic = [
        [0.0004, 0.0035, 0.0006, -0.1548, -0.1482, -0.2139, -0.0684, -0.0832, -0.0554, -0.077, -0.0716, -0.0836],  # Spectral
        [-0.0001, -0.0003, 0.0006, -0.0077, -0.0174, -0.0012, 0.0006, -0.0015, -0.001, -0.0053, -0.0009, -0.001],  # Louvain
        [-0.0001, -0.0003, -0.0003, -0.0582, -0.0702, -0.0541, -0.0006, -0.0019, -0.0008, -0.016, -0.0069, -0.0231],  # LP
        [0.5465, 0.5511, 0.5452, 0.4764, 0.552, 0.5612, 0.3778, 0.5069, 0.5131, 0.4238, 0.5171, 0.5634],  # grapHIC
    ]

    accuracies_losses = [
        [0.8705, 0.8765, 0.8824, 0.8433, 0.8924, 0.8765, 0.8074, 0.8734, 0.874, 0.8236, 0.8718, 0.8847],  # Pairloss Global
        [0.8553, 0.8548, 0.8564, 0.803, 0.8857, 0.8706, 0.8206, 0.8434, 0.856, 0.7951, 0.8285, 0.8525],  # Pairloss Local
        [0.8719, 0.8668, 0.8731, 0.7805, 0.8905, 0.8804, 0.8291, 0.8655, 0.8713, 0.7978, 0.8401, 0.8539],  # Triplet Loss
        [0.8682, 0.8788, 0.8921, 0.7533, 0.8847, 0.8557, 0.792, 0.8655, 0.874, 0.825, 0.8703, 0.8686],  # InfoNCE Loss
        [0.8802, 0.5398, 0.5405, 0.5469, 0.8818, 0.8735, 0.56, 0.8712, 0.5471, 0.8128, 0.8516, 0.866]   # SupCon Loss
    ]

    omegas_losses = [
        [0.5465, 0.5511, 0.5452, 0.4764, 0.552, 0.5612, 0.3778, 0.5069, 0.5131, 0.4238, 0.5171, 0.5634],  # Pairloss Global
        [0.4716, 0.4641, 0.4757, 0.2253, 0.3649, 0.3583, 0.3447, 0.3977, 0.4302, 0.3122, 0.3579, 0.4336],  # Pairloss Local
        [0.5152, 0.5341, 0.5327, 0.4005, 0.5931, 0.6102, 0.4008, 0.5025, 0.526, 0.3517, 0.4588, 0.5112],  # Triplet Loss
        [0.4716, 0.4994, 0.5178, 0.1721, 0.3239, 0.3012, 0.297, 0.4594, 0.4865, 0.3031, 0.3999, 0.394],  # InfoNCE Loss
        [0.5005, 0.4902, 0.4827, -0.1979, -0.2115, -0.221, -0.0532, -0.0267, 0.4733, -0.1093, -0.1567, -0.1642] # SupCon Loss
    ]

    accuracies_ablation = [
        [0.8705, 0.8765, 0.8824, 0.8433, 0.8924, 0.8765, 0.8074, 0.8734, 0.874, 0.8236, 0.8718, 0.8847],  # Baseline
        [0.5175, 0.5227, 0.5141, 0.5206, 0.5418, 0.5099, 0.5166, 0.5114, 0.5016, 0.502, 0.5418, 0.5013],  # No Hic
        [0.5065, 0.5088, 0.5072, 0.5131, 0.513, 0.5079, 0.5131, 0.5011, 0.511, 0.5034, 0.5058, 0.5013],  # No Overlap
        #[0.8829, 0.8881, 0.8782, 0.8912, 0.903, 0.9022, 0.8611, 0.8632, 0.8801, 0.8494, 0.9035, 0.8941],  # No Edge Weights
        [0.5065, 0.5032, 0.5002, 0.5103, 0.5139, 0.502, 0.516, 0.5028, 0.5104, 0.5047, 0.5058, 0.5],  # No GNN
        [0.8641, 0.8585, 0.8592, 0.7899, 0.8732, 0.8745, 0.8257, 0.8479, 0.8549, 0.7965, 0.8228, 0.8552],  # No Transformer
        [0.8687, 0.8765, 0.8759, 0.7955, 0.8876, 0.8794, 0.8057, 0.87, 0.8691, 0.8494, 0.8761, 0.8727],  # No Aux Loss
    ]

    omegas_ablation = [
        [0.5465, 0.5511, 0.5452, 0.4764, 0.552, 0.5612, 0.3778, 0.5069, 0.5131, 0.4238, 0.5171, 0.5634],  # Baseline
        [0.0108, 0.0084, 0.0091, 0.2033, 0.1653, 0.1928, 0.0133, 0.0098, 0.0089, 0.0476, 0.0273, 0.0663],  # No Hic
        [-0.0292, -0.0316, -0.0303, -0.0879, -0.0908, -0.0768, -0.043, -0.0517, -0.046, -0.0639, -0.0418, -0.0687],  # No Overlap
        #[0.5734, 0.5883, 0.57, 0.4273, 0.4725, 0.5508, 0.4787, 0.5122, 0.5321, 0.4474, 0.6372, 0.6129],  # No Edge Weights
        [-0.0443, -0.0505, -0.0493, -0.1414, -0.1369, -0.1184, -0.0531, -0.0769, -0.0694, -0.0945, -0.073, -0.1123],  # No GNN
        [0.4742, 0.4909, 0.486, 0.3082, 0.3767, 0.3733, 0.3675, 0.4239, 0.4464, 0.3014, 0.4088, 0.4622],  # No Transformer
        [0.5581, 0.5588, 0.5489, 0.3492, 0.4741, 0.5213, 0.38, 0.5066, 0.5277, 0.4617, 0.5303, 0.561],  # No Aux Loss
    ]

    # Method names
    #method_names = ["Spectral", "Louvain", "Label Prop.", "grapHIC"]
    #method_names = ["Pairloss Global", "Pairloss Local", "Triplet Loss", "InfoNCE Loss", "SupCon Loss"]
    method_names = ["Baseline", "No Hic", "No Overlap", "No GNN", "No Transformer", "No Aux Loss"]
    data_lists = accuracies_ablation

    # Group names
    group_names = ["chr10", "chr19", "chr15", "chr22"]
    
    # Create a single plot with all 4 chromosomes
    fig, ax = create_grouped_plot(
        data_lists, 
        method_names=method_names,
        group_names=group_names,
        title=" ",
        groups_to_plot=[0, 1, 2, 3],  # Include all chromosomes
        save_path="accuracy_ablation_.png"  # Save the figure
    )
    
    plt.show()