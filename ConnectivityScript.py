#!/usr/bin/env python3
# Functional Connectivity Analysis Script
# This script performs analysis of fMRI data across three conditions:
# - Task Graz
# - NeurowMI
# - Rest

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from nilearn import input_data, plotting

# Create directories for outputs if they don't exist
os.makedirs('Images', exist_ok=True)
os.makedirs('Matrices', exist_ok=True)

# ==============================
# 1. CONFIGURATION
# ==============================

# Define paths (adjust these to your data location)
project_path = 'Project4'
atlas_path = 'Schaefer_100parcels_7Networks/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
lut_path = 'Schaefer_100parcels_7Networks/Schaefer2018_100Parcels_7Networks_order.lut'

# Define the network prefixes for both hemispheres
network_prefixes = [
    '7Networks_LH_Vis', '7Networks_LH_SomMot', '7Networks_LH_DorsAttn', 
    '7Networks_LH_SalVentAttn', '7Networks_LH_Limbic', '7Networks_LH_Cont', 
    '7Networks_LH_Default', '7Networks_RH_Vis', '7Networks_RH_SomMot', 
    '7Networks_RH_DorsAttn', '7Networks_RH_SalVentAttn', '7Networks_RH_Limbic', 
    '7Networks_RH_Cont', '7Networks_RH_Default'
]

# Identify subjects
subjects = ['sub-12', 'sub-13', 'sub-14', 'sub-16', 'sub-17']

# Condition names for outputs
condition_names = ['Task_Graz', 'NeurowMI', 'Rest']

# ==============================
# 2. HELPER FUNCTIONS
# ==============================

def load_atlas():
    """Load the atlas image and region names."""
    print("Loading atlas...")
    atlas_img = nib.load(atlas_path)
    region_names = load_region_names(lut_path)
    return atlas_img, region_names

def load_region_names(lut_path):
    """Load region names from the LUT file."""
    try:
        # Load LUT file assuming space-separated values
        lut_df = pd.read_csv(lut_path, sep=r'\s+', header=None, names=['ID', 'R', 'G', 'B', 'Region'])
        
        # Ensure we have at least 100 regions
        if len(lut_df) < 100:
            raise ValueError(f"LUT file contains only {len(lut_df)} regions, expected 100.")

        return lut_df['Region'].tolist()  # Extract and return region names
    
    except Exception as e:
        print(f"Error loading LUT file: {e}")
        return [f"Region_{i+1}" for i in range(100)]  # Fallback default names

def join_files_by_condition(project_path):
    """Categorizes files in the project directory into task_graz, neurowMI, and rest conditions."""
    task_graz_files = []
    neurowMI_files = []
    rest_files = []

    print("Categorizing files by condition...")
    for folder in os.listdir(project_path):
        folder_path = os.path.join(project_path, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file_path.endswith(".nii.gz"):  # Only include NIFTI files
                    if "graz" in file_path:
                        task_graz_files.append(file_path)
                    elif "neurowMI" in file_path:
                        neurowMI_files.append(file_path)
                    elif "rest" in file_path:
                        rest_files.append(file_path)

    print(f"Found {len(task_graz_files)} Task Graz files, {len(neurowMI_files)} NeurowMI files, and {len(rest_files)} Rest files")
    return [task_graz_files, neurowMI_files, rest_files]

def extract_time_series(fmri_img, atlas_img):
    """Extracts time series from fMRI data using a given atlas."""
    # Create a masker to extract time series from each region
    masker = input_data.NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=True,  # Z-score normalization
        memory='nilearn_cache',  # Cache results
        verbose=0
    )

    # Extract time series
    time_series = masker.fit_transform(fmri_img)
    return time_series

def create_network_indices(region_names, network_prefixes):
    """Create a dictionary mapping each network prefix to region indices."""
    network_indices = {prefix: [] for prefix in network_prefixes}

    # Populate the dictionary with indices
    for idx, region in enumerate(region_names):
        for prefix in network_prefixes:
            if region.startswith(prefix):
                network_indices[prefix].append(idx)
                break
                
    return network_indices
def plot_correlation_matrix(correlation_matrix, region_names, title, output_path):
    """Plot and save a full correlation matrix with network boundary markers."""
    # Find network boundaries
    network_boundaries = []
    current_network = None
    
    for i, region in enumerate(region_names):
        # Extract network prefix (e.g., "7Networks_LH_Vis")
        parts = region.split('_')
        if len(parts) >= 3:
            network = f"{parts[0]}_{parts[1]}_{parts[2]}"
            
            if network != current_network:
                network_boundaries.append(i)
                current_network = network
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    ax = sns.heatmap(
        correlation_matrix,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=[],
        yticklabels=[]
    )
    
    # Add network boundary lines
    for boundary in network_boundaries:
        ax.axhline(y=boundary, color='black', linewidth=0.5)
        ax.axvline(x=boundary, color='black', linewidth=0.5)
    
    # Add network labels at the midpoint of each network section
    prev_boundary = 0
    for i, boundary in enumerate(network_boundaries[1:] + [len(region_names)]):
        # Skip the first boundary since it's the start
        if i == 0 and network_boundaries[0] == 0:
            prev_boundary = network_boundaries[0]
            continue
            
        # Calculate midpoint
        midpoint = (prev_boundary + boundary) // 2
        
        # Get network name from the region at the midpoint
        if midpoint < len(region_names):
            parts = region_names[midpoint].split('_')
            if len(parts) >= 3:
                network_label = f"{parts[1]}_{parts[2]}"  # e.g., "LH_Vis"
                
                # Add labels
                ax.text(-5, midpoint, network_label, rotation=90, 
                        va='center', ha='right', fontsize=8)
                ax.text(midpoint, -5, network_label, 
                        va='top', ha='center', rotation=45, fontsize=8)
        
        prev_boundary = boundary
    
    plt.title(f'Functional Connectivity: {title}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return correlation_matrix

def calculate_network_correlation_matrix(correlation_matrix, network_indices, network_prefixes):
    """Calculate the average correlation matrix between networks."""
    # Initialize an empty matrix to hold the average correlation values
    network_correlation_matrix = np.zeros((len(network_prefixes), len(network_prefixes)))

    # Calculate the average correlation for each network pair
    for i, prefix_i in enumerate(network_prefixes):
        for j, prefix_j in enumerate(network_prefixes):
            indices_i = network_indices[prefix_i]
            indices_j = network_indices[prefix_j]
            
            # Extract the submatrix for the current network pair
            submatrix = correlation_matrix[np.ix_(indices_i, indices_j)]

            # Calculate the average value of the submatrix
            average_value = np.mean(submatrix)

            # Assign the average value to the corresponding entry in the new matrix
            network_correlation_matrix[i, j] = average_value

    return network_correlation_matrix

def plot_network_correlation_heatmap(network_correlation_matrix, network_prefixes, title, output_path):
    """Plot and save a network-level correlation heatmap."""
    # Create more readable labels
    labels = [prefix.split('_')[-2:] for prefix in network_prefixes]
    labels = [f"{l[0]}-{l[1]}" for l in labels]
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        network_correlation_matrix,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title(f'Network Connectivity: {title}', fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# ==============================
# 3. MAIN ANALYSIS FUNCTION
# ==============================

def process_conditions(condition_files, atlas_img, region_names, network_indices, subjects):
    """Process all conditions and calculate correlation matrices."""
    
    all_condition_matrices = []
    all_network_matrices = []
    
    # Process each condition (Task Graz, NeurowMI, Rest)
    for condition_idx, condition_paths in enumerate(condition_files):
        condition_name = condition_names[condition_idx]
        print(f"\nProcessing condition: {condition_name}")
        
        # Initialize matrices for this condition (across all subjects)
        all_subject_matrices = []
        
        # Process each subject
        for subject in subjects:
            print(f"  Processing subject: {subject}")
            subject_files = [file for file in condition_paths if subject in file]
            
            if not subject_files:
                print(f"  No files found for {subject} in {condition_name} condition. Skipping.")
                continue
                
            # Initialize matrices for this subject (across all runs)
            subject_correlation_matrices = []
            
            # Process each file (run) for this subject
            for file_path in tqdm(subject_files, desc=f"  Processing {subject} {condition_name} files"):
                try:
                    # Load the fMRI image
                    fmri_img = nib.load(file_path)
                    
                    # Extract time series
                    time_series = extract_time_series(fmri_img, atlas_img)
                    
                    # Calculate correlation matrix
                    correlation_matrix = np.corrcoef(time_series.T)
                    
                    # Add to subject matrices
                    subject_correlation_matrices.append(correlation_matrix)
                    
                except Exception as e:
                    print(f"  Error processing file {file_path}: {e}")
                    continue
            
            # Calculate average correlation matrix for this subject (if any runs were processed)
            if subject_correlation_matrices:
                subject_avg_matrix = np.mean(subject_correlation_matrices, axis=0)
                all_subject_matrices.append(subject_avg_matrix)
                print(f"  Finished processing {subject} ({len(subject_correlation_matrices)} runs)")
            else:
                print(f"  No valid data for {subject}. Skipping.")
        
        # Calculate average correlation matrix across all subjects for this condition
        if all_subject_matrices:
            condition_avg_matrix = np.mean(all_subject_matrices, axis=0)
            
            # Plot and save the granular correlation matrix
            plot_path = f'Images/correlation_matrix_{condition_name}.png'
            plot_correlation_matrix(condition_avg_matrix, region_names, condition_name, plot_path)
            
            # Calculate, plot and save the network correlation matrix
            network_matrix = calculate_network_correlation_matrix(condition_avg_matrix, network_indices, network_prefixes)
            network_plot_path = f'Images/network_correlation_matrix_{condition_name}.png'
            plot_network_correlation_heatmap(network_matrix, network_prefixes, condition_name, network_plot_path)
            
            # Save the matrices as numpy arrays
            np.save(f'Matrices/correlation_matrix_{condition_name}.npy', condition_avg_matrix)
            np.save(f'Matrices/network_correlation_matrix_{condition_name}.npy', network_matrix)
            
            # Add to all conditions lists
            all_condition_matrices.append(condition_avg_matrix)
            all_network_matrices.append(network_matrix)
            
            print(f"Finished processing condition: {condition_name}")
        else:
            print(f"No valid data for condition: {condition_name}. Skipping.")
    
    return all_condition_matrices, all_network_matrices

# ==============================
# 4. STATISTICAL COMPARISON
# ==============================

def compare_conditions(all_condition_matrices, all_network_matrices):
    """Compare the matrices between conditions."""
    condition_pairs = [
        (0, 1, "Task_Graz vs NeurowMI"),
        (0, 2, "Task_Graz vs Rest"),
        (1, 2, "NeurowMI vs Rest")
    ]
    
    print("\nStatistical Comparison of Conditions:")
    
    # Compare full matrices
    for i, j, label in condition_pairs:
        if i < len(all_condition_matrices) and j < len(all_condition_matrices):
            # Calculate Frobenius norm of difference
            diff = np.linalg.norm(all_condition_matrices[i] - all_condition_matrices[j], 'fro')
            print(f"  {label} (Full Matrix) - Difference: {diff:.4f}")
            
            # Calculate element-wise absolute differences and get statistics
            abs_diff = (all_condition_matrices[i] - all_condition_matrices[j])
            print(f"    Mean absolute difference: {np.mean(abs_diff):.4f}")
            print(f"    Max absolute difference: {np.max(abs_diff):.4f}")
            
            # Plot difference matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(abs_diff, cmap='viridis')
            plt.title(f'Absolute Difference: {label}')
            plt.tight_layout()
            plt.savefig(f'Images/difference_matrix_{condition_names[i]}_vs_{condition_names[j]}.png', dpi=300)
            plt.close()
    
    # Compare network matrices
    print("\nNetwork-Level Comparisons:")
    for i, j, label in condition_pairs:
        if i < len(all_network_matrices) and j < len(all_network_matrices):
            # Calculate Frobenius norm of difference
            diff = np.linalg.norm(all_network_matrices[i] - all_network_matrices[j], 'fro')
            print(f"  {label} (Network Matrix) - Difference: {diff:.4f}")
            
            # Plot difference matrix
            abs_diff = (all_network_matrices[i] - all_network_matrices[j])
            plt.figure(figsize=(10, 8))
            sns.heatmap(abs_diff, cmap='viridis', annot=True, fmt='.2f')
            plt.title(f'Network Differences: {label}')
            plt.tight_layout()
            plt.savefig(f'Images/network_difference_{condition_names[i]}_vs_{condition_names[j]}.png', dpi=300)
            plt.close()

# ==============================
# 5. MAIN EXECUTION
# ==============================

def main():
    """Main execution function."""
    print("Starting functional connectivity analysis...")
    
    # Load atlas and region names
    atlas_img, region_names = load_atlas()
    
    # Create the network indices dictionary
    network_indices = create_network_indices(region_names, network_prefixes)
    
    # Plot the atlas for reference
    print("Generating atlas visualization...")
    display = plotting.plot_roi(atlas_img, title="Schaefer 100-Parcel Atlas")
    plt.savefig('Images/atlas_visualization.png')
    plt.close()
    
    # Categorize files by condition
    condition_files = join_files_by_condition(project_path)
    
    # Process all conditions
    all_condition_matrices, all_network_matrices = process_conditions(
        condition_files, atlas_img, region_names, network_indices, subjects
    )
    
    # Compare conditions
    if len(all_condition_matrices) > 1:
        compare_conditions(all_condition_matrices, all_network_matrices)
    
    # Save all matrices in one file for later use
    np.save('Matrices/all_condition_matrices.npy', all_condition_matrices)
    np.save('Matrices/all_network_matrices.npy', all_network_matrices)
    
    print("\nAnalysis complete! Results saved in the Images/ and Matrices/ directories.")

if __name__ == "__main__":
    main()