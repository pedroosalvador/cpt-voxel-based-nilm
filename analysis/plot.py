import os, sys
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.process_data import process_data
from data_processing.cpt_decomposition import CPT  

from loaders.whited_loader import load_whited
from loaders.plaid_loader import load_plaid

# file paths
x_path = './preprocessed_data/X_PLAID-WHITED_R8_AUG.npy'
y_path = './preprocessed_data/y_PLAID-WHITED_R8_AUG.npy' 

dados = load_plaid() 
data = dados[1]  # Select the i-th file

dt = data.get_time_step()
points_per_cycle = data.get_points_per_cycle() 
DURATION = len(data.current_segment) / data.sampling_frequency  # duration in seconds

cpt_result = CPT(data)
i_active, i_reactive, i_void = cpt_result
cpt = SimpleNamespace(i_active=i_active, i_reactive=i_reactive, i_void=i_void)

t_clean = np.arange(len(i_active)) * dt
t = np.arange(len(data.current_segment)) * dt

# PLOTS 2D
def plot_2d(i_active, i_reactive, i_void, data, t_clean, t, DURATION):
    fig = plt.figure(figsize=(15, 10))
    
    # Título geral no topo da figura
    fig.suptitle(
        f"{data.label} - {data.sampling_frequency/1000:.0f} kHz, 60 Hz, {DURATION:.2f}s",
        fontsize=16,
        fontweight='bold'
    )

    plots = [
        (t_clean, i_active, 'Time [s]', 'Active Current [Ia]'),
        (t_clean, i_reactive, 'Time [s]', 'Reactive Current [Ir]'),
        (t_clean, i_void, 'Time [s]', 'Void Current [Iv]'),
        (t, data.current_segment, 'Time [s]', 'Total Current [It]'),
    ]

    for idx, (x, y, xlabel, ylabel) in enumerate(plots, start=1):
        ax = fig.add_subplot(2, 2, idx) 
        ax.plot(x, y, color='blue')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)  # título simplificado (só o nome da corrente)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # ajustar para não sobrepor suptitle
    return fig, ax

# PLOT 3D
def plot_3d(cpt):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(cpt.i_active, cpt.i_reactive, cpt.i_void)

    ax.set_xlabel('Active [A]')
    ax.set_ylabel('Reactive [A]')
    ax.set_zlabel('Void [A]')
    ax.set_title(f"{data.label}\n")

    return fig, ax

# LOAD DATA FOR VOXEL VISUALIZATION AND STATISTICS
X, y = np.load(x_path), np.load(y_path)

def visualize_voxel_3d(voxel_grid):
    # Remove channel dimension if present
    if voxel_grid.ndim == 4:
        voxel_grid = voxel_grid.squeeze()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get coordinates of non-zero voxels above threshold
    filled = voxel_grid > 0.01
    x, y, z = np.where(filled)
    
    # Get density values for coloring
    colors = voxel_grid[filled]
    
    # Normalize colors for colormap
    if colors.max() > 0:
        colors_normalized = colors / colors.max()
    else:
        colors_normalized = colors
    
    # Create scatter plot with density-based coloring
    scatter = ax.scatter(x, y, z, c=colors_normalized, 
                        cmap='viridis', marker='o', 
                        s=20, alpha=0.6, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Normalized Density', rotation=270, labelpad=15)
    
    # Labels
    ax.set_xlabel('Ia', fontsize=10)
    ax.set_ylabel('Ir', fontsize=10)
    ax.set_zlabel('Iv', fontsize=10)
    ax.set_title("Voxel Visualization", fontsize=12, pad=20)
    
    # Set equal aspect ratio
    resolution = voxel_grid.shape[0]
    ax.set_xlim([0, resolution])
    ax.set_ylim([0, resolution])
    ax.set_zlim([0, resolution])
    
    plt.tight_layout()
    return fig, ax

def visualize_voxel_slices(voxel_grid):
    n_slices=8

    # Remove channel dimension if present
    if voxel_grid.ndim == 4:
        voxel_grid = voxel_grid.squeeze()
    
    resolution = voxel_grid.shape[0]
    slice_indices = np.linspace(0, resolution-1, n_slices, dtype=int)
    
    fig, axes = plt.subplots(3, n_slices, figsize=(16, 6))
    fig.suptitle("Voxel Slices", fontsize=14)
    
    # Slices along Ia axis (YZ plane)
    for i, idx in enumerate(slice_indices):
        axes[0, i].imshow(voxel_grid[idx, :, :], cmap='hot', aspect='auto')
        axes[0, i].set_title(f'Ia={idx}', fontsize=8)
        axes[0, i].axis('off')
    
    # Slices along Ir axis (XZ plane)
    for i, idx in enumerate(slice_indices):
        axes[1, i].imshow(voxel_grid[:, idx, :], cmap='hot', aspect='auto')
        axes[1, i].set_title(f'Ir={idx}', fontsize=8)
        axes[1, i].axis('off')
    
    # Slices along Iv axis (XY plane)
    for i, idx in enumerate(slice_indices):
        axes[2, i].imshow(voxel_grid[:, :, idx], cmap='hot', aspect='auto')
        axes[2, i].set_title(f'Iv={idx}', fontsize=8)
        axes[2, i].axis('off')
    
    # Add axis labels
    axes[0, 0].set_ylabel('Slices\nalong Ia', fontsize=10, rotation=0, labelpad=40)
    axes[1, 0].set_ylabel('Slices\nalong Ir', fontsize=10, rotation=0, labelpad=40)
    axes[2, 0].set_ylabel('Slices\nalong Iv', fontsize=10, rotation=0, labelpad=40)
    
    plt.tight_layout()
    return fig

def visualize_multiple_samples(voxel_dataset, labels, indices=None):
    if indices is None:
        indices = list(range(min(6, len(voxel_dataset))))
    
    n_samples = len(indices)
    cols = 3
    rows = (n_samples + cols - 1) // cols
    
    fig = plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        
        voxel = voxel_dataset[idx].squeeze()
        label = labels[idx]
        
        # Get non-zero voxels
        filled = voxel > 0.01
        x, y, z = np.where(filled)
        colors = voxel[filled]
        
        if len(x) > 0 and colors.max() > 0:
            colors_normalized = colors / colors.max()
            ax.scatter(x, y, z, c=colors_normalized, cmap='viridis',
                      marker='o', s=10, alpha=0.6, edgecolors='none')
        
        ax.set_xlabel('Ia', fontsize=8)
        ax.set_ylabel('Ir', fontsize=8)
        ax.set_zlabel('Iv', fontsize=8)
        ax.set_title(f'Sample {idx} - Label: {label}', fontsize=10)
        
        resolution = voxel.shape[0]
        ax.set_xlim([0, resolution])
        ax.set_ylim([0, resolution])
        ax.set_zlim([0, resolution])
    
    plt.tight_layout()
    return fig

def compare_voxel_statistics(voxel_dataset):
    print("=" * 60)
    print("VOXEL DATASET STATISTICS")
    print("=" * 60)
    
    # Overall statistics
    print(f"\nDataset shape: {voxel_dataset.shape}")
    print(f"Number of samples: {len(voxel_dataset)}")
    print(f"Voxel resolution: {voxel_dataset.shape[1]}³")
    print(f"Total voxels per sample: {voxel_dataset.shape[1]**3:,}")
    
    # Sparsity analysis
    sparsities = []
    occupied_voxels = []
    max_densities = []
    mean_densities = []
    
    for i in range(len(voxel_dataset)):
        voxel = voxel_dataset[i].squeeze()
        non_zero = np.count_nonzero(voxel)
        total = voxel.size
        sparsity = 1 - (non_zero / total)
        
        sparsities.append(sparsity)
        occupied_voxels.append(non_zero)
        max_densities.append(voxel.max())
        mean_densities.append(voxel[voxel > 0].mean() if non_zero > 0 else 0)
    
    print(f"\n--- SPARSITY ---")
    print(f"Average sparsity: {np.mean(sparsities)*100:.2f}%")
    print(f"Min sparsity: {np.min(sparsities)*100:.2f}%")
    print(f"Max sparsity: {np.max(sparsities)*100:.2f}%")
    
    print(f"\n--- OCCUPIED VOXELS ---")
    print(f"Average occupied voxels: {np.mean(occupied_voxels):.0f}")
    print(f"Min occupied voxels: {np.min(occupied_voxels)}")
    print(f"Max occupied voxels: {np.max(occupied_voxels)}")
    
    print(f"\n--- DENSITY VALUES ---")
    print(f"Average max density: {np.mean(max_densities):.4f}")
    print(f"Average mean density (non-zero): {np.mean(mean_densities):.4f}")
    
    
    print("=" * 60) 

# visualize plot 2d
fig, ax = plot_2d(i_active, i_reactive, i_void, data, t_clean, t, DURATION)
plt.tight_layout()
plt.show()  

# visualize plot 3d
fig, ax = plot_3d(cpt)
plt.tight_layout()
plt.show()  

# visualize voxel 3d 
fig, ax = visualize_voxel_3d(X[2])
plt.tight_layout()
plt.show()

# visualize voxel slices
fig = visualize_voxel_slices(X[2])
plt.tight_layout()  
plt.show()

# visualize dataset statistics
fig = visualize_multiple_samples(X, y, indices=[301, 500, 700])
plt.tight_layout()
plt.show() 

compare_voxel_statistics(X)  