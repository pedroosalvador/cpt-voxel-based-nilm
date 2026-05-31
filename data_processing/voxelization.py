import numpy as np
from scipy.ndimage import gaussian_filter

# parameters
RESOLUTION = 8
SMOOTH = True

def voxelize_3d_trajectory(ia, ir, iv):
    voxel_grid = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION), dtype=np.float32)

    # map from [-1, 1] to [0, RESOLUTION-1] using float32
    scale = (RESOLUTION - 1) * 0.5
    ia_indices = ((ia + 1.0) * scale).astype(np.int32)
    ir_indices = ((ir + 1.0) * scale).astype(np.int32)
    iv_indices = ((iv + 1.0) * scale).astype(np.int32)
    
    # clip to valid range
    ia_indices = np.clip(ia_indices, 0, RESOLUTION - 1)
    ir_indices = np.clip(ir_indices, 0, RESOLUTION - 1)
    iv_indices = np.clip(iv_indices, 0, RESOLUTION - 1)
    
    # fill voxel grid 
    for i in range(len(ia)):
        voxel_grid[ia_indices[i], ir_indices[i], iv_indices[i]] += 1.0
        
    # normalize density to [0, 1]
    max_density = voxel_grid.max()
    if max_density > 0:
        voxel_grid *= (1.0 / max_density)  # in-place multiplication
    
    # gaussian smoothing (can be replaced with simpler filter in embedded)
    if SMOOTH:
        voxel_grid = gaussian_filter(voxel_grid, sigma=1.0)
        max_val = voxel_grid.max()
        if max_val > 0:
            voxel_grid *= (1.0 / max_val)
    
    return voxel_grid

def build_voxel_dataset(normalized_current):
    voxel_grid = voxelize_3d_trajectory(
        normalized_current.i_active, 
        normalized_current.i_reactive, 
        normalized_current.i_void
    )
    
    # add channel dimension (ensures float32)
    voxel_grid = np.expand_dims(voxel_grid, axis=-1)
    
    # add batch dimension
    voxel_tensor = np.expand_dims(voxel_grid, axis=0)
    
    return voxel_tensor