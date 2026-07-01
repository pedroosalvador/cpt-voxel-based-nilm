import numpy as np
from scipy.ndimage import gaussian_filter

# parameters
RESOLUTION = 8
SMOOTH = True

def _normalize_to_minus_one_one(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_min == arr_max:
        return np.zeros_like(arr)
    return 2.0 * (arr - arr_min) / (arr_max - arr_min) - 1.0


def _voxelize_3d_grid(x, y, z, resolution=RESOLUTION):
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    scale = (resolution - 1) * 0.5

    x_idx = np.clip(((x + 1.0) * scale).astype(np.int32), 0, resolution - 1)
    y_idx = np.clip(((y + 1.0) * scale).astype(np.int32), 0, resolution - 1)
    z_idx = np.clip(((z + 1.0) * scale).astype(np.int32), 0, resolution - 1)

    np.add.at(voxel_grid, (x_idx, y_idx, z_idx), 1.0)

    max_density = voxel_grid.max()
    if max_density > 0:
        voxel_grid *= (1.0 / max_density)

    if SMOOTH:
        voxel_grid = gaussian_filter(voxel_grid, sigma=1.0)
        max_val = voxel_grid.max()
        if max_val > 0:
            voxel_grid *= (1.0 / max_val)

    return voxel_grid

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

def voxelize_3d_signal(voltage, current, time_axis, resolution=RESOLUTION):
    voltage_norm = _normalize_to_minus_one_one(voltage)
    current_norm = _normalize_to_minus_one_one(current)
    time_norm = _normalize_to_minus_one_one(time_axis)
    return _voxelize_3d_grid(voltage_norm, current_norm, time_norm, resolution=resolution)


def build_voxel_dataset(normalized_current, voltage_segment=None, sampling_frequency=None, use_time_axis=False, resolution=RESOLUTION):
    if use_time_axis:
        if voltage_segment is None or sampling_frequency is None:
            raise ValueError('voltage_segment and sampling_frequency are required when use_time_axis=True')

        n_points = len(normalized_current.i_active)
        if len(voltage_segment) != n_points:
            voltage_segment = np.asarray(voltage_segment, dtype=np.float32)
            if len(voltage_segment) < n_points:
                raise ValueError('Voltage segment is shorter than CPT component length for time-based voxelization')
            voltage_segment = voltage_segment[-n_points:]

        time_axis = np.arange(n_points, dtype=np.float32) / float(sampling_frequency)

        volumes = []
        for current_component in (normalized_current.i_active, normalized_current.i_reactive, normalized_current.i_void):
            volume = voxelize_3d_signal(voltage_segment, current_component, time_axis, resolution=resolution)
            volumes.append(volume)

        voxel_grid = np.stack(volumes, axis=-1)
        voxel_grid = np.expand_dims(voxel_grid, axis=0)
        return voxel_grid

    voxel_grid = voxelize_3d_trajectory(
        normalized_current.i_active, 
        normalized_current.i_reactive, 
        normalized_current.i_void,
    )
    
    voxel_grid = np.expand_dims(voxel_grid, axis=-1)
    voxel_tensor = np.expand_dims(voxel_grid, axis=0)
    return voxel_tensor