import numpy as np
from data_processing.Data import Data

def normalize_component(arr):
    # min-max normalization to [-1, 1] range 
    arr = np.asarray(arr, dtype=np.float32)
    arr_min, arr_max = np.min(arr), np.max(arr)
    
    if arr_min == arr_max:
        return np.zeros_like(arr)
    
    return 2.0 * (arr - arr_min) / (arr_max - arr_min) - 1.0

def normalize(cpt_components, data):
    # normalizes CPT components and creates Currents container
    i_a, i_r, i_v = cpt_components
    
    norm_ia = normalize_component(i_a)
    norm_ir = normalize_component(i_r)
    norm_iv = normalize_component(i_v)
    
    return Currents(data, norm_ia, norm_ir, norm_iv)


class Currents(Data):    
    def __init__(self, base_data, i_active, i_reactive, i_void):
        super().__init__(
            base_data.current_segment,
            base_data.voltage_segment,
            base_data.label,
            base_data.sampling_frequency,
            base_data.f_mains
        )
        self.i_active = np.asarray(i_active, dtype=np.float32)
        self.i_reactive = np.asarray(i_reactive, dtype=np.float32)
        self.i_void = np.asarray(i_void, dtype=np.float32)
    
    def __repr__(self):
        base_repr = super().__repr__()
        return base_repr[:-1] + f", CPT={len(self.i_active)}pts)"