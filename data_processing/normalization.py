import numpy as np
from data_processing.Data import Data

# parameters
RESOLUTION = 32
DENSITY_MODE = True
SMOOTH = True

class Currents(Data):
    def __init__(self, current_segment, voltage_segment, label, sampling_frequency, f_mains,
                 i_active, i_reactive, i_void):
        super().__init__(current_segment, voltage_segment, label, sampling_frequency, f_mains)
        
        # normalized current components
        self.i_active = i_active
        self.i_reactive = i_reactive
        self.i_void = i_void

def normalize(cpt, data): 
    # desempacotar tupla retornada por CPT
    i_a, i_r, i_v = cpt
    i_a = np.asarray(i_a)
    i_r = np.asarray(i_r)
    i_v = np.asarray(i_v)

    ia_min, ia_max = np.min(i_a), np.max(i_a)
    ir_min, ir_max = np.min(i_r), np.max(i_r)
    iv_min, iv_max = np.min(i_v), np.max(i_v)

    # handle case where all values are constant
    if ia_max == ia_min and ir_max == ir_min and iv_max == iv_min:
        norm_result = Currents(
            data.current_segment,
            data.voltage_segment,
            data.label,
            data.sampling_frequency,
            data.f_mains,
            np.zeros_like(i_a), 
            np.zeros_like(i_r), 
            np.zeros_like(i_v)
        )
        
        return norm_result

    # min-max normalization to [-1, 1]
    norm_ia = 2 * (i_a - ia_min) / (ia_max - ia_min) - 1
    norm_ir = 2 * (i_r - ir_min) / (ir_max - ir_min) - 1
    norm_iv = 2 * (i_v - iv_min) / (iv_max - iv_min) - 1

    norm_result = Currents(
        data.current_segment,
        data.voltage_segment,
        data.label,
        data.sampling_frequency,
        data.f_mains,
        np.asarray(norm_ia),
        np.asarray(norm_ir),
        np.asarray(norm_iv)
    )
    
    return norm_result