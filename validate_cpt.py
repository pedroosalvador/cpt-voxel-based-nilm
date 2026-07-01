import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.signal import savgol_filter

from loaders.plaid_loader import load_plaid
from loaders.smart_meter_loader import load_smart_meter_dataset
from data_processing.cpt_decomposition import CPT

dataset = load_smart_meter_dataset()
data_obj = dataset[0]
print(f"Sample: {data_obj}\n")

i_active, i_reactive, i_void = CPT(data_obj)

# parameters
fs = data_obj.sampling_frequency
f_mains = 60.0
samples_per_cycle = int(fs / f_mains)
n_cycles_total = len(i_active) // samples_per_cycle

print(f"Samples per cycle: {samples_per_cycle}")
print(f"Total cycles in signal: {n_cycles_total}\n")

# rely on CPT internal transient removal
v = data_obj.voltage_segment

# select 10 cycles
n_cycles = min(10, len(i_active) // samples_per_cycle)
n_samples = n_cycles * samples_per_cycle

v = v[:n_samples]
i_active = i_active[:n_samples]
i_reactive = i_reactive[:n_samples]
i_void = i_void[:n_samples]

print(f"Using {n_cycles} cycles\n")

# reshape into cycles and average
def reshape_and_average(signal):
    cycles = signal.reshape(n_cycles, samples_per_cycle)
    mean_cycle = np.mean(cycles, axis=0)
    smoothed = savgol_filter(mean_cycle, 21, 3)
    return smoothed

v_avg = reshape_and_average(v)
i_active_avg = reshape_and_average(i_active)
i_reactive_avg = reshape_and_average(i_reactive)
i_void_avg = reshape_and_average(i_void)

# normalize
def normalize(x):
    x = x - np.mean(x)
    m = np.max(np.abs(x))
    return x / m if m > 1e-12 else x

v_norm = normalize(v_avg)
i_active_norm = normalize(i_active_avg)
i_reactive_norm = normalize(i_reactive_avg)
i_void_norm = normalize(i_void_avg)

# replicate to full 10 cycles
v_full = np.tile(v_norm, n_cycles)
i_active_full = np.tile(i_active_norm, n_cycles)
i_reactive_full = np.tile(i_reactive_norm, n_cycles)
i_void_full = np.tile(i_void_norm, n_cycles)

# time axis in cycles
t = np.linspace(0, n_cycles, len(v_full))

# create single 3D plot with all components
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# define component styles
components = {
    'Ia': {'data': i_active_full, 'label': 'ativa',      'color': 'blue'},
    'Ir': {'data': i_reactive_full, 'label': 'reativa',  'color': 'red'},
    'Iv': {'data': i_void_full, 'label': 'residual',         'color': 'green'},
}

# plot each component
for key, config in components.items():
    ax.plot(v_full, t, config['data'], 
            linewidth=2.5, 
            color=config['color'],
            label=f"Corrente {config['label']}")

ax.set_xlabel('Tensão (V)', fontsize=12)
ax.set_ylabel('Tempo (ciclos)', fontsize=12)
ax.set_zlabel('Corrente (I)', fontsize=12)
ax.set_title(f'Componentes CPT - {data_obj.label}', fontsize=14, pad=20)
ax.legend(fontsize=11, loc='upper left')

ax.view_init(elev=25, azim=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

