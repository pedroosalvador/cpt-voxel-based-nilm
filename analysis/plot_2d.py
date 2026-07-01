import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loaders.smart_meter_loader import load_smart_meter_dataset
from data_processing.cpt_decomposition import CPT

# Load first 10 files
data = load_smart_meter_dataset(n_files=10)
first = data[0]
fs = first.sampling_frequency

# 1) One figure with 1 row and 2 columns
plt.figure(figsize=(16, 4))

plt.subplot(1, 2, 1)
t = np.arange(0, len(first.current_segment)) / fs
plt.plot(t, first.voltage_segment, label='Voltage')
plt.plot(t, first.current_segment, label='Current')
plt.title('Voltage and Current (first sample)')
plt.xlabel('Time (s)')
plt.ylabel('Signal')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
ia0, ir0, iv0 = CPT(first)
if ia0 is None:
    ia0 = []
if ir0 is None:
    ir0 = []
if iv0 is None:
    iv0 = []

if len(ia0) > 0:
    t0 = np.arange(0, len(ia0)) / fs
    plt.plot(t0, ia0, label='I_active')
if len(ir0) > 0:
    t_ir = np.arange(0, len(ir0)) / fs
    plt.plot(t_ir, ir0, label='I_reactive')
if len(iv0) > 0:
    t_iv = np.arange(0, len(iv0)) / fs
    plt.plot(t_iv, iv0, label='I_void')

plt.title('CPT Components (first sample)')
plt.xlabel('Time (s)')
plt.ylabel('Current')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 2) One figure with 2 rows and 5 columns for voltage/current of first 10 samples
plt.figure(figsize=(16, 6))
for idx in range(0, 10):
    plt.subplot(2, 5, idx + 1)
    if idx < len(data):
        sample = data[idx]
        t_sample = np.arange(0, len(sample.current_segment)) / sample.sampling_frequency
        plt.plot(t_sample, sample.voltage_segment, label='Voltage', color='tab:blue')
        plt.plot(t_sample, sample.current_segment, label='Current', color='tab:orange')
        plt.title('Sample {}'.format(idx + 1))
        plt.grid(True)
        if idx == 0:
            plt.legend(fontsize='small')
    else:
        plt.axis('off')

plt.suptitle('Voltage and Current - first 10 samples')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 3) One figure with 2 rows and 5 columns for CPT components of first 10 samples
plt.figure(figsize=(16, 6))
for idx in range(0, 10):
    plt.subplot(2, 5, idx + 1)
    if idx < len(data):
        ia, ir, iv = CPT(data[idx])
        if ia is None:
            ia = []
        if ir is None:
            ir = []
        if iv is None:
            iv = []

        if len(ia) > 0:
            t_ia = np.arange(0, len(ia)) / fs
            plt.plot(t_ia, ia, color='tab:blue')
        if len(ir) > 0:
            t_ir = np.arange(0, len(ir)) / fs
            plt.plot(t_ir, ir, color='tab:orange')
        if len(iv) > 0:
            t_iv = np.arange(0, len(iv)) / fs
            plt.plot(t_iv, iv, color='tab:green')
        plt.title('Sample {}'.format(idx + 1))
        plt.grid(True)
    else:
        plt.axis('off')

plt.suptitle('CPT Components - first 10 samples')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()