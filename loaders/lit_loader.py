import os
import scipy.io
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Mapping from ID to appliance name (Synthetic Subset - Table 2.1)
APPLIANCE_LABELS = {
    "1A0": "Microwave",
    "1B0": "LED Lamp",
    "1C0": "CRT Monitor",
    "1D0": "LED Panel",
    "1E0": "Fume Extrator",
    "1F0": "LED Monitor",
    "1G0": "Phone Charger Asus",
    "1H0": "Soldering Station",
    "1I0": "Phone Charger Motorola",
    "1J0": "Laptop Lenovo",
    "1K0": "Fan",
    "1L0": "Resistor",
    "1M0": "Laptop Vaio",
    "1N0": "Incandescent Lamp",
    "1O0": "Drill Speed 1",
    "1P0": "Drill Speed 2",
    "1Q0": "Oil Heater Power 1",
    "1R0": "Oil Heater Power 2",
    "1S0": "Microwave On",
    "1T0": "Air Heater Nilko",
    "1U0": "HairDryer Eleganza - Fan1",
    "1V0": "HairDryer Eleganza - Fan2",
    "1W0": "HairDryer Super 4.0 - Fan1 - Heater1",
    "1X0": "HairDryer Super 4.0 - Fan1 - Heater2",
    "1Y0": "HairDryer Parlux - Fan1 - Heater1",
    "1Z0": "HairDryer Parlux - Fan2 - Heater1"
}

class LitData:
    def __init__(self, current_segment, voltage_segment,
                 appliance_type, sampling_frequency, duration):
        self.current_segment = current_segment
        self.voltage_segment = voltage_segment
        self.appliance_type = appliance_type
        self.sampling_frequency = sampling_frequency
        self.duration = duration

def process_file(file_path, appliance_id):
    mat = scipy.io.loadmat(file_path)
    
    # extract variables
    iShunt = mat['iShunt'].squeeze()
    vGrid = mat['vGrid'].squeeze()
    sps = int(mat['sps'][0][0])
    duration = float(mat['duration_t'][0][0])
    
    # set time parameters
    Ti = 2.5
    Tf = 2.8
    
    # calculate indices
    n_points = len(iShunt)
    start_index = int(Ti * sps)
    end_index = int(Tf * sps)
    end_index = min(end_index, n_points)
    
    # extract segments
    current_segment = iShunt[start_index:end_index]
    voltage_segment = vGrid[start_index:end_index]
    
    # get appliance name from mapping
    appliance_type = APPLIANCE_LABELS.get(appliance_id, appliance_id)
    
    return LitData(current_segment, voltage_segment, appliance_type, sps, duration)

def load_lit(max_workers=8, appliance_type_filter=None):
    print("------------------------------")
    print("Initiating LIT dataset loading...")
    
    folder_path = './datasets/LIT_Dataset/Matlab_Data/Synthetic/1'
    
    file_list = []
    
    # Recursively find all .mat files in single load folders (1A0, 1B0, etc)
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith('.mat'):
                # extract appliance ID from folder name (e.g., 1A0, 1B0)
                folder_name = os.path.basename(root)
                
                # Only process if folder matches pattern 1XY (single loads)
                if folder_name.startswith('1') and len(folder_name) == 3:
                    appliance_id = folder_name  # e.g., "1A0", "1B0"
                    
                    if appliance_type_filter is None or appliance_id == appliance_type_filter:
                        file_list.append((os.path.join(root, f), appliance_id))
    
    file_list = file_list[:6]  # <-- Limit after filling the list!
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resultados = list(executor.map(lambda x: process_file(x[0], x[1]), file_list))
    
    print(f"Loaded {len(resultados)} files from LIT dataset.")
    print("------------------------------")
    return resultados

def get_all_data():
    return load_lit(max_workers=8, appliance_type_filter=None)