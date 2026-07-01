import os
import numpy as np
import polars as pl
from concurrent.futures import ThreadPoolExecutor

FOLDER_PATH = './datasets/WHITED'

def parse_whited_filename(filename):
    name_without_ext = os.path.splitext(filename)[0]
    if '(' in name_without_ext:
        appliance_type = name_without_ext.split('(')[0].strip()
    else:
        appliance_type = name_without_ext.strip()
    return appliance_type if appliance_type else "UNKNOWN"

def process_file(file_path):
    import soundfile as sf

    data, samplerate = sf.read(file_path)
    f_mains = 50

    voltage_segment = data[:, 0].astype('float32').tolist()
    current_segment = data[:, 1].astype('float32').tolist()

    filename = os.path.basename(file_path)
    label = parse_whited_filename(filename).upper()

    return {
        "current_segment": current_segment,
        "voltage_segment": voltage_segment,
        "label": label,
        "sampling_frequency": int(samplerate),
        "f_mains": f_mains,
        "source": "whited",
        "file_path": file_path,
    }

def load_whited(n_files=None):
    print('-'*60)
    print("Loading WHITED dataset")

    file_list = []
    for root_dir, _, files in os.walk(FOLDER_PATH):
        for f in files:
            if f.endswith('.flac'):
                file_list.append(os.path.join(root_dir, f))

    file_list = file_list[:n_files]

    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(process_file, file_list))

    print(f"Loaded {len(records)} files from WHITED dataset.")
    return pl.DataFrame(records)
