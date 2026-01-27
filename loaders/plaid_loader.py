import os, json
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from data_processing.Data import Data

FOLDER_PATH = './datasets/PLAID/submetered'
METADATA_PATH = './datasets/PLAID/metadata_submetered.json'

def process_file(file_path, metadata): # process a single file
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    data = pl.read_csv(file_path, has_header=False, new_columns=["Current", "Voltage"])

    meta = metadata[file_id]
    label = (meta["appliance"]["type"]).upper()
    sampling_frequency = int(meta["header"]["sampling_frequency"].replace("Hz", ""))
    f_mains = 60

    current_segment = data["Current"].to_numpy()
    voltage_segment = data["Voltage"].to_numpy()

    return Data(current_segment, voltage_segment, label, sampling_frequency, f_mains)

def load_plaid(): # load whole PLAID dataset
    print(f"{'-'*60}")
    print("Loading PLAID dataset")

    # load metadata
    with open(METADATA_PATH, "r") as f: 
        metadata = json.load(f)

    file_list = []

    # walk through dataset directory
    for root, _, files in os.walk(FOLDER_PATH): 
        for f in files:
            if f.endswith('.csv'):
                file_list.append(os.path.join(root, f))

    #file_list = file_list[:10]

    # parallel loading utilizing threads and 8 workers
    with ThreadPoolExecutor(max_workers=8) as executor: 
        results = list(executor.map(partial(process_file, metadata=metadata), file_list))

    print(f"Loaded {len(results)} files from PLAID dataset.")

    return results