import os, json
import polars as pl
from concurrent.futures import ThreadPoolExecutor

FOLDER_PATH = './datasets/PLAID/submetered'
METADATA_PATH = './datasets/PLAID/metadata_submetered.json'

def process_file(file_path, metadata):
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    data = pl.read_csv(file_path, has_header=False, new_columns=["Current", "Voltage"])

    meta = metadata[file_id]
    label = (meta["appliance"]["type"]).upper()
    sampling_frequency = int(meta["header"]["sampling_frequency"].replace("Hz", ""))
    f_mains = 60

    current_segment = data["Current"].to_numpy().astype('float32').tolist()
    voltage_segment = data["Voltage"].to_numpy().astype('float32').tolist()

    return {
        "current_segment": current_segment,
        "voltage_segment": voltage_segment,
        "label": label,
        "sampling_frequency": sampling_frequency,
        "f_mains": f_mains,
        "source": "plaid",
        "file_path": file_path,
    }

def load_plaid(n_files=None):
    print(f"{'-'*60}")
    print("Loading PLAID dataset")

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    file_list = []
    for root, _, files in os.walk(FOLDER_PATH):
        for f in files:
            if f.endswith('.csv'):
                file_list.append(os.path.join(root, f))

    file_list = file_list[:n_files]

    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(lambda file_path: process_file(file_path, metadata), file_list))

    print(f"Loaded {len(records)} files from PLAID dataset.")
    return pl.DataFrame(records)