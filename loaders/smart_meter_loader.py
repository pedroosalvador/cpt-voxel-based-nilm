import os
import polars as pl
from data_processing.Data import Data

FOLDER_PATH = './datasets/smart_meter_dataset/refrigerator'
LABEL = 'REFRIGERATOR'
SAMPLING_FREQUENCY = 2000
F_MAINS = 60

def process_file(file_path, sampling_frequency=SAMPLING_FREQUENCY, f_mains=F_MAINS, label=LABEL):
    data = pl.read_csv(file_path)
    columns = [col.strip().lower() for col in data.columns]

    if 'voltage' not in columns or 'current' not in columns:
        data = pl.read_csv(file_path, has_header=False, new_columns=['voltage', 'current'])
        columns = ['voltage', 'current']

    data.columns = columns
    current_segment = data['current'].to_numpy()
    voltage_segment = data['voltage'].to_numpy()

    return Data(current_segment, voltage_segment, label, sampling_frequency, f_mains)


def load_smart_meter_dataset(folder_path=FOLDER_PATH, sampling_frequency=SAMPLING_FREQUENCY, f_mains=F_MAINS, label=LABEL, n_files=None):
    print(f"{'-'*60}")
    print("Loading smart meter dataset")

    file_list = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith('.csv'):
                file_list.append(os.path.join(root, filename))

    if n_files is not None:
        file_list = file_list[:n_files]

    if not file_list:
        raise FileNotFoundError(f'No CSV files found in {folder_path}')

    results = []
    for file_path in file_list:
        results.append(process_file(file_path, sampling_frequency=sampling_frequency, f_mains=f_mains, label=label))

    print(f"Loaded {len(results)} files from smart meter dataset.")
    return results
