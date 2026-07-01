import os
import numpy as np
import polars as pl
from tqdm import tqdm

from loaders import load_dataset
from data_processing.cpt_decomposition import CPT
from data_processing.data_augmentation import augment_dataset
from data_processing.normalization import normalize
from data_processing.voxelization import build_voxel_dataset
from data_processing.Data import Data


def _row_to_data_object(row):
    current_segment = np.asarray(row["current_segment"], dtype=np.float32)
    voltage_segment = np.asarray(row["voltage_segment"], dtype=np.float32)
    return Data(
        current_segment,
        voltage_segment,
        row["label"],
        int(row["sampling_frequency"]),
        int(row["f_mains"]),
    )


def _process_samples(samples: pl.DataFrame, dataset_name: str):
    tensors, labels = [], []
    total_samples = len(samples)

    print(f"\nProcessing {total_samples} samples from {dataset_name}...")

    with tqdm(
        total=total_samples,
        desc="Overall Progress",
        unit="sample",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar:
        for row in samples.iter_rows(named=True):
            data_obj = _row_to_data_object(row)
            cpt_component = CPT(data_obj)
            normalized_sample = normalize(cpt_component, data_obj)
            tensor = build_voxel_dataset(normalized_sample)
            tensors.append(tensor)
            labels.append(data_obj.label)
            pbar.update(1)

    if tensors:
        X = np.concatenate(tensors, axis=0)
    else:
        X = np.zeros((0, 8, 8, 8, 1), dtype=np.float32)
    y = np.array(labels)
    return X, y


def _summarize_labels(y, dataset_name: str):
    print(f"{'-'*60}")
    print(f"DATASET SUMMARY - {dataset_name.upper()}")
    print(f"{'-'*60}")
    unique_classes, class_counts = np.unique(y, return_counts=True)

    for class_name, count in zip(unique_classes, class_counts):
        percentage = (count / len(y)) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"{class_name:30s}: {count:4d} samples {bar} {percentage:5.1f}%")

    print(f"\n{'─'*60}")
    print(f"Total samples: {len(y)}")
    print(f"Number of classes: {len(unique_classes)}")
    print(f"Final dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"\n{'─'*60}")


def process_data(
    dataset: str,
    x_path: str,
    y_path: str,
    augment: bool = False,
    save: bool = False,
    n_files: int | None = None,
):
    dataset_key = dataset.lower().strip()

    if dataset_key == 'plaid+whited':
        plaid_samples = load_dataset('plaid', n_files=n_files)
        whited_samples = load_dataset('whited', n_files=n_files)

        X_plaid, y_plaid = _process_samples(plaid_samples, 'PLAID')
        X_whited, y_whited = _process_samples(whited_samples, 'WHITED')

        X = np.concatenate((X_plaid, X_whited), axis=0)
        y = np.concatenate((y_plaid, y_whited), axis=0)
    else:
        samples = load_dataset(dataset_key, n_files=n_files)
        X, y = _process_samples(samples, dataset_key)

    if augment:
        print(f"{'-'*60}")
        print("Applying data augmentation")
        X, y = augment_dataset(X, y)
        print(f"{'-'*60}")

    _summarize_labels(y, dataset_key)

    if save:
        os.makedirs(os.path.dirname(x_path) or '.', exist_ok=True)
        print(f"\nSaving processed data...")
        np.save(x_path, X)
        np.save(y_path, y)
        print(f"Saved features to: {x_path}")
        print(f"Saved labels to: {y_path}")
        print(f"{'='*60}")

    return X, y


if __name__ == "__main__":
    print("This file is intended to be used as a module from main.py.")
