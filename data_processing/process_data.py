import sys, os
import numpy as np  
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loaders.plaid_loader import load_plaid
from loaders.whited_loader import load_whited

from data_processing.cpt_decomposition import CPT
from data_processing.voxelization import build_voxel_dataset
from data_processing.data_augmentation import augment_dataset
from data_processing.normalization import normalize

def process_data(x_path, y_path, augment=False, save=False, dataset=''):
    print(f"Processing {dataset.upper()} dataset")
    
    # load dataset 
    if dataset.lower() == 'plaid':
        samples = load_plaid()
    elif dataset.lower() == 'whited':
        samples = load_whited()

    tensors, labels = [], []
    total_samples = len(samples)
    
    print(f"\nProcessing {total_samples} samples...")

    with tqdm(total=total_samples, desc="Overall Progress", unit="sample", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        for sample in samples:
            pbar.set_description(f"Processing [{sample.label:30s}]")
            
            cpt_component = CPT(sample)
            normalized_sample = normalize(cpt_component, sample)    
            tensor = build_voxel_dataset(normalized_sample)

            tensors.append(tensor)
            labels.append(sample.label)
            
            pbar.update(1) # update overall progress bar

    # consolidate all tensors and labels
    print(f"\n{'='*60}")
    print("CONSOLIDATING TENSORS...")
    print(f"{'='*60}")
    
    X = np.concatenate(tensors, axis=0)
    y = np.array(labels)
    
    if augment:
        print("APPLYING DATA AUGMENTATION")
        
        X, y = augment_dataset(X, y)

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY - {dataset.upper()}")
    print(f"{'='*60}")
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
    print(f"{'='*60}")

    if save:
        print(f"\nSaving processed data...")
        np.save(x_path, X)
        np.save(y_path, y)
        print(f"Saved features to: {x_path}")
        print(f"Saved labels to: {y_path}")
        print(f"{'='*60}")

    return X, y 

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CHOOSE DATASET TO PROCESS:")
    print("="*60)

    print("\n1 - PLAID")
    print("2 - WHITED")
    print("3 - PLAID + WHITED")
    print("4 - Exit\n")

    user_choice = input("Process which dataset?: ").strip()

    match user_choice:
        case '1' :
            X, y = process_data(f'X_plaid.npy', f'y_plaid.npy', augment=True, save=True, dataset='plaid')

        case '2':
            X, y = process_data(f'X_whited.npy', f'y_whited.npy', augment=True, save=False, dataset='whited')

        case '3' :
            X_whited, y_whited = process_data('X_whited.npy', 'y_whited.npy', save=False, dataset='whited')
            X_plaid, y_plaid = process_data('X_plaid.npy', 'y_plaid.npy', save=False, dataset='plaid')

            X_combined = np.concatenate((X_whited, X_plaid), axis=0)
            y_combined = np.concatenate((y_whited, y_plaid), axis=0)

            # apply augmentation to complete dataset
            print(f"\n{'='*60}")
            print("APPLYING DATA AUGMENTATION...")
            print(f"{'='*60}\n")
    
            X, y = augment_dataset(X_combined, y_combined)

            np.save('X_PLAID-WHITED_RES8.npy', X)
            np.save('y_PLAID-WHITED_RES8.npy', y) 

        case '4':
            print("Exiting...")
            sys.exit(0)

        case _:
            print("Invalid choice.")   