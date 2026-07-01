from .plaid_loader import load_plaid
from .whited_loader import load_whited


def load_dataset(dataset_name: str, n_files=None):
    key = dataset_name.lower().strip()
    if key == 'plaid':
        return load_plaid(n_files=n_files)
    if key == 'whited':
        return load_whited(n_files=n_files)
    raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'plaid', 'whited', or 'plaid+whited'.")


SUPPORTED_DATASETS = ['plaid', 'whited', 'plaid+whited']
