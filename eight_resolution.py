import numpy as np
# ---------- load data ----------
x_path = './preprocessed_data/X_PLAID-WHITED_RES8.npy'
y_path = './preprocessed_data/y_PLAID-WHITED_RES8.npy'

X, y = np.load(x_path), np.load(y_path)

# add channel dimension if not present
if len(X.shape) == 4:
    X = np.expand_dims(X, axis=-1)

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique classes: {len(np.unique(y))}")