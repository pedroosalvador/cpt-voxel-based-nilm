import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from FocalLoss import FocalLoss

# ---------- load data ----------
x_path = './preprocessed_data/X_PLAID-WHITED_R8_AUG.npy'
y_path = './preprocessed_data/y_PLAID-WHITED_R8_AUG.npy'

X, y = np.load(x_path), np.load(y_path)

BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = 'RESNET3D_PLAID-WHITED_R8_AUG_FL.keras'
VOXEL_RESOLUTION = 8
NUM_CLASSES = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# encode labels
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_true = le.transform(y_test)

# convert labels to categorical (one-hot encoding)
y_train_onehot = tf.keras.utils.to_categorical(y_train_int, NUM_CLASSES)
y_true_onehot = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)

# ---------- build model ----------
def residual_block_3d(x, filters, stride=1): 
    shortcut = x

    # first conv
    x = tf.keras.layers.Conv3D(
        filters,
        kernel_size=3,
        strides=stride,
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # second conv
    x = tf.keras.layers.Conv3D(
        filters,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer='he_normal'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # adjust shortcut when stride > 1 OR channels change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv3D(
            filters,
            kernel_size=1,
            strides=stride,
            padding='same',
            kernel_initializer='he_normal'
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # sum + activation
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x

def build_resnet3d(input_shape, num_classes): 
    inputs = tf.keras.Input(shape=input_shape)

    # initial block
    x = tf.keras.layers.Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # first convolutional block (16 filters, no downsample)
    x = residual_block_3d(x, 16, stride=1)
    x = residual_block_3d(x, 16, stride=1)

    # second convolutional block (32 filters, downsample to 4x4x4)
    x = residual_block_3d(x, 32, stride=2)
    x = residual_block_3d(x, 32, stride=1)

    # third convolutional block (64 filters, downsample to 2x2x2)
    x = residual_block_3d(x, 64, stride=2)
    x = residual_block_3d(x, 64, stride=1)

    # fourth convolutional block (128 filters, no downsample)
    x = residual_block_3d(x, 128, stride=1)
    x = residual_block_3d(x, 128, stride=1)

    # classification
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

# ------------------ BUILD AND TRAIN ------------------
model = build_resnet3d(
    input_shape=(VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, 1),
    num_classes=NUM_CLASSES
)

model.compile(
    optimizer='adam',
    loss=FocalLoss(gamma=2.0, alpha=0.25), # antes era 'categorical_crossentropy'
    metrics=['accuracy']
)

# ---------- callbacks ----------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7, # 5 para 7
    restore_best_weights=True, 
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ---------- train ----------
history = model.fit(
    X_train, y_train_onehot,
    validation_data=(X_test, y_true_onehot),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[early_stop, reduce_lr] 
)

model.save(MODEL_PATH)

# ------------------ EVALUATION ------------------
y_pred_prob = model.predict(X_test)
y_pred_int = np.argmax(y_pred_prob, axis=1)

test_loss, test_acc = model.evaluate(X_test, y_true_onehot, verbose=0)
print(f"\nAccuracy: {test_acc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred_int,
    labels=range(NUM_CLASSES),
    target_names=le.classes_,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred_int)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=le.classes_, yticklabels=le.classes_
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()