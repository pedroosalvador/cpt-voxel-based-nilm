import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# file paths
x_path = './preprocessed_data/X_plaid_whited.npy'
y_path = './preprocessed_data/y_plaid_whited.npy'

# ---------- preprocess data ----------
X, y = np.load(x_path), np.load(y_path)

# hyperparameters
BATCH_SIZE = 32
EPOCHS = 50 
MODEL_PATH = 'model_plaid_whited.keras'
VOXEL_RESOLUTION = 32 
NUM_CLASSES = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30,
    random_state=42, 
    stratify=y
)
    
# transform labels to integers
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_true = le.transform(y_test)

# convert labels to categorical (one-hot encoding)
y_train_onehot = tf.keras.utils.to_categorical(y_train_int, NUM_CLASSES)
y_true_onehot = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)

# ---------- build model ----------
model = tf.keras.models.Sequential([
    #tf.keras.Input(shape=(32, 32, 32, 1)),

    # convolutional block 1
    tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', 
                           input_shape=(VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

    # convolutional block 2
    tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

    # convolutional block 3
    tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
    
    tf.keras.layers.Flatten(),

    # classification head
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.6),  
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- callbacks ----------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
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

# ---------- reports and evaluation ----------
y_pred_prob = model.predict(X_test)
y_pred_int = np.argmax(y_pred_prob, axis=1)

test_loss, test_acc = model.evaluate(X_test, y_true_onehot, verbose=0)
print(f"\nAccuracy: {test_acc:.4f}")

report = classification_report(
    y_true, 
    y_pred_int, 
    labels=range(NUM_CLASSES), 
    target_names=le.classes_,
    zero_division=0
)
print("\nClassification Report:")
print(report) 

cm = confusion_matrix(y_true, y_pred_int)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()