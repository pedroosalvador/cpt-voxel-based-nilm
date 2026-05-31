import numpy as np
import os, sys
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

# file paths
x_path = './preprocessed_data/X_plaid_whited.npy'
y_path = './preprocessed_data/y_plaid_whited.npy'

X, y = np.load(x_path), np.load(y_path)

# ---------- load model ----------
# hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = './models/model_resnet3d.keras'
NUM_CLASSES = len(np.unique(y))

model = tf.keras.models.load_model(MODEL_PATH)

unique_classes, class_counts = np.unique(y, return_counts=True)
for cls, count in zip(unique_classes, class_counts):
    print(f"{cls:30s}: {count:4d} samples")
print(f"Data shape: {X.shape}")
print(f"Total samples: {len(y)}")
print(f"Number of classes: {len(unique_classes)}")

# split data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ---------- encode labels ----------    
# transform labels to integers
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_true = le.transform(y_test)

# convert labels to categorical (one-hot encoding)
y_train_onehot = tf.keras.utils.to_categorical(y_train_int, NUM_CLASSES)
y_true_onehot = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)

# ---------------- extract embeddings ----------------
embedding_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)

emb_train = embedding_model.predict(X_train, batch_size=32, verbose=1)
emb_test  = embedding_model.predict(X_test,  batch_size=32, verbose=1)

# ---------------- apply t-SNE ----------------
# t-SNE is non-parametric; train/test must be combined so the 2D space is consistent.
emb_all = np.concatenate([emb_train, emb_test], axis=0) # concatenate train + test

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    learning_rate='auto',
    init='pca'
)

all_2d = tsne.fit_transform(emb_all)

# split again
train_2d = all_2d[:len(emb_train)]
test_2d  = all_2d[len(emb_train):]

# ---------------- classifier ----------------
clf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

clf.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

clf.fit(
    train_2d, y_train_int,
    epochs=30,
    validation_data=(test_2d, y_true),  
    verbose=1
)

clf.save('./models/tsne_model.keras')

# ---------- reports and evaluation ----------
y_pred_prob = clf.predict(test_2d)
y_pred_int = np.argmax(y_pred_prob, axis=1)

test_loss, test_acc = clf.evaluate(test_2d, y_true, verbose=0)
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