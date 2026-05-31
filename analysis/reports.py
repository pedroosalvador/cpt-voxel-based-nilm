import os, sys
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.architectures.FocalLoss import FocalLoss

# file paths, change as needed
x_path = './preprocessed_data/X_PLAID-WHITED_R8_AUG.npy'
y_path = './preprocessed_data/y_PLAID-WHITED_R8_AUG.npy'

X, y = np.load(x_path), np.load(y_path)

# ---------- load model ----------
# hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = './models/checkpoints/embedded_model.keras' # path to the trained model
NUM_CLASSES = len(np.unique(y))

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'FocalLoss': FocalLoss})

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
embedding_layer = model.layers[-2] 
embedding_model = tf.keras.Model(inputs=model.inputs, outputs=embedding_layer.output)

embeddings = embedding_model.predict(X_test, batch_size=32, verbose=1)

# ---------------- apply t-sne ----------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate='auto', init='pca')
X_embedded = tsne.fit_transform(embeddings)

# ---------- reports and evaluation ----------
# make predictions
y_pred_prob = model.predict(X_test)
y_pred_int = np.argmax(y_pred_prob, axis=1)

# geral accuracy
test_loss, test_acc = model.evaluate(X_test, y_true_onehot, verbose=0)
print(f"Accuracy: {test_acc:.4f}")

# classification report
report = classification_report(y_true, y_pred_int, target_names=le.classes_, zero_division=0)
print("Classification Report:")
print(report) 

# confusion matrix
cm = confusion_matrix(y_true, y_pred_int)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
""" plt.savefig(f'confusion_matrix_{MODEL_PATH.split("/")[-1].split(".")[0]}.png', dpi=300, bbox_inches='tight') """
plt.show()

# t-sne visualization
plt.figure(figsize=(10, 8))
palette = sns.color_palette("hsv", NUM_CLASSES)
sns.scatterplot(
    x=X_embedded[:, 0], y=X_embedded[:, 1],
    hue=[le.classes_[i] for i in y_true],
    palette=palette, legend='full', s=50, alpha=0.8
)
plt.title("t-SNE visualization of learned embeddings")
plt.tight_layout()
""" plt.savefig(f'tsne_visualization_{MODEL_PATH.split("/")[-1].split(".")[0]}.png', dpi=300, bbox_inches='tight') """
plt.show()