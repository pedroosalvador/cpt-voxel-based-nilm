import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from models.architectures.FocalLoss import FocalLoss

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ========== CONFIGURAÇÃO ==========
x_path = './preprocessed_data/X_PLAID-WHITED_R32_AUG.npy'
y_path = './preprocessed_data/y_PLAID-WHITED_R32_AUG.npy'
model_path = './models/checkpoints/RESNET3D_PLAID-WHITED_AUG_FL.keras'

# ========== CARREGAR DADOS ==========
print("Carregando dados...")
X, y = np.load(x_path), np.load(y_path)

# Análise das classes
unique_classes, counts = np.unique(y, return_counts=True)
num_classes = len(unique_classes)

print(f"\n{'='*60}")
print(f"ANÁLISE DO DATASET")
print(f"{'='*60}")
print(f"Total de classes: {num_classes}")
print(f"Total de amostras: {len(y)}")
print(f"\nDistribuição de amostras por classe:")
print("-" * 60)
for class_id, count in zip(unique_classes, counts):
    print(f"{str(class_id):25s}: {count:5d} amostras ({count/len(y)*100:.2f}%)")
print("-" * 60)

# ========== SPLIT TRAIN/TEST ==========
print("\nDividindo dados em train/test (70/30)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Encode labels
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_test_int = le.transform(y_test)

# Convert to categorical (one-hot encoding)
y_train_onehot = tf.keras.utils.to_categorical(y_train_int, num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test_int, num_classes)

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ========== CARREGAR MODELO ==========
print(f"\nCarregando modelo: {model_path}")
model = tf.keras.models.load_model(
    model_path,
    custom_objects={'FocalLoss': FocalLoss}
)

print("Modelo carregado com sucesso!")
model.summary()

# ========== AVALIAÇÃO ==========
print(f"\n{'='*60}")
print("AVALIANDO MODELO NO CONJUNTO DE TESTE")
print(f"{'='*60}")

# Predict
print("\nRealizando predições...")
y_pred_prob = model.predict(X_test, batch_size=32, verbose=1)
y_pred_int = np.argmax(y_pred_prob, axis=1)

# Calculate metrics
test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0, batch_size=32)

print(f"\n{'='*60}")
print("MÉTRICAS GERAIS")
print(f"{'='*60}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Classification Report
print(f"\n{'='*60}")
print("CLASSIFICATION REPORT")
print(f"{'='*60}")
print(classification_report(
    y_test_int,
    y_pred_int,
    labels=range(num_classes),
    target_names=le.classes_,
    zero_division=0
))

# Confusion Matrix
print(f"\n{'='*60}")
print("CONFUSION MATRIX")
print(f"{'='*60}")
cm = confusion_matrix(y_test_int, y_pred_int)

# Save metrics to file
with open('metrics_report.txt', 'w') as f:
    f.write(f"{'='*60}\n")
    f.write("MÉTRICAS DO MODELO RESNET3D - FOCAL LOSS\n")
    f.write(f"{'='*60}\n")
    f.write(f"Dataset: {x_path}\n")
    f.write(f"Modelo: {model_path}\n")
    f.write(f"\nTest Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
    f.write(classification_report(
        y_test_int,
        y_pred_int,
        labels=range(num_classes),
        target_names=le.classes_,
        zero_division=0
    ))

print("\nMétricas salvas em 'metrics_report.txt'")

# Plot Confusion Matrix
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=le.classes_, 
    yticklabels=le.classes_,
    cbar_kws={'label': 'Count'}
)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix - ResNet3D with Focal Loss', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix salva em 'confusion_matrix.png'")
plt.show()

print(f"\n{'='*60}")
print("ANÁLISE CONCLUÍDA!")
print(f"{'='*60}")

