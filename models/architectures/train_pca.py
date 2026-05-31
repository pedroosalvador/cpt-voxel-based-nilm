import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- load data ----------
x_path = './preprocessed_data/X_plaid.npy'
y_path = './preprocessed_data/y_plaid.npy'

X, y = np.load(x_path), np.load(y_path)

NUM_CLASSES = len(np.unique(y))

BATCH_SIZE = 32
EPOCHS = 50 
MODEL_PATH = './models/model_plaid.keras'
VOXEL_RESOLUTION = 32 
PCA_COMPONENTS = 64

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30,
    random_state=42, 
    stratify=y
)
    
le = LabelEncoder()
y_train_int = le.fit_transform(y_train)
y_test_int = le.transform(y_test)

y_train_onehot = tf.keras.utils.to_categorical(y_train_int, NUM_CLASSES)
y_test_onehot = tf.keras.utils.to_categorical(y_test_int, NUM_CLASSES) 

# ============================================
# LOAD MODEL & EXTRACT FEATURES
# ============================================
model_original = tf.keras.models.load_model(MODEL_PATH)

print(f"Camada -3: {model_original.layers[-3].name}")
print(f"Output shape: {model_original.layers[-3].output.shape}")

# Extrator de features (até Dense(256))
embedding_model = tf.keras.Model(
    inputs=model_original.inputs,
    outputs=model_original.layers[-3].output
)

features_train = embedding_model.predict(X_train, batch_size=32, verbose=1)  
features_test = embedding_model.predict(X_test, batch_size=32, verbose=1)

# ============================================
# BASELINE (sem PCA)
# ============================================
_, acc_baseline = model_original.evaluate(X_test, y_test_onehot, verbose=0)
print(f"Acurácia baseline (256D): {acc_baseline*100:.2f}%")

# ============================================
# APLICAR PCA
# ============================================
pca = PCA(n_components=PCA_COMPONENTS, random_state=42)

# Fit PCA no conjunto de treino
pca.fit(features_train)

# Transform ambos os conjuntos
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)

# Variância explicada
variance_explained = np.sum(pca.explained_variance_ratio_) * 100

# ============================================
# CRIAR CAMADA PCA CUSTOMIZADA
# ============================================
class PCALayer(tf.keras.layers.Layer):
    """Camada TensorFlow que aplica transformação PCA"""
    def __init__(self, pca_components, pca_mean, name='pca_layer'):
        super(PCALayer, self).__init__(name=name)
        self.pca_components = pca_components
        self.pca_mean = pca_mean
    
    def build(self, input_shape):
        # Pesos não treináveis (frozen PCA transformation)
        self.components = self.add_weight(
            shape=self.pca_components.shape,
            initializer=tf.keras.initializers.Constant(self.pca_components),
            trainable=False,
            name='pca_components'
        )
        self.mean = self.add_weight(
            shape=self.pca_mean.shape,
            initializer=tf.keras.initializers.Constant(self.pca_mean),
            trainable=False,
            name='pca_mean'
        )
    
    def call(self, inputs):
        # Aplicar PCA: (X - mean) @ components.T
        centered = inputs - self.mean
        return tf.matmul(centered, tf.transpose(self.components))
    
    def get_config(self):
        config = super().get_config()
        return config

pca_layer = PCALayer(
    pca_components=pca.components_.astype(np.float32),
    pca_mean=pca.mean_.astype(np.float32)
)

# ============================================
# CONSTRUIR MODELO COM PCA - SOLUÇÃO CORRETA
# ============================================
print("\nConstruindo modelo com PCA integrado...")

# ✅ MÉTODO 1: Usar o extrator de features + PCA + nova cabeça
inputs = tf.keras.Input(shape=(VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION, 1))

# Usar o embedding model (até Dense 256) - frozen
x = embedding_model(inputs)

# Congelar todas as camadas do embedding model
for layer in embedding_model.layers:
    layer.trainable = False

# Aplicar PCA
x = pca_layer(x)

# Novas camadas treináveis
x = tf.keras.layers.Dropout(0.6, name='dropout_pca')(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='dense_output_pca')(x)

model_pca = tf.keras.Model(inputs=inputs, outputs=outputs)

model_pca.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nArquitetura do modelo com PCA:")
model_pca.summary()

# ============================================
# TREINAR MODELO COM PCA
# ============================================
print(f"\nTreinando modelo com PCA ({PCA_COMPONENTS}D)...")

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model_pca.fit(
    X_train, y_train_onehot,
    validation_data=(X_test, y_test_onehot),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================
# AVALIAR MODELO COM PCA
# ============================================
print("\nAvaliando modelo com PCA...")
_, acc_pca = model_pca.evaluate(X_test, y_test_onehot, verbose=0)

y_pred_prob = model_pca.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

# ============================================
# RESULTADOS
# ============================================
print("\n" + "="*70)
print("RESULTADOS COMPARATIVOS")
print("="*70)
print(f"Modelo Original (256D):       {acc_baseline*100:.2f}%")
print(f"Modelo com PCA ({PCA_COMPONENTS}D):         {acc_pca*100:.2f}%")
print(f"Variância explicada PCA:      {variance_explained:.2f}%")
print(f"Redução de dimensionalidade:  256 → {PCA_COMPONENTS} ({(1-PCA_COMPONENTS/256)*100:.1f}%)")
print(f"Perda de acurácia:            {(acc_baseline - acc_pca)*100:.2f} pontos percentuais")
print("="*70)

print("\nClassification Report (modelo com PCA):")
print(classification_report(y_test_int, y_pred, target_names=le.classes_, zero_division=0))

# ============================================
# VISUALIZAÇÃO COMPARATIVA
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras
ax1 = axes[0]
models = ['Original\n(256D)', f'PCA\n({PCA_COMPONENTS}D)']
accuracies = [acc_baseline*100, acc_pca*100]
colors = ['#3498db', '#e74c3c']

bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax1.set_ylabel('Acurácia (%)', fontsize=12)
ax1.set_title(f'Comparação de Acurácia\nPerda: {(acc_baseline - acc_pca)*100:.2f}pp', 
              fontsize=14, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.grid(axis='y', alpha=0.3)

# Curva de treinamento
ax2 = axes[1]
ax2.plot(history.history['accuracy'], label='Treino', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Validação', linewidth=2)
ax2.set_xlabel('Época', fontsize=12)
ax2.set_ylabel('Acurácia', fontsize=12)
ax2.set_title(f'Curva de Aprendizado (PCA {PCA_COMPONENTS}D)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'pca_{PCA_COMPONENTS}d_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Gráfico salvo: pca_{PCA_COMPONENTS}d_comparison.png")
plt.show()

# ============================================
# SALVAR MODELO
# ============================================
model_pca.save(f'./models/model_plaid_pca_{PCA_COMPONENTS}d.keras')
print(f"✓ Modelo com PCA salvo: ./models/model_plaid_pca_{PCA_COMPONENTS}d.keras")

# Salvar objeto PCA para reutilização
import pickle
with open(f'./models/pca_{PCA_COMPONENTS}d.pkl', 'wb') as f:
    pickle.dump(pca, f)
print(f"✓ Objeto PCA salvo: ./models/pca_{PCA_COMPONENTS}d.pkl")