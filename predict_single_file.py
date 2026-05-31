import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import sys
import os
import json
import polars as pl

# Adicionar path para importar FocalLoss
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'architectures'))
from models.architectures.FocalLoss import FocalLoss

from data_processing.cpt_decomposition import CPT
from data_processing.normalization import normalize
from data_processing.voxelization import build_voxel_dataset
from data_processing.Data import Data

# ==================== CONFIGURAÇÕES ====================
# Modelo
MODEL_PATH = './models/checkpoints/embedded_model.keras'  # Modelo reduzido para embarcado
VOXEL_RESOLUTION = 8

# Arquivo para testar - COLOQUE O CAMINHO COMPLETO AQUI
FILE_PATH = './datasets/PLAID/submetered/950.csv'

# Caminho dos metadados
METADATA_PATH = './datasets/PLAID/metadata_submetered.json'
# ======================================================

# Carregar o dataset para obter os labels
print("Carregando dados para obter labels...")
X_full = np.load('./preprocessed_data/X_PLAID-WHITED_R8_AUG.npy')
y_full = np.load('./preprocessed_data/y_PLAID-WHITED_R8_AUG.npy')

# Preparar encoder de labels
le = LabelEncoder()
le.fit(y_full)

print(f"\nClasses disponíveis: {le.classes_}")
print(f"Total de classes: {len(le.classes_)}")

# Carregar o modelo treinado
print(f"\nCarregando modelo: {MODEL_PATH}")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'FocalLoss': FocalLoss}
)

# Carregar arquivo específico
print(f"\n{'='*70}")
print(f"CARREGANDO ARQUIVO ESPECÍFICO:")
print(f"{'='*70}")
print(f"Caminho: {FILE_PATH}")

# Verificar se o arquivo existe
if not os.path.exists(FILE_PATH):
    print(f"\n❌ ERRO: Arquivo não encontrado: {FILE_PATH}")
    print("\nExemplos de caminhos válidos:")
    print("  ./datasets/PLAID/submetered/0.csv")
    print("  ./datasets/PLAID/submetered/100.csv")
    print("  C:/Users/pedro/Downloads/Research/cpt-voxel-based-nilm/datasets/PLAID/submetered/0.csv")
    exit(1)

# Carregar metadados
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

# Extrair ID do arquivo
file_id = os.path.splitext(os.path.basename(FILE_PATH))[0]

# Verificar se existe nos metadados
if file_id not in metadata:
    print(f"\n❌ ERRO: Arquivo {file_id} não encontrado nos metadados")
    exit(1)

# Carregar dados do CSV
data_csv = pl.read_csv(FILE_PATH, has_header=False, new_columns=["Current", "Voltage"])

# Obter metadados do arquivo
meta = metadata[file_id]
label = (meta["appliance"]["type"]).upper()
sampling_frequency = int(meta["header"]["sampling_frequency"].replace("Hz", ""))
f_mains = 60

current_segment = data_csv["Current"].to_numpy()
voltage_segment = data_csv["Voltage"].to_numpy()

# Criar objeto Data
data_obj = Data(current_segment, voltage_segment, label, sampling_frequency, f_mains)

print(f"✓ Arquivo carregado com sucesso!")
print(f"  ID: {file_id}")
print(f"  Label: {label}")
print(f"  Frequência de amostragem: {sampling_frequency} Hz")
print(f"  Amostras: {len(current_segment)}")
print(f"{'='*70}")

# Processar o arquivo (converter para voxel)
print("Processando arquivo (CPT + Voxelização)...")
cpt_components = CPT(data_obj)
normalized_sample = normalize(cpt_components, data_obj)
X_voxel = build_voxel_dataset(normalized_sample)

# Usar diretamente (já tem batch dimension)
X_input = X_voxel  # shape: (1, 8, 8, 8, 1)

# Fazer predição
print("Fazendo predição...")
prediction_prob = model.predict(X_input, verbose=0)
predicted_class_idx = np.argmax(prediction_prob[0])
predicted_label = le.classes_[predicted_class_idx]
confidence = prediction_prob[0][predicted_class_idx] * 100

# Mostrar resultados
print("\n" + "="*70)
print("RESULTADO DA PREDIÇÃO:")
print("="*70)
print(f"Label Real:      {data_obj.label}")
print(f"Label Predito:   {predicted_label}")
print(f"Confiança:       {confidence:.2f}%")
print(f"Correto:         {'✓ SIM' if predicted_label == data_obj.label else '✗ NÃO'}")
print("="*70)

# Mostrar top 3 predições
print("\nTop 3 predições mais prováveis:")
top_3_indices = np.argsort(prediction_prob[0])[-3:][::-1]
for i, idx in enumerate(top_3_indices, 1):
    label = le.classes_[idx]
    prob = prediction_prob[0][idx] * 100
    print(f"  {i}. {label:25s} - {prob:5.2f}%")

