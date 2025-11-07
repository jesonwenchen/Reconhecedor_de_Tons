import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- Carregar o modelo e os dados de teste ---
print("Carregando modelo e dados de teste...")
model = tf.keras.models.load_model('modelo_classificador_v3_94_07.keras')
X_test = np.load('features_combinado_v2_X.npy') # Vamos usar todos os dados para ter uma visão geral
y_test = np.load('labels_combinado_v2_y.npy')

# Ajustar os dados da mesma forma que no treino
y_true = y_test - 1
X_test = np.expand_dims(X_test, axis=-1)

# --- Fazer previsões ---
print("Fazendo previsões nos dados de teste...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- Gerar a Matriz de Confusão ---
print("Gerando Matriz de Confusão...")
cm = confusion_matrix(y_true, y_pred)
tons = ['Tom 1', 'Tom 2', 'Tom 3', 'Tom 4']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tons, yticklabels=tons)
plt.title('Matriz de Confusão')
plt.ylabel('Tom Verdadeiro')
plt.xlabel('Tom Previsto')
plt.show()

# --- Relatório de Classificação ---
print("\n--- Relatório de Classificação ---")
print(classification_report(y_true, y_pred, target_names=tons))