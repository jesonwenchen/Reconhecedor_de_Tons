import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.preprocessing import StandardScaler # Importante para SVMs!
from sklearn.metrics import classification_report, accuracy_score

print("\n--- Treinamento do Modelo SVM ---")

# --- 1. Carregamento dos Dados ---
print("\nCarregando features e labels pré-processados...")
X = np.load('features_X.npy')
y = np.load('labels_y.npy')
print(f"Dados carregados. Formato de X: {X.shape}, Formato de y: {y.shape}")


# --- 2. Preparação dos Dados ---
# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# MUDANÇA IMPORTANTE: Normalização dos Dados
# SVMs são muito sensíveis à escala das features. É crucial normalizá-las.
print("Normalizando os dados...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 3. Criação e Treinamento do Modelo SVM ---
print("\nCriando e treinando o modelo SVM...")
# Usamos o kernel 'rbf' (Radial Basis Function), que é um ótimo ponto de partida.
# O parâmetro 'C' controla a regularização.
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)

# O treinamento é feito com uma única chamada .fit()
svm_model.fit(X_train_scaled, y_train)
print("Treinamento concluído!")


# --- 4. Avaliação do Modelo ---
print("\n--- Avaliação Final do Modelo SVM ---")
y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")

print("\n--- Relatório de Classificação ---")
# Lembre-se que y vai de 1 a 4
target_names = ['Tom 1', 'Tom 2', 'Tom 3', 'Tom 4']
print(classification_report(y_test, y_pred, target_names=target_names))

# (Opcional) Salvar o modelo SVM
import joblib
joblib.dump(svm_model, 'modelo_svm.pkl')
joblib.dump(scaler, 'scaler_svm.pkl') # Precisa salvar o scaler também!