import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print(f"TensorFlow Version: {tf.__version__}")

# --- 2. Carregamento dos Dados ---
print("\nCarregando features e labels pré-processados...")
X = np.load('features_X.npy')
y = np.load('labels_y.npy')
print(f"Dados carregados. Formato de X: {X.shape}, Formato de y: {y.shape}")

# --- 3. Preparação dos Dados para o Treinamento ---
print("\nPreparando dados para o treinamento...")
y = y - 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print(f"Dados de treino (X_train): {X_train.shape}")
print(f"Dados de teste (X_test): {X_test.shape}")

# --- 4. Construção da Arquitetura do Modelo (1D-CNN) ---
print("\nConstruindo um modelo 1D-CNN mais simples e regularizado...")


# MUDANÇA 1: Arquitetura do Modelo Simplificada
model = Sequential([
    Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', input_shape=(100, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    LSTM(80),
    BatchNormalization(), # Estabiliza o treinamento da LSTM
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

# MUDANÇA 3: EarlyStopping com mais paciência
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,  # Um pouco mais de paciência para o modelo se estabilizar
    restore_best_weights=True
)

# --- 5. Compilação do Modelo ---
from tensorflow.keras.optimizers import Adam


model.compile(
    # MUDANÇA: Use um otimizador Adam com uma taxa de aprendizado menor
    optimizer=Adam(learning_rate=0.001), # O padrão é 0.001
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. Treinamento do Modelo ---
print("\nIniciando o treinamento do modelo...")
# MUDANÇA 4: Mais épocas, deixando o EarlyStopping trabalhar
history = model.fit(
    X_train, 
    y_train, 
    epochs=50,  # Aumente para 50. Não se preocupe, ele não vai rodar tudo.
    batch_size=32, # Mude para 32, um valor mais padrão e estável
    validation_data=(X_test, y_test),
    callbacks=[early_stopping] # Mantenha o EarlyStopping, ele é seu melhor amigo
)

# --- 7. Avaliação do Modelo ---
print("\n--- Avaliação Final do Modelo ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")
print(f"Loss no conjunto de teste: {loss:.4f}")

# --- 8. Salvando o Modelo Treinado ---
MODEL_SAVE_PATH = 'modelo_classificador_tons_otimizado.keras' # Novo nome para o modelo
print(f"\nSalvando o modelo treinado em '{MODEL_SAVE_PATH}'...")
model.save(MODEL_SAVE_PATH)
print("Modelo salvo com sucesso!")

# --- 9. (Bônus) Visualização do Histórico de Treinamento ---
def plotar_historico(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Acurácia de Treino')
    ax1.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    ax1.set_title('Acurácia do Modelo')
    ax1.set_ylabel('Acurácia')
    ax1.set_xlabel('Época')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    ax2.plot(history.history['loss'], label='Perda de Treino')
    ax2.plot(history.history['val_loss'], label='Perda de Validação')
    ax2.set_title('Perda (Loss) do Modelo')
    ax2.set_ylabel('Perda')
    ax2.set_xlabel('Época')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    plt.show()

print("\nGerando gráficos do treinamento...")
plotar_historico(history)