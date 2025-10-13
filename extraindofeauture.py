import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm # Para ter uma barra de progresso bonita!


df = pd.read_csv('C:\ID_tones\mapa_do_dataset_aumentado.csv')

def extrair_contorno_pitch(caminho_arquivo, fmin=80, fmax=450):
    """
    Carrega um arquivo de áudio, extrai o contorno de pitch (F0) e o limpa.

    Args:
        caminho_arquivo (str): O caminho para o arquivo de áudio (.wav, .mp3, etc.).
        fmin (int): Frequência mínima para a busca do pitch (em Hz).
        fmax (int): Frequência máxima para a busca do pitch (em Hz).

    Returns:
        np.ndarray: Um array NumPy contendo o contorno de pitch limpo.
    """
    # 1. Carregar o arquivo de áudio
    # Usamos uma taxa de amostragem (sr) fixa para consistência entre os arquivos.
    # 16000 Hz é comum para processamento de fala.
    y, sr = librosa.load(caminho_arquivo, sr=16000)

    # 2. Extrair o pitch usando o algoritmo pYIN
    # pYIN é robusto e nos dá não apenas o F0, mas também a probabilidade de ser um som "vozeado"
    f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                                 fmin=fmin,
                                                 fmax=fmax,
                                                 sr=sr)

    # 3. Lidar com valores "NaN" (Not a Number)
    # O F0 terá 'NaN' em partes silenciosas ou não-vozeadas (como o som de 's').
    # Para o modelo de ML, precisamos de um vetor contínuo.
    # Uma estratégia simples é a interpolação: preencher os valores ausentes.
    
    # Encontrar os índices onde f0 não é NaN
    indices_validos = np.where(~np.isnan(f0))[0]
    
    # Se não houver nenhum pitch detectado, retorna um array de zeros
    if len(indices_validos) < 2: # Precisa de pelo menos 2 pontos para interpolar
        return np.zeros(100) # Retorna um vetor de tamanho fixo

    # Criar os pontos para interpolação
    f0_validos = f0[indices_validos]
    indices_todos = np.arange(len(f0))

    # Interpolar para preencher os NaNs
    f0_interpolado = np.interp(indices_todos, indices_validos, f0_validos)

    return f0_interpolado

def visualizar_pitch(caminho_arquivo):
    """
    Extrai e visualiza o contorno de pitch de um arquivo de áudio.
    """
    # Supondo que você tenha um arquivo chamado 'ma1.wav' (1º tom)
    # ou 'ma2.wav' (2º tom), etc.
    contorno_pitch = extrair_contorno_pitch(caminho_arquivo)

    plt.figure(figsize=(12, 4))
    plt.plot(contorno_pitch, label='Contorno de Pitch (F0)', color='red')
    plt.title(f'Pitch para o arquivo: {caminho_arquivo}')
    plt.xlabel('Tempo (em frames)')
    plt.ylabel('Frequência (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Exemplo de Uso ---
# Crie um arquivo de áudio de teste ou use um do dataset Tone Perfect
# Por exemplo, se você tem um arquivo 'minha_gravacao.wav'
# visualize_pitch('caminho/para/sua/gravacao.wav')

# Se você não tiver um arquivo, esta função não será executada.
# Substitua 'seu_arquivo_de_audio.wav' pelo caminho real.


def extrair_e_normalizar_pitch(caminho_arquivo, tamanho_fixo=100):
    """
    Extrai o pitch e o redimensiona para um tamanho fixo para o modelo de ML.
    """
    contorno_pitch = extrair_contorno_pitch(caminho_arquivo)

    # Redimensionar o contorno para um tamanho fixo usando interpolação
    # Isso estica ou comprime a curva de pitch para que caiba no tamanho desejado
    x_original = np.linspace(0, 1, len(contorno_pitch))
    x_novo = np.linspace(0, 1, tamanho_fixo)
    
    vetor_de_feature = np.interp(x_novo, x_original, contorno_pitch)
    
    return vetor_de_feature

# a tqdm mostra uma barra de progresso enquanto o loop executa
df['feature_vector'] = [extrair_e_normalizar_pitch(caminho) for caminho in tqdm(df['caminho_arquivo'])]

print("Extração de features concluída!")

# X são todos os vetores de pitch empilhados
X = np.array(df['feature_vector'].tolist())

# y são as etiquetas de tom
y = df['tom'].values

np.save('featuresaumentado_X.npy', X)
np.save('labelsaumentao_y.npy', y)
print("Features e labels salvos em arquivos .npy!")
