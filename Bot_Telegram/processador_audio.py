import librosa
import numpy as np

# Copiado diretamente do seu script extraindofeauture.py
def extrair_contorno_pitch(caminho_arquivo, fmin=80, fmax=450):
    """
    Carrega um arquivo de áudio, extrai o contorno de pitch (F0) e o limpa.
    """
    try:
        y, sr = librosa.load(caminho_arquivo, sr=16000)
        if len(y) == 0:
            print("Aviso: Arquivo de áudio vazio ou não pôde ser lido.")
            return np.zeros(100) # Retorna um vetor nulo se o áudio estiver vazio
            
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
        indices_validos = np.where(~np.isnan(f0))[0]
        
        if len(indices_validos) < 2:
            return np.zeros(100)

        f0_validos = f0[indices_validos]
        indices_todos = np.arange(len(f0))
        f0_interpolado = np.interp(indices_todos, indices_validos, f0_validos)
        return f0_interpolado
    except Exception as e:
        print(f"Erro ao processar o áudio {caminho_arquivo}: {e}")
        return np.zeros(100)

# Copiado diretamente do seu script extraindofeauture.py
def extrair_e_normalizar_pitch(caminho_arquivo, tamanho_fixo=100):
    """
    Extrai o pitch e o redimensiona para um tamanho fixo para o modelo de ML.
    """
    contorno_pitch = extrair_contorno_pitch(caminho_arquivo)
    x_original = np.linspace(0, 1, len(contorno_pitch))
    x_novo = np.linspace(0, 1, tamanho_fixo)
    vetor_de_feature = np.interp(x_novo, x_original, contorno_pitch)
    return vetor_de_feature