import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import os
from tqdm import tqdm

# --- 1. CONFIGURAÇÃO ---
ARQUIVO_MAPA_ENTRADA = "mapa_dataset_combinado_v2.csv"
PASTA_SAIDA_AUDIOS = "dataset_combinado_aumentado"
ARQUIVO_MAPA_SAIDA = "mapa_dataset_aumentado_v1.csv"
TAXA_AMOSTRAGEM = 16000 # Mantenha a mesma do seu projeto

# --- 2. FUNÇÕES DE AUMENTAÇÃO (VERSÃO MAIS LEVE) ---

def add_noise(y, noise_factor=0.001): # ANTES: 0.005. Agora é 5x mais fraco.
    """Adiciona ruído branco gaussiano BEM SUTIL."""
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    augmented_data = np.clip(augmented_data, -1.0, 1.0)
    return augmented_data

def time_stretch(y, rate=0.95): # ANTES: 0.9. Agora é uma mudança de 5%, não 10%.
    """Aplica um time stretching BEM SUTIL."""
    # Vamos adicionar uma taxa aleatória para mais variedade
    rate = np.random.uniform(low=0.95, high=1.05) # Entre 5% mais lento e 5% mais rápido
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shift(y, sr, n_steps=0.5): # ANTES: 1 semitom. Agora é meio semitom.
    """Muda o pitch BEM SUTILMENTE."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

# --- 3. LÓGICA PRINCIPAL ---

print("--- Iniciando Data Augmentation ---")

# Verifica se o mapa de entrada existe
if not os.path.exists(ARQUIVO_MAPA_ENTRADA):
    print(f"ERRO: Arquivo de mapa '{ARQUIVO_MAPA_ENTRADA}' não encontrado.")
    exit()

# Cria a pasta de saída para os novos áudios
os.makedirs(PASTA_SAIDA_AUDIOS, exist_ok=True)
print(f"Novos áudios serão salvos em: '{PASTA_SAIDA_AUDIOS}/'")

# Carrega o mapa combinado
df = pd.read_csv(ARQUIVO_MAPA_ENTRADA)
print(f"Carregadas {len(df)} amostras do mapa original.")

# Lista para guardar os dados do novo mapa (originais + aumentados)
lista_novos_dados = []

# Loop principal com barra de progresso
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Aumentando dataset"):
    caminho_original = row['caminho_arquivo']
    pinyin = row['pinyin']
    tom = row['tom']
    
    # Pega o nome base do arquivo (ex: "ba1_JJ_g_1.wav" -> "ba1_JJ_g_1")
    try:
        nome_base = os.path.basename(caminho_original).rsplit('.', 1)[0]
    except Exception as e:
        print(f"Erro ao processar nome: {caminho_original}. Pulando. Erro: {e}")
        continue
        
    # 1. Adiciona o arquivo ORIGINAL à lista
    # (Mantemos o caminho original, não precisamos copiá-lo)
    lista_novos_dados.append(row.to_dict())
    
    # Carrega o áudio original
    try:
        y, sr = librosa.load(caminho_original, sr=TAXA_AMOSTRAGEM)
        if len(y) == 0:
            print(f"AVISO: Arquivo vazio {caminho_original}. Pulando aumentações.")
            continue
            
        # 2. AUMENTAÇÃO 1: Adicionar Ruído
        y_noise = add_noise(y)
        nome_noise = f"{nome_base}_noise.wav"
        caminho_saida_noise = os.path.join(PASTA_SAIDA_AUDIOS, nome_noise)
        sf.write(caminho_saida_noise, y_noise, sr)
        lista_novos_dados.append({'caminho_arquivo': caminho_saida_noise, 'pinyin': pinyin, 'tom': tom})

        # 3. AUMENTAÇÃO 2: Time Stretch (lento)
        y_stretch = time_stretch(y, rate=0.9)
        nome_stretch = f"{nome_base}_stretch.wav"
        caminho_saida_stretch = os.path.join(PASTA_SAIDA_AUDIOS, nome_stretch)
        sf.write(caminho_saida_stretch, y_stretch, sr)
        lista_novos_dados.append({'caminho_arquivo': caminho_saida_stretch, 'pinyin': pinyin, 'tom': tom})

        # 4. AUMENTAÇÃO 3: Pitch Shift (agudo)
        y_pitch = pitch_shift(y, sr, n_steps=1)
        nome_pitch = f"{nome_base}_pitch.wav"
        caminho_saida_pitch = os.path.join(PASTA_SAIDA_AUDIOS, nome_pitch)
        sf.write(caminho_saida_pitch, y_pitch, sr)
        lista_novos_dados.append({'caminho_arquivo': caminho_saida_pitch, 'pinyin': pinyin, 'tom': tom})

    except Exception as e:
        print(f"AVISO: Falha ao carregar ou processar {caminho_original}. Pulando aumentações. Erro: {e}")

# --- 4. SALVAR O NOVO MAPA ---
print("\nProcesso de aumentação concluído.")
df_aumentado = pd.DataFrame(lista_novos_dados)
df_aumentado.to_csv(ARQUIVO_MAPA_SAIDA, index=False)

print(f"\n--- ✅ Sucesso! ---")
print(f"Dataset original tinha: {len(df)} amostras.")
print(f"Novo dataset aumentado tem: {len(df_aumentado)} amostras.")
print(f"Novo mapa salvo como: '{ARQUIVO_MAPA_SAIDA}'")