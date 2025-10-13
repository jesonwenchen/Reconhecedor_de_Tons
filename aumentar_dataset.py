import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import os

# Carrega o mapa original
df = pd.read_csv('mapa_do_dataset.csv')

# Cria a pasta para o novo dataset
pasta_aumentada = 'dataset_aumentado'
os.makedirs(pasta_aumentada, exist_ok=True)

novos_dados = []

for index, row in df.iterrows():
    caminho_original = row['caminho_arquivo']
    y, sr = librosa.load(caminho_original, sr=16000)
    nome_base = os.path.basename(caminho_original)

    # Adiciona o arquivo original à nova lista
    novos_dados.append(row.to_dict())

    # 1. Time Stretching (mais lento)
    y_lento = librosa.effects.time_stretch(y, rate=0.9)
    novo_nome = f"{nome_base.replace('.wav', '')}_lento.wav"
    sf.write(os.path.join(pasta_aumentada, novo_nome), y_lento, sr)
    novos_dados.append({'caminho_arquivo': os.path.join(pasta_aumentada, novo_nome), 'pinyin': row['pinyin'], 'tom': row['tom']})

    # 2. Pitch Shifting (um pouco mais agudo)
    y_agudo = librosa.effects.pitch_shift(y, sr=sr, n_steps=0.5)
    novo_nome = f"{nome_base.replace('.wav', '')}_agudo.wav"
    sf.write(os.path.join(pasta_aumentada, novo_nome), y_agudo, sr)
    novos_dados.append({'caminho_arquivo': os.path.join(pasta_aumentada, novo_nome), 'pinyin': row['pinyin'], 'tom': row['tom']})

df_aumentado = pd.DataFrame(novos_dados)
df_aumentado.to_csv('mapa_do_dataset_aumentado.csv', index=False)
print(f"Dataset aumentado criado com {len(df_aumentado)} amostras!")