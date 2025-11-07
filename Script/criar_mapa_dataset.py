import os
import pandas as pd

# Caminho para a pasta onde você descompactou os arquivos .wav
caminho_dataset = 'C:/ID_tones/DATASET_V3' 

dados = []
# Loop por cada arquivo na pasta
for nome_arquivo in os.listdir(caminho_dataset):
    if nome_arquivo.endswith('.mp3'):
        # Caminho completo para o arquivo
        caminho_completo = os.path.join(caminho_dataset, nome_arquivo)

        # Extrai o pinyin e o tom do nome do arquivo zhong4_FV1_MP3.mp3
        # Ex: 'a1.wav' -> pinyin='a', tom='1'
        pinyin = nome_arquivo[:-4] # Remove o '.wav' e o número do tom
        tom = nome_arquivo[-5]   # Pega apenas o número do tom
        # Adiciona as informações à nossa lista 
        dados.append({
            'caminho_arquivo': caminho_completo,
            'pinyin': pinyin,
            'tom': int(tom) # Converte o tom para número (1, 2, 3, 4)
        })

# Cria o DataFrame
df = pd.DataFrame(dados)

# Salva o mapa em um arquivo CSV para não precisar fazer isso toda vez
df.to_csv('mapa_do_dataset_v3.csv', index=False)

print("DataFrame criado com sucesso!")
print(df.head()) # Mostra as 5 primeiras linhas
print(f"\nTotal de amostras: {len(df)}")