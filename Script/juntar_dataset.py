import pandas as pd
import os

# --- 1. CONFIGURAÇÃO ---
# Verifique se os nomes dos seus arquivos CSV estão corretos
ARQUIVO_DATASET_ORIGINAL = "mapa_do_dataset.csv"
ARQUIVO_DATASET_PESSOAL = "mapa_do_dataset_v3.csv"

ARQUIVO_DATASET_COMBINADO = "mapa_dataset_combinado_v2.csv"

# --- 2. VERIFICAÇÃO ---
if not os.path.exists(ARQUIVO_DATASET_ORIGINAL):
    print(f"ERRO: Não foi possível encontrar o arquivo '{ARQUIVO_DATASET_ORIGINAL}'")
    exit()

if not os.path.exists(ARQUIVO_DATASET_PESSOAL):
    print(f"ERRO: Não foi possível encontrar o arquivo '{ARQUIVO_DATASET_PESSOAL}'")
    print("Execute o script para criar o mapa do seu dataset pessoal primeiro.")
    exit()

# --- 3. CARREGAR E JUNTAR ---
print(f"Carregando '{ARQUIVO_DATASET_ORIGINAL}'...")
df_original = pd.read_csv(ARQUIVO_DATASET_ORIGINAL)
print(f"Encontradas {len(df_original)} amostras no dataset original.")

print(f"Carregando '{ARQUIVO_DATASET_PESSOAL}'...")
df_pessoal = pd.read_csv(ARQUIVO_DATASET_PESSOAL)
print(f"Encontradas {len(df_pessoal)} amostras no seu dataset pessoal.")

# Juntando os dois DataFrames
# ignore_index=True é importante para criar um novo índice de 0 até o fim
df_combinado = pd.concat([df_original, df_pessoal], ignore_index=True)

# --- 4. SALVAR O NOVO MAPA ---
try:
    df_combinado.to_csv(ARQUIVO_DATASET_COMBINADO, index=False)
    print(f"\n--- ✅ Sucesso! ---")
    print(f"Total de amostras combinadas: {len(df_combinado)}")
    print(f"Novo mapa salvo como: '{ARQUIVO_DATASET_COMBINADO}'")

except Exception as e:
    print(f"\nERRO ao salvar o arquivo combinado: {e}")