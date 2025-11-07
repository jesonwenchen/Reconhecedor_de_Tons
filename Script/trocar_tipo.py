import os
import glob
from pydub import AudioSegment
from tqdm import tqdm # Para uma barra de progresso! (Se não tiver: pip install tqdm)

# --- 1. CONFIGURAÇÃO ---
# Coloque o caminho para a pasta onde estão seus arquivos .m4a
PASTA_ENTRADA = "C:/Users/JesonWen/Downloads/DB_MISSING" 

# Coloque o caminho para onde você quer salvar os novos .mp3
PASTA_SAIDA = "C:/ID_tones/DB_MISSING_MP3" 

# --- 2. VERIFICAÇÃO E CRIAÇÃO DE PASTAS ---
if not os.path.exists(PASTA_ENTRADA):
    print(f"ERRO: A pasta de entrada não foi encontrada em: {PASTA_ENTRADA}")
    exit()

# Cria a pasta de saída se ela não existir
os.makedirs(PASTA_SAIDA, exist_ok=True)

# --- 3. ENCONTRAR OS ARQUIVOS .M4A ---
# O 'glob' encontra todos os arquivos que terminam com .m4a
# O 'recursive=True' permite que ele procure em subpastas
arquivos_m4a = glob.glob(os.path.join(PASTA_ENTRADA, "**/*.m4a"), recursive=True)

if not arquivos_m4a:
    print(f"Nenhum arquivo .m4a foi encontrado em {PASTA_ENTRADA}")
    exit()

print(f"Encontrados {len(arquivos_m4a)} arquivos .m4a para converter...")

# --- 4. CONVERSÃO EM LOTE ---
for caminho_m4a in tqdm(arquivos_m4a, desc="Convertendo áudios"):
    try:
        # Define o nome do arquivo de saída
        # Pega o nome do arquivo (ex: 'gravacao1.m4a') e muda para 'gravacao1.mp3'
        nome_arquivo = os.path.basename(caminho_m4a)
        nome_mp3 = nome_arquivo.replace('.m4a', '.mp3')
        caminho_mp3 = os.path.join(PASTA_SAIDA, nome_mp3)

        # Carrega o arquivo .m4a
        # 'format="m4a"' ajuda o pydub a identificar o formato
        audio = AudioSegment.from_file(caminho_m4a, format="m4a")

        # Exporta o arquivo como .mp3
        # 'bitrate="192k"' define a qualidade. 192k é um bom padrão.
        audio.export(caminho_mp3, format="mp3", bitrate="192k")
    
    except Exception as e:
        print(f"\nErro ao converter o arquivo {caminho_m4a}: {e}")

print("\n--- Conversão Concluída! ---")
print(f"Todos os arquivos foram salvos em: {PASTA_SAIDA}")