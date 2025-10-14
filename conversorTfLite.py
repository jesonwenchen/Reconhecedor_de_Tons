import tensorflow as tf
import os

# --- Configuração ---
# Coloque aqui o nome do seu melhor modelo Keras.
# Pelo que vi nos seus arquivos, 'modelo_classificador_tons_otimizado.keras'
# parece ser um bom candidato, treinado com dados aumentados.
KERAS_MODEL_FILENAME = 'modelo_classificador_tons_otimizado.keras'

# Nome do arquivo de saída para o modelo convertido.
TFLITE_MODEL_FILENAME = 'model.tflite'

def convert_model():
    """
    Carrega um modelo Keras e o converte para o formato TensorFlow Lite.
    """
    print("--- Iniciando a conversão do modelo para TensorFlow Lite ---")

    # 1. Verificação do arquivo de entrada
    if not os.path.exists(KERAS_MODEL_FILENAME):
        print(f"ERRO: O arquivo do modelo '{KERAS_MODEL_FILENAME}' não foi encontrado.")
        print("Por favor, certifique-se de que o nome do arquivo está correto e ele está na mesma pasta que este script.")
        return

    try:
        # 2. Carregar o modelo Keras
        print(f"Carregando o modelo Keras de '{KERAS_MODEL_FILENAME}'...")
        model = tf.keras.models.load_model(KERAS_MODEL_FILENAME)
        print("Modelo carregado com sucesso.")
        model.summary()

        # 3. Criar o conversor TFLite
        print("Criando o conversor TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # 4. Otimizar o modelo (opcional, mas recomendado para mobile)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # 5. Converter o modelo
        print("Convertendo o modelo... Isso pode levar um momento.")
        tflite_model = converter.convert()
        print("Conversão concluída!")

        # 6. Salvar o modelo TFLite em um arquivo
        print(f"Salvando o modelo convertido como '{TFLITE_MODEL_FILENAME}'...")
        with open(TFLITE_MODEL_FILENAME, 'wb') as f:
            f.write(tflite_model)

        print("\n--- Processo finalizado com sucesso! ---")
        print(f"Seu modelo está pronto para ser usado em aplicativos móveis: '{TFLITE_MODEL_FILENAME}'")
        print(f"O tamanho do arquivo final é de aproximadamente {len(tflite_model) / 1024:.2f} KB.")

    except Exception as e:
        print(f"\nOcorreu um erro durante o processo: {e}")
        print("Verifique se o TensorFlow está instalado corretamente ('pip install tensorflow') e se o arquivo do modelo não está corrompido.")

if __name__ == '__main__':
    convert_model()
