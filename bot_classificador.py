import logging
import os
import uuid
import numpy as np
import tensorflow as tf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Importa nossas funções de processamento de áudio
from processador_audio import extrair_e_normalizar_pitch

# --- CONFIGURAÇÕES ---
TOKEN = "SEU_TOKEN_AQUI" # Insira o token do seu bot
MODEL_PATH = 'modelo_classificador_tons_otimizado.keras'
TEMP_DIR = "temp_audios"
CLASS_NAMES = ['Tom 1', 'Tom 2', 'Tom 3', 'Tom 4', 'Tom 5', 'Tom 6', 'Tom 7', 'Tom 8'] # Mapeia a saída do modelo para nomes legíveis

# Configura o logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CARREGAMENTO DO MODELO ---
# Carregamos o modelo uma vez quando o bot inicia para não recarregá-lo a cada mensagem.
logger.info("Carregando modelo de classificação de tons...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary() # Mostra a arquitetura do modelo no console ao iniciar
    logger.info("Modelo carregado com sucesso!")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo '{MODEL_PATH}': {e}")
    model = None

# --- FUNÇÕES DO BOT ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Envia uma mensagem de boas-vindas."""
    user = update.effective_user
    await update.message.reply_html(
        f"Olá, {user.mention_html()}! 👋\n\n"
        "Eu sou um bot que classifica tons de áudio. "
        "Envie-me uma mensagem de voz ou um arquivo de áudio para eu analisar.",
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Processa o áudio recebido, classifica e responde."""
    if not model:
        await update.message.reply_text("Desculpe, o modelo de IA não está carregado. Contate o administrador.")
        return

    message = update.message
    audio_file_obj = message.audio or message.voice
    if not audio_file_obj:
        return

    # Gera um nome de arquivo único para evitar conflitos
    file_id = str(uuid.uuid4())
    file_extension = ".ogg" if message.voice else os.path.splitext(audio_file_obj.file_name)[1]
    temp_audio_path = os.path.join(TEMP_DIR, f"{file_id}{file_extension}")

    try:
        await update.message.reply_text("Analisando seu áudio... 🧠")
        
        # 1. Baixar o áudio
        file_info = await context.bot.get_file(audio_file_obj.file_id)
        await file_info.download_to_drive(temp_audio_path)
        logger.info(f"Áudio salvo temporariamente em: {temp_audio_path}")

        # 2. Extrair features do áudio
        feature_vector = extrair_e_normalizar_pitch(temp_audio_path) #
        
        # 3. Preparar o vetor para o modelo
        # O modelo espera um "lote" de dados. A forma precisa ser (1, 100, 1)
        # 1 = um áudio por vez | 100 = tamanho da feature | 1 = canal
        feature_vector = np.expand_dims(feature_vector, axis=0) # Adiciona dimensão do lote
        feature_vector = np.expand_dims(feature_vector, axis=-1) # Adiciona dimensão do canal
        
        # 4. Fazer a previsão
        prediction_probs = model.predict(feature_vector)
        predicted_class_index = np.argmax(prediction_probs, axis=1)[0]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction_probs) * 100

        # 5. Responder ao usuário
        logger.info(f"Previsão: {predicted_class_name} com {confidence:.2f}% de confiança.")
        await update.message.reply_text(
            f"Análise concluída! 🎼\n\n"
            f"Eu acredito que este áudio seja do: **{predicted_class_name}**\n"
            f"_(Confiança: {confidence:.2f}%)_"
        )

    except Exception as e:
        logger.error(f"Erro no processamento do áudio: {e}")
        await update.message.reply_text("Ocorreu um erro ao analisar seu áudio. Por favor, tente novamente.")
    
    finally:
        # 6. Limpeza: apagar o arquivo de áudio temporário
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.info(f"Arquivo temporário removido: {temp_audio_path}")

def main() -> None:
    """Inicia o bot e configura os handlers."""
    # Cria o diretório temporário se não existir
    os.makedirs(TEMP_DIR, exist_ok=True)

    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio))

    print("Bot iniciado. Pressione Ctrl+C para parar.")
    application.run_polling()

if __name__ == "__main__":
    main()