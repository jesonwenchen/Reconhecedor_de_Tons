import logging
import os
import uuid
import numpy as np
import tensorflow as tf
import configparser  # <--- Importa a biblioteca para ler o .ini
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from processador_audio import extrair_e_normalizar_pitch

# --- 1. LEITURA DAS CONFIGURAÇÕES ---
config = configparser.ConfigParser()
config.read('config.ini')

# Carrega as configurações da seção [Telegram]
TOKEN = config['Telegram']['Token']
allowed_ids_str = config['Telegram']['AllowedChatIds']
# Converte a string de IDs em uma lista de números inteiros
ALLOWED_CHAT_IDS = [int(id.strip()) for id in allowed_ids_str.split(',') if id.strip()]

# Carrega as configurações da seção [Model]
MODEL_PATH = config['Model']['ModelPath']
class_names_str = config['Model']['ClassNames']
CLASS_NAMES = [name.strip() for name in class_names_str.split(',')]

# Carrega as configurações da seção [Paths]
TEMP_DIR = config['Paths']['TempDir']

# Configura o logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. CARREGAMENTO DO MODELO ---
logger.info("Carregando modelo de classificação de tons...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
    logger.info("Modelo carregado com sucesso!")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo '{MODEL_PATH}': {e}")
    model = None

# --- 3. FUNÇÕES DO BOT ---

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
    chat_id = update.effective_chat.id

    # ---> NOVIDADE: Verifica se o usuário tem permissão <---
    if ALLOWED_CHAT_IDS and chat_id not in ALLOWED_CHAT_IDS:
        logger.warning(f"Acesso negado para o Chat ID: {chat_id}")
        await update.message.reply_text("Desculpe, você não tem permissão para usar este bot.")
        return

    if not model:
        await update.message.reply_text("Desculpe, o modelo de IA não está carregado. Contate o administrador.")
        return

    # O resto da função continua exatamente igual...
    message = update.message
    audio_file_obj = message.audio or message.voice
    temp_audio_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.ogg")

    try:
        await update.message.reply_text("Analisando seu áudio... 🧠")
        file_info = await context.bot.get_file(audio_file_obj.file_id)
        await file_info.download_to_drive(temp_audio_path)
        logger.info(f"Áudio de {chat_id} salvo em: {temp_audio_path}")

        feature_vector = extrair_e_normalizar_pitch(temp_audio_path)
        feature_vector = np.expand_dims(np.expand_dims(feature_vector, axis=0), axis=-1)

        prediction_probs = model.predict(feature_vector)
        predicted_class_index = np.argmax(prediction_probs, axis=1)[0]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction_probs) * 100

        logger.info(f"Previsão para {chat_id}: {predicted_class_name} ({confidence:.2f}%)")
        await update.message.reply_text(
            f"Análise concluída! 🎼\n\n"
            f"Eu acredito que este áudio seja do: **{predicted_class_name}**\n"
            f"_(Confiança: {confidence:.2f}%)_"
        )
    except Exception as e:
        logger.error(f"Erro no processamento do áudio para {chat_id}: {e}")
        await update.message.reply_text("Ocorreu um erro ao analisar seu áudio. Por favor, tente novamente.")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def main() -> None:
    """Inicia o bot."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio))
    
    logger.info("Bot iniciado. Pressione Ctrl+C para parar.")
    if ALLOWED_CHAT_IDS:
        logger.info(f"Bot configurado para aceitar requisições apenas dos seguintes Chat IDs: {ALLOWED_CHAT_IDS}")
    else:
        logger.warning("Nenhum Chat ID foi configurado. O bot está aberto para qualquer pessoa.")
        
    application.run_polling()

if __name__ == "__main__":
    main()