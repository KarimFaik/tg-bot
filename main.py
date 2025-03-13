import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from model import question_answer

load_dotenv()

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка текста из файла
def load_text():
    # Указываем путь к файлу напрямую
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data.txt")
    
    try:
        with open(data_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"Файл не найден: {data_path}")
        return None
    except Exception as e:
        logger.error(f"Ошибка при чтении файла: {e}")
        return None

# Команда /start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Привет! Я бот. Задайте мне вопрос!')

# Обработка текстовых сообщений
async def handle_message(update: Update, context: CallbackContext) -> None:
    question = update.message.text
    if not question:
        await update.message.reply_text("Пожалуйста, задайте вопрос.")
        return

    # Загрузка текста из файла
    text = load_text()
    if not text:
        await update.message.reply_text("Ошибка: текст не найден.")
        return

    # Поиск ответа
    answer = question_answer(text, question)
    await update.message.reply_text(answer)

# Обработка ошибок
async def error(update: Update, context: CallbackContext) -> None:
    logger.warning(f'Update {update} caused error {context.error}')

def main() -> None:
    # Получаем токен бота из переменной окружения
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        logger.error("Переменная TELEGRAM_BOT_TOKEN не найдена в .env")
        return

    # Создаем приложение
    application = Application.builder().token(telegram_bot_token).build()

    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start))

    # Регистрация обработчика текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Регистрация обработчика ошибок
    application.add_error_handler(error)

    # Запуск бота
    application.run_polling()

if __name__ == '__main__':
    main()