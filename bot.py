import logging
import multiprocessing
import uvicorn
import sys
from telegram import Update, WebAppInfo, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# Ensure the 'web' module can be found by adding the project root to the path
sys.path.append('.')
from web.app import app as fastapi_app

# --- LOGGING SETUP ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
BOT_TOKEN = "7839428354:AAG53ugX-BPtorfl3SW-VaGOXKxJugDVGzw"
# This URL must point to your ngrok tunnel forwarding to LOCAL_PORT
WEB_APP_URL = "https://30a8-212-23-200-50.ngrok-free.app"
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8000

def run_fastapi():
    """Function to run the FastAPI server using uvicorn."""
    print(f"Starting FastAPI server on http://{LOCAL_HOST}:{LOCAL_PORT}")
    uvicorn.run(fastapi_app, host=LOCAL_HOST, port=LOCAL_PORT)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /start command. Sends a button to open the web app."""
    keyboard = [
        [InlineKeyboardButton("Открыть магазин", web_app=WebAppInfo(url=WEB_APP_URL))]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        'Добро пожаловать в наш магазин! Нажмите кнопку ниже, чтобы начать.',
        reply_markup=reply_markup
    )

def main() -> None:
    """Main function to start the web server and the Telegram bot."""
    # Start the FastAPI server in a separate process
    server_process = multiprocessing.Process(target=run_fastapi)
    server_process.start()
    logger.info(f"Web server started as a separate process (PID: {server_process.pid}).")

    # Build and run the Telegram bot application
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))

    logger.info("Bot is running... Press Ctrl-C to stop.")
    try:
        application.run_polling()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot polling stopped by user.")
    finally:
        logger.info("Stopping web server...")
        server_process.terminate()
        server_process.join()
        logger.info("Web server stopped.")

if __name__ == "__main__":
    # freeze_support() is necessary for multiprocessing to work correctly on Windows
    multiprocessing.freeze_support()
    main()
