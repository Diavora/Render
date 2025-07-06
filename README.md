# Telegram Bot Marketplace

This is a Telegram bot with a Web App that functions as a marketplace for in-game items, similar to Playerok.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure the bot:**
    - Open `bot.py` and replace `YOUR_TELEGRAM_BOT_TOKEN` with your actual bot token from BotFather.

3.  **Run the application:**
    - Start the web server:
      ```bash
      uvicorn web.app:app --reload
      ```
    - Start the Telegram bot:
      ```bash
      python bot.py
      ```
