"""
Simplified Sports Prediction Telegram Bot with Fixed Display
"""

import asyncio
import logging
from telegram.ext import Application
from .handlers import setup_handlers
from database_manager import DatabaseManager
from aiohttp import web
import os

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SimpleSportsBot:
    """Simplified Telegram bot for sports predictions"""

    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

        self.db_manager = DatabaseManager()
        self.application = Application.builder().token(self.bot_token).build()

    def run(self):
        """Run the bot"""
        try:
            setup_handlers(self.application)
            logger.info("Bot handlers configured")
            logger.info("Starting Sports Prediction Bot...")
            self.application.run_polling()
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            raise

async def health_check(request):
    """Health check endpoint for Replit"""
    return web.Response(text="Sports Prediction Bot is running!", status=200)

async def start_web_server():
    """Start web server for deployment platforms like Render.com"""
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_get('/', health_check)

    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv('PORT', 10000))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Health check server started on port {port}")

    while True:
        await asyncio.sleep(3600)

async def run_bot_and_server():
    """Run both the Telegram bot and HTTP server concurrently"""
    bot = SimpleSportsBot()

    await asyncio.gather(
        start_web_server(),
        bot.application.run_polling()
    )

def main():
    """Main function"""
    try:
        asyncio.run(run_bot_and_server())
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise

if __name__ == "__main__":
    main()