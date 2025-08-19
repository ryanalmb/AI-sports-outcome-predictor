"""
Main entry point for the Sports Prediction Telegram Bot
"""

import asyncio
import logging
import os
from telegram.ext import Application
from .handlers import setup_handlers
from aiohttp import web

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
    # Use PORT environment variable for deployment flexibility (Render, Railway, etc.)
    port = int(os.getenv('PORT', 10000))  # Default to Render.com's standard port
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Health check server started on port {port}")

    # Keep the server running
    while True:
        await asyncio.sleep(3600)  # Keep alive

async def run_bot_and_server():
    """Run both the Telegram bot and HTTP server concurrently"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

    # Create application
    application = Application.builder().token(bot_token).build()

    # Setup handlers
    setup_handlers(application)

    # Run both bot and server concurrently
    await asyncio.gather(
        start_web_server(),
        application.run_polling()
    )

def main():
    """Main function"""
    try:
        # Run both bot and server in the same event loop
        asyncio.run(run_bot_and_server())
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise

if __name__ == "__main__":
    main()
