#!/usr/bin/env python3
"""
Main entry point for the Sports Prediction Telegram Bot
"""

import logging
import os
import asyncio
from telegram.ext import Application
from bot.telegram_bot import SportsBot
from config.settings import Settings
from data.sports_api import SportsDataCollector
from ml.predictor import MatchPredictor

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SportsPredicationApp:
    """Main application class for the Sports Prediction Bot"""
    
    def __init__(self):
        self.settings = Settings()
        self.data_collector = SportsDataCollector(self.settings)
        self.predictor = MatchPredictor()
        self.bot = None
        self.application = None
    
    async def initialize(self):
        """Initialize the application components"""
        try:
            # Initialize data collector
            await self.data_collector.initialize()
            logger.info("Sports data collector initialized")
            
            # Initialize ML predictor
            await self.predictor.initialize()
            logger.info("ML predictor initialized")
            
            # Initialize Telegram bot
            self.application = Application.builder().token(self.settings.TELEGRAM_BOT_TOKEN).build()
            self.bot = SportsBot(self.application, self.data_collector, self.predictor)
            await self.bot.setup_handlers()
            logger.info("Telegram bot initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            raise
    
    async def start(self):
        """Start the application"""
        try:
            await self.initialize()
            
            # Start the bot using polling (simpler and more reliable for development)
            if self.application:
                logger.info("Starting bot with polling...")
                await self.application.run_polling(
                    allowed_updates=['message', 'callback_query'],
                    drop_pending_updates=True
                )
                
        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            raise
    
    async def stop(self):
        """Stop the application"""
        if self.application and self.application.running:
            await self.application.stop()
        if self.data_collector:
            await self.data_collector.close()
        logger.info("Application stopped")

async def main():
    """Main function"""
    app = SportsPredicationApp()
    
    try:
        await app.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        try:
            await app.stop()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
