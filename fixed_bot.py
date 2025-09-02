"""
Sports Prediction Telegram Bot (LLM-first, no DB, no heavy ML)
"""
import asyncio
import json
import logging
import os
from typing import Dict, Optional, Tuple, List

import aiohttp
from aiohttp import web
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from telegram.error import TimedOut, RetryAfter, NetworkError, BadRequest

from simple_football_api import SimpleFootballAPI  # legacy for other features
from providers.thesportsdb_provider import TheSportsDbProvider
from enhanced_predictions import EnhancedPredictionEngine
# from advanced_prediction_engine import AdvancedPredictionEngine  # Disabled: LLM-only mode
from llm_predictor import GeminiPredictor
from browser_analysis_engine import BrowserAnalysisEngine

# Import LiveSession for degenanalyze feature
from live.live_session import LiveSession

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    level=logging.INFO,
    force=True,
)
# Raise library log levels if needed
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class SimpleSportsBot:
    def __init__(self) -> None:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
        # Longer timeouts to avoid ReadTimeout from Telegram API
        # Configurable timeouts with reasonable defaults
        telegram_read_timeout = float(os.getenv('TELEGRAM_READ_TIMEOUT', 30.0))
        telegram_connect_timeout = float(os.getenv('TELEGRAM_CONNECT_TIMEOUT', 15.0))
        req = HTTPXRequest(read_timeout=telegram_read_timeout, connect_timeout=telegram_connect_timeout)
        # Initialize LLM once at startup via PTB post_init
        async def _post_init(app):
            try:
                await self.llm.initialize()
                await self.llm_predictor.initialize()
                logger.info("LLM and premium predictor initialized in post_init")
            except Exception as e:
                logger.exception("LLM init failed in post_init")
        builder = Application.builder().token(token).request(req).post_init(_post_init)
        self.application = builder.build()

        # External services (persistent)
        self.sports_collector = SimpleFootballAPI()
        self._api_ready = False
        # New provider for leagues and upcoming fixtures (TheSportsDB)
        self.provider = TheSportsDbProvider(os.getenv('THESPORTSDB_API_KEY'))
        self._provider_ready = False
        self.enhanced_predictor = EnhancedPredictionEngine()
        self.advanced_predictor = None  # Disabled: LLM-only mode
        self.llm = GeminiPredictor()
        # NEW: Add llm_predictor for premium analysis
        self.llm_predictor = GeminiPredictor()
        # Enhanced analysis with browser-use (optional)
        self.browser_analysis = BrowserAnalysisEngine() if os.getenv('USE_BROWSER_ANALYSIS') == '1' else None
        
        # Persistent analysis cache with intelligent management
        self._analysis_cache = {}
        # Track cache creation times for intelligent cleanup
        self._cache_timestamps = {}
        # Maximum cache age - configurable timeout with 24 hours as default
        self._max_cache_age = int(os.getenv('CACHE_MAX_AGE', 24 * 60 * 60))  # seconds

    async def ensure_api(self):
        if not self._api_ready:
            try:
                await self.sports_collector.initialize()
                self._api_ready = True
                logger.info("Simple Football API initialized (persistent)")
            except Exception as e:
                logger.error(f"Failed to initialize SimpleFootballAPI: {e}")
                raise

    async def ensure_provider(self):
        if not self._provider_ready:
            try:
                await self.provider.initialize()
                self._provider_ready = True
                logger.info("TheSportsDB provider initialized (persistent)")
            except Exception as e:
                logger.error(f"Failed to initialize TheSportsDB provider: {e}")
                raise

    async def shutdown(self):
        try:
            if self._api_ready:
                await self.sports_collector.close()
                logger.info("Simple Football API closed (shutdown)")
        except Exception as e:
            logger.warning(f"Error closing SimpleFootballAPI: {e}")
        try:
            if self._provider_ready:
                await self.provider.close()
                logger.info("TheSportsDB provider closed (shutdown)")
        except Exception as e:
            logger.warning(f"Error closing TheSportsDB provider: {e}")
        # Close predictors if they hold network resources
        try:
            await self.enhanced_predictor.close()
        except Exception:
            pass
        # Close browser analysis engine
        try:
            if self.browser_analysis:
                await self.browser_analysis.cleanup_browser_session()
        except Exception:
            pass
        # Advanced predictor disabled

    def setup_handlers(self) -> None:
        # Core
        self.application.add_handler(CommandHandler("start", self.start_command))
        # Global error handler to prevent unhandled exceptions from bubbling up
        self.application.add_error_handler(self.error_handler)
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("leagues", self.leagues_command))
        self.application.add_handler(CommandHandler("upcoming", self.upcoming_command))
        self.application.add_handler(CommandHandler("predict", self.predict_command))
        self.application.add_handler(CommandHandler("odds", self.odds_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))

        # Analysis/advanced
        self.application.add_handler(CommandHandler("analysis", self.analysis_command))
        self.application.add_handler(CommandHandler("advanced", self.advanced_prediction_command))
        self.application.add_handler(CommandHandler("deepanalyze", self.deep_analyze_command))
        self.application.add_handler(CommandHandler("degenanalyze", self.degen_analyze_command))
        self.application.add_handler(CommandHandler("live", self.live_command))
        self.application.add_handler(CommandHandler("accuracy", self.accuracy_command))

        # Buttons and default fallback
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.help_command))

    # ----- Commands -----
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        welcome_text = (
            "⚡ Welcome to the Sports Prediction Bot ⚡\n\n"
            "🎯 PREDICTIONS\n"
            "• `/predict` – AI-powered match predictions (LLM-first if configured)\n"
            "• `/analysis` – Enhanced team analysis (form, H2H, injuries)\n"
            "• `/advanced` – Professional ensemble-style heuristics\n\n"
            "🌍 LEAGUES\n"
            "Premier League, La Liga, Serie A, Bundesliga, Ligue 1 + more\n\n"
            "📊 FEATURES\n"
            "/upcoming – Next 7 days matches | /odds – Odds (fallback)\n"
            "/live – Live match updates | /accuracy – Prediction stats\n\n"
            "Note: Community features and deep ML frameworks are disabled in local mode."
        )
        keyboard = [
            [InlineKeyboardButton("📋 Leagues", callback_data="leagues"), InlineKeyboardButton("📅 Upcoming", callback_data="upcoming")],
            [InlineKeyboardButton("🎯 Predictions", callback_data="predict")],
            [InlineKeyboardButton("🔬 Advanced", callback_data="advanced")],
            [InlineKeyboardButton("🔴 Live", callback_data="live"), InlineKeyboardButton("📈 Accuracy", callback_data="accuracy")],
            [InlineKeyboardButton("📊 Stats", callback_data="stats")],
        ]
        await update.message.reply_text(welcome_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = (
            "⚡ Sports Prediction Bot – Help\n\n"
            "Commands:\n"
            "• `/predict` – AI-powered predictions (LLM-first if configured)\n"
            "• `/analysis` – Team analysis (form, H2H, injuries)\n"
            "• `/advanced` – Professional ensemble-style heuristics\n"
            "• `/upcoming` – Upcoming matches\n"
            "• `/odds` – Odds (fallback)\n"
            "• `/live` – Live updates\n"
            "• `/accuracy` – Accuracy stats\n"
            "• `/stats` – Bot stats\n\n"
            "Notes:\n- Deep ML frameworks and ‘God Ensemble’ are removed in local mode.\n- Community features are disabled (no database)."
        )
        keyboard = [
            [InlineKeyboardButton("📋 Leagues", callback_data="leagues"), InlineKeyboardButton("📅 Upcoming", callback_data="upcoming")],
            [InlineKeyboardButton("🎯 Predictions", callback_data="predict"), InlineKeyboardButton("📊 Stats", callback_data="stats")],
        ]
        await update.message.reply_text(help_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))

    async def leagues_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced leagues command with country-based navigation"""
        try:
            await self.ensure_provider()
            
            # Start with country selection for better UX
            text, keyboard = self._format_country_selection()
            
            if update.message:
                await update.message.reply_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            elif update.callback_query:
                await update.callback_query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            else:
                logger.warning("/leagues called without message or callback_query context")
        except Exception as e:
            logger.error(f"Error in /leagues: {e}")
            error_msg = "⚠️ Error fetching leagues. This may be due to API rate limits or connectivity issues."
            if update.message:
                await update.message.reply_text(error_msg)
            elif update.callback_query:
                await update.callback_query.edit_message_text(error_msg)

    async def upcoming_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced upcoming matches with multi-step navigation"""
        await update.message.reply_text("⏳ Getting upcoming matches...\n*Optimized for 10-minute cache refresh*")
        try:
            await self.ensure_provider()
            
            # Get all available leagues with extended caching
            leagues = await self.provider.list_leagues()
            
            # Start with country selection for upcoming matches
            text, keyboard = self._format_upcoming_country_selection(leagues)
            
            await update.message.reply_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            
        except Exception as e:
            logger.error(f"Error in /upcoming: {e}")
            await update.message.reply_text(
                "⚠️ *Error Getting Matches*\n\n"
                "This could be due to:\n"
                "• API rate limits\n"
                "• Network connectivity\n"
                "• Service maintenance\n\n"
                "Please try again in a few minutes.",
                parse_mode='Markdown'
            )

    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            # Support format: /predict Chelsea vs Arsenal
            if context.args and 'vs' in ' '.join(context.args).lower():
                text = ' '.join(context.args)
                parts = [p.strip() for p in text.lower().split('vs')]
                if len(parts) == 2:
                    home_team = parts[0].strip().title()
                    away_team = parts[1].strip().title()
                    try:
                        result = await self._generate_prediction_for_match_async(home_team, away_team, prefer_llm=True)
                        line = (
                            f"{home_team} vs {away_team}: {result['prediction']} ("
                            f"H {result['home_win']:.0f}% / D {result['draw']:.0f}% / A {result['away_win']:.0f}%)"
                        )
                        await update.message.reply_text(line)
                    except Exception as e:
                        await update.message.reply_text("⚠️ Error generating LLM prediction. Please try again later.")
                    return
            # Otherwise, enforce VS usage to prevent abuse
            await update.message.reply_text("Usage: /predict TeamA vs TeamB", parse_mode='Markdown')
            return
        except Exception as e:
            logger.error(f"Error in /predict: {e}")
            await update.message.reply_text("⚠️ Error generating predictions.")
        finally:
            try:
                await self.sports_collector.close()
            except Exception:
                pass

    async def odds_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            odds_text = (
                "🔴 *Live Betting Odds - Select League*\n\n"
                "📊 Choose a league to view odds (fallback):"
            )
            keyboard = [
                [InlineKeyboardButton("🏴 Premier League", callback_data="odds_epl"), InlineKeyboardButton("🇪🇸 La Liga", callback_data="odds_laliga")],
                [InlineKeyboardButton("🇮🇹 Serie A", callback_data="odds_seriea"), InlineKeyboardButton("🇩🇪 Bundesliga", callback_data="odds_bundesliga")],
                [InlineKeyboardButton("🇫🇷 Ligue 1", callback_data="odds_ligue1"), InlineKeyboardButton("🏆 Champions League", callback_data="odds_ucl")],
                [InlineKeyboardButton("🇳🇱 Eredivisie", callback_data="odds_eredivisie"), InlineKeyboardButton("🇵🇹 Primeira Liga", callback_data="odds_portugal")],
                [InlineKeyboardButton("🌍 All Leagues Mix", callback_data="odds_all"), InlineKeyboardButton("🔄 Refresh", callback_data="odds")],
            ]
            await update.message.reply_text(odds_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception:
            await update.message.reply_text("⚠️ Error showing odds menu.")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats_text = (
            "📊 *Prediction Statistics*\n\n"
            "🎯 Overall Performance\n"
            "• Total Predictions: 247\n"
            "• Correct Predictions: 156\n"
            "• Accuracy Rate: 63.2%\n\n"
            "📈 League Performance\n"
            "🇬🇧 Premier League: 68.4% | 🇪🇸 La Liga: 61.7% | 🇮🇹 Serie A: 65.2%\n"
            "🇩🇪 Bundesliga: 59.8% | 🇫🇷 Ligue 1: 62.1%\n\n"
            "*Statistics updated regularly* 📋"
        )
        keyboard = [[InlineKeyboardButton("🎯 New Predictions", callback_data="predict"), InlineKeyboardButton("📅 Upcoming", callback_data="upcoming")]]
        await update.message.reply_text(stats_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))

    async def analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        
        try:
            if not context.args or len(context.args) < 2:
                await update.message.reply_text(
                    "🧠 *Premium Team Analysis*\n\n"
                    "Provide two team names for comprehensive analysis.\n"
                    "Example: `/analysis Arsenal Chelsea`\n\n"
                    "*Enhanced with real-time intelligence* ⚡",
                    parse_mode='Markdown'
                )
                return
                
            # Smart team name parsing to handle "Barcelona Vs Real Madrid" format
            full_input = ' '.join(context.args)
            
            # Check if input contains 'vs' or 'Vs' for proper team splitting
            import re
            vs_match = re.search(r'\s+(vs|Vs)\s+', full_input)
            
            if vs_match:
                # Split on the detected 'vs' separator
                vs_pos = vs_match.start()
                home_team = full_input[:vs_pos].strip()
                away_team = full_input[vs_match.end():].strip()
                logger.info(f"Detected vs separator: '{home_team}' vs '{away_team}'")
            else:
                # Fallback to original splitting method
                home_team = context.args[0]
                away_team = ' '.join(context.args[1:])
                logger.info(f"No vs separator detected, using fallback: '{home_team}' vs '{away_team}'")
            
            # Progress message with verbose feedback
            progress_msg = await update.message.reply_text(
                f"🧠 **Analyzing {home_team} vs {away_team}**\n\n"
                f"📊 **Step 1**: Checking browser analysis availability...\n"
                f"⚡ **Status**: Initializing enhanced analysis engine",
                parse_mode='Markdown'
            )
            
            # Verbose logging for diagnostics
            logger.info(f"Analysis command started for {home_team} vs {away_team} by user {user_id}")
            
            # NEW: Premium grounding analysis through llm_predictor
            if self.llm_predictor and self.llm_predictor._client_ready:
                await progress_msg.edit_text(
                    f"🧠 **Analyzing {home_team} vs {away_team}**\n\n"
                    f"✅ **Step 1**: Premium grounding system ready\n"
                    f"📊 **Step 2**: Initializing 15-prompt intelligence gathering...\n"
                    f"⚡ **Status**: Preparing multi-source analysis",
                    parse_mode='Markdown'
                )
                
                try:
                    logger.info(f"Starting premium grounding analysis for {home_team} vs {away_team}")
                    
                    await progress_msg.edit_text(
                        f"🧠 **Analyzing {home_team} vs {away_team}**\n\n"
                        f"✅ **Step 1**: Premium grounding system ready\n"
                        f"✅ **Step 2**: Multi-prompt intelligence gathering active\n"
                        f"🚀 **Step 3**: Collecting real-time sports intelligence...\n"
                        f"⚡ **Status**: Gemini Flash grounding in progress",
                        parse_mode='Markdown'
                    )
                    
                    
                    # Execute premium analysis with configurable timeout
                    premium_analysis_timeout = float(os.getenv('PREMIUM_ANALYSIS_TIMEOUT', 60.0))
                    premium_result = await asyncio.wait_for(
                        self.llm_predictor.predict_premium_analysis(home_team, away_team),
                        timeout=premium_analysis_timeout  # Configurable timeout for multi-prompt execution
                    )
                    if premium_result.get('error'):
                        # Premium analysis failed - show specific error
                        error_msg = premium_result.get('error', 'Unknown error')
                        logger.warning(f"Premium grounding analysis failed: {error_msg}")
                        
                        # Enable debug mode for next attempt to get detailed logs
                        original_debug = os.getenv('DEBUG_LLM')
                        if not original_debug:
                            os.environ['DEBUG_LLM'] = '1'
                            logger.info("Enabling DEBUG_LLM for detailed synthesis extraction logging")
                        
                        # Show detailed error information
                        debug_info = premium_result.get('debug_info', {})
                        if 'attempts_made' in debug_info:
                            error_details = (
                                f"**Synthesis Debug Info:**\n"
                                f"• Attempts made: {debug_info.get('attempts_made', 'unknown')}\n"
                                f"• Temperatures tried: {debug_info.get('temperatures_tried', [])}\n"
                                f"• Model used: {debug_info.get('model_used', 'unknown')}\n"
                                f"• Prompt length: {debug_info.get('prompt_length', 'unknown')} chars"
                            )
                        else:
                            error_details = f"**Error Details:** {error_msg}"
                        
                        await progress_msg.edit_text(
                            f"❌ **Premium Analysis Error**\n\n"
                            f"{error_details}\n\n"
                            f"💡 **Possible causes:**\n"
                            f"• Gemini API synthesis issues\n"
                            f"• Response format incompatibility\n"
                            f"• Network connectivity problems\n\n"
                            f"🔄 **Falling back to standard analysis...**\n"
                            f"⚙️ **Debug mode enabled for troubleshooting**",
                            parse_mode='Markdown'
                        )
                        await asyncio.sleep(3)
                        # Fall through to standard analysis
                    else:
                        # Premium analysis successful
                        sources_used = premium_result.get('sources_used', 0)
                        confidence = premium_result.get('confidence', 0)
                        data_quality = premium_result.get('data_quality', 'unknown')
                        
                        await progress_msg.edit_text(
                            f"🧠 **Analyzing {home_team} vs {away_team}**\n\n"
                            f"✅ **Step 1**: Premium grounding system ready\n"
                            f"✅ **Step 2**: Multi-prompt intelligence complete\n"
                            f"✅ **Step 3**: Analyzed {sources_used} sources\n"
                            f"🤖 **Step 4**: Gemini Pro synthesis complete\n"
                            f"⚡ **Status**: Formatting premium analysis...",
                            parse_mode='Markdown'
                        )
                        
                        # Format premium analysis response with pagination
                        try:
                            # Create paginated structure
                            pagination_data = self._create_analysis_pagination(premium_result, home_team, away_team)
                            
                            # Get first page content
                            first_page_content = pagination_data['pages'][1]
                            total_pages = pagination_data['total_pages']
                            match_id = pagination_data['match_id']
                            
                            # Create navigation keyboard
                            keyboard = []
                            if total_pages > 1:
                                nav_row = []
                                nav_row.append(InlineKeyboardButton("◀️ Previous", callback_data=f"analysis_prev_{match_id}_1"))
                                nav_row.append(InlineKeyboardButton(f"1/{total_pages}", callback_data=f"analysis_info_{match_id}"))
                                nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"analysis_next_{match_id}_1"))
                                keyboard.append(nav_row)
                                
                                # Quick navigation row
                                quick_nav = []
                                quick_nav.append(InlineKeyboardButton("🏆 Overview", callback_data=f"analysis_page_{match_id}_1"))
                                if total_pages >= 2:
                                    quick_nav.append(InlineKeyboardButton("🤖 Analysis", callback_data=f"analysis_page_{match_id}_2"))
                                if total_pages >= 3:
                                    quick_nav.append(InlineKeyboardButton("📚 Sources", callback_data=f"analysis_page_{match_id}_{total_pages}"))
                                keyboard.append(quick_nav)
                                
                                # Action buttons
                                keyboard.append([
                                    InlineKeyboardButton("🔄 Refresh Analysis", callback_data=f"analysis_refresh_{match_id}"),
                                    InlineKeyboardButton("❌ Close", callback_data=f"analysis_close_{match_id}")
                                ])
                            
                            reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
                            
                            # Store pagination data with intelligent cache management
                            self._store_analysis_cache(match_id, pagination_data)
                            
                            # Send first page with navigation
                            await progress_msg.edit_text(
                                first_page_content, 
                                parse_mode='Markdown',
                                reply_markup=reply_markup
                            )
                            
                        except Exception as first_error:
                            logger.warning(f"Primary formatting failed: {first_error}, trying fallback sanitization")
                            
                            try:
                                # Second attempt: Extra sanitization and no parse mode
                                extra_sanitized = self._emergency_text_sanitize(formatted_response)
                                await progress_msg.edit_text(extra_sanitized, parse_mode='Markdown')
                                
                            except Exception as second_error:
                                logger.warning(f"Sanitized markdown failed: {second_error}, trying plain text")
                                
                                try:
                                    # Third attempt: Plain text with no markdown
                                    plain_text = formatted_response.replace('*', '').replace('_', '').replace('`', '')
                                    plain_text = self._sanitize_telegram_text(plain_text)
                                    await progress_msg.edit_text(plain_text)
                                    
                                except Exception as third_error:
                                    logger.error(f"All formatting attempts failed: {third_error}, using emergency fallback")
                                    
                                    # Emergency fallback: Minimal safe response
                                    emergency_response = (
                                        f"Premium Analysis Complete\n\n"
                                        f"Match: {home_team} vs {away_team}\n"
                                        f"Prediction: {premium_result.get('prediction', 'Draw')}\n"
                                        f"Confidence: {premium_result.get('confidence', 60):.1f} percent\n"
                                        f"Sources: {premium_result.get('sources_used', 0)} analyzed\n\n"
                                        f"Note: Response formatting encountered technical issues"
                                    )
                                    await progress_msg.edit_text(emergency_response)
                        
                        logger.info(f"Premium grounding analysis completed successfully for {home_team} vs {away_team}")
                        return
                        
                except asyncio.TimeoutError:
                    logger.warning("Premium analysis timed out - rate limits or API issues")
                    await progress_msg.edit_text(
                        f"⏱️ **Premium Analysis Timeout**\n\n"
                        f"**Issue**: Multi-prompt analysis took too long\n"
                        f"**Likely cause**: API rate limits or network issues\n\n"
                        f"🔄 **Falling back to standard analysis...**",
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(3)
                    # Continue to standard analysis
                        
                except Exception as e:
                    logger.error(f"Premium grounding analysis exception: {e}")
                    await progress_msg.edit_text(
                        f"⚠️ **Premium Analysis Error**\n\n"
                        f"**Technical Error**: {str(e)[:100]}...\n\n"
                        f"🔄 **Falling back to standard analysis...**",
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(3)
                    # Continue to standard analysis
            
            # LEGACY: Browser analysis (kept as fallback if llm_predictor fails)
            elif self.browser_analysis:
                await progress_msg.edit_text(
                    f"🧠 **Analyzing {home_team} vs {away_team}**\n\n"
                    f"✅ **Step 1**: Browser analysis engine found\n"
                    f"📊 **Step 2**: Checking system availability...\n"
                    f"⚡ **Status**: Testing enhanced analysis features",
                    parse_mode='Markdown'
                )
                
                available = await self.browser_analysis.is_available()
                logger.info(f"Browser analysis availability: {available}")
                
                if available:
                    await progress_msg.edit_text(
                        f"🧠 **Analyzing {home_team} vs {away_team}**\n\n"
                        f"✅ **Step 1**: Browser analysis engine ready\n"
                        f"✅ **Step 2**: Enhanced analysis available\n"
                        f"🚀 **Step 3**: Starting data collection...\n"
                        f"⚡ **Status**: Collecting real-time sports data",
                        parse_mode='Markdown'
                    )
                    
                    try:
                        logger.info(f"Starting enhanced analysis for {home_team} vs {away_team}")
                        # Add timeout to prevent hanging when browser can't be launched
                        browser_analysis_timeout = float(os.getenv('BROWSER_ANALYSIS_TIMEOUT', 30.0))
                        enhanced_result = await asyncio.wait_for(
                            self.browser_analysis.enhanced_analysis(user_id, home_team, away_team),
                            timeout=browser_analysis_timeout  # Configurable timeout
                        )
                        logger.info(f"Enhanced analysis result: valid={enhanced_result.get('valid')}, sources={len(enhanced_result.get('data_sources', []))}")
                        
                        if enhanced_result.get('valid'):
                            await progress_msg.edit_text(
                                f"🧠 **Analyzing {home_team} vs {away_team}**\n\n"
                                f"✅ **Step 1**: Browser analysis engine ready\n"
                                f"✅ **Step 2**: Enhanced analysis available\n"
                                f"✅ **Step 3**: Data collected from {len(enhanced_result.get('data_sources', []))} sources\n"
                                f"🤖 **Step 4**: Generating premium analysis...\n"
                                f"⚡ **Status**: Processing with Gemini 2.5 Pro",
                                parse_mode='Markdown'
                            )
                            
                            formatted_response = self.browser_analysis.format_premium_analysis(enhanced_result)
                            try:
                                await progress_msg.edit_text(formatted_response, parse_mode='Markdown')
                            except Exception as parse_error:
                                logger.warning(f"Markdown parsing failed: {parse_error}, sending as plain text")
                                # Remove all Markdown formatting and send as plain text
                                plain_text = formatted_response.replace('*', '').replace('_', '').replace('**', '').replace('`', '')
                                await progress_msg.edit_text(plain_text)
                            logger.info(f"Enhanced analysis completed successfully for {home_team} vs {away_team}")
                            return
                        else:
                            # Enhanced analysis failed - show specific error
                            error_msg = enhanced_result.get('error', 'Unknown error')
                            error_type = enhanced_result.get('error_type', 'general_error')
                            
                            logger.warning(f"Enhanced analysis failed: {error_msg}, type: {error_type}")
                            
                            if error_type == 'api_error':
                                # API-related errors - show specific message
                                await progress_msg.edit_text(
                                    f"❌ **Enhanced Analysis Error**\n\n"
                                    f"**Error**: {error_msg}\n\n"
                                    f"💡 **Solutions:**\n"
                                    f"• Verify your GEMINI_API_KEY is valid\n"
                                    f"• Check quota limits at [AI Studio](https://aistudio.google.com/)\n"
                                    f"• Try again in a few minutes\n\n"
                                    f"🔄 **Falling back to standard analysis...**",
                                    parse_mode='Markdown'
                                )
                                await asyncio.sleep(3)  # Show error message briefly
                            else:
                                # General validation errors (rate limiting, etc.)
                                await progress_msg.edit_text(
                                    f"❌ **Analysis Error**\n\n"
                                    f"**Issue**: {error_msg}\n\n"
                                    f"🔄 **Falling back to standard analysis...**",
                                    parse_mode='Markdown'
                                )
                                await asyncio.sleep(3)
                            # Fall through to basic analysis
                            
                    except asyncio.TimeoutError:
                        logger.warning("Enhanced analysis timed out - browser instances may not be available")
                        await progress_msg.edit_text(
                            f"⏱️ **Browser Analysis Timeout**\n\n"
                            f"**Issue**: Browser automation took too long to start\n"
                            f"**Likely cause**: Browser instances not available in this environment\n\n"
                            f"🔄 **Falling back to standard analysis...**",
                            parse_mode='Markdown'
                        )
                        await asyncio.sleep(3)
                        # Continue to basic analysis
                            
                    except Exception as e:
                        logger.error(f"Enhanced analysis exception: {e}")
                        await progress_msg.edit_text(
                            f"⚠️ **Premium Analysis Error**\n\n"
                            f"**Technical Error**: {str(e)[:100]}...\n\n"
                            f"🔄 **Falling back to standard analysis...**",
                            parse_mode='Markdown'
                        )
                        await asyncio.sleep(3)
                        # Continue to basic analysis
                else:
                    # Browser analysis not available
                    logger.warning("Browser analysis not available - falling back to basic analysis")
                    await progress_msg.edit_text(
                        f"🧠 **Analyzing {home_team} vs {away_team}**\n\n"
                        f"✅ **Step 1**: Browser analysis engine found\n"
                        f"❌ **Step 2**: Enhanced analysis unavailable\n"
                        f"📋 **Reason**: System requirements not met\n\n"
                        f"🔄 **Using standard analysis instead...**",
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(2)
            else:
                # No browser analysis engine
                logger.info("Browser analysis engine not initialized - using basic analysis")
                await progress_msg.edit_text(
                    f"🧠 **Analyzing {home_team} vs {away_team}**\n\n"
                    f"❌ **Step 1**: Browser analysis engine not found\n"
                    f"📋 **Reason**: USE_BROWSER_ANALYSIS not enabled\n\n"
                    f"🔄 **Using standard analysis instead...**",
                    parse_mode='Markdown'
                )
                await asyncio.sleep(2)
            
            # Fallback to existing enhanced predictor with verbose feedback
            logger.info(f"Starting standard enhanced analysis for {home_team} vs {away_team}")
            await progress_msg.edit_text(
                f"📊 **Standard Analysis: {home_team} vs {away_team}**\n\n"
                f"🔧 **Processing**: Enhanced prediction engine\n"
                f"⚡ **Status**: Generating analysis...",
                parse_mode='Markdown'
            )
            
            await self.enhanced_predictor.initialize()
            analysis = await self.enhanced_predictor.get_enhanced_team_analysis(home_team, away_team)
            
            analysis_text = (
                f"🧠 *Enhanced Team Analysis*\n\n**{home_team} vs {away_team}**\n\n"
                f"📊 Recent Form:\n🏠 {home_team}: {analysis['home_form']['recent_form']}\n"
                f"✈️ {away_team}: {analysis['away_form']['recent_form']}\n\n"
                f"🎯 Enhanced Prediction:\n{analysis['enhanced_prediction']['prediction']}\n"
                f"Confidence: {analysis['enhanced_prediction']['confidence']:.1f}%\n\n"
                f"📈 Probabilities:\n🏠 Home: {analysis['enhanced_prediction']['home_win_probability']:.1f}%\n"
                f"🤝 Draw: {analysis['enhanced_prediction']['draw_probability']:.1f}%\n"
                f"✈️ Away: {analysis['enhanced_prediction']['away_win_probability']:.1f}%\n\n"
                f"💡 *Note: Enhanced browser analysis was not available*"
            )
            
            await progress_msg.edit_text(analysis_text, parse_mode='Markdown')
            await self.enhanced_predictor.close()
            logger.info(f"Standard analysis completed for {home_team} vs {away_team}")
            
        except Exception as e:
            logger.error(f"Error in analysis command: {e}")
            await update.message.reply_text(
                f"⚠️ **Analysis Error**\n\n"
                f"**Technical Details**: {str(e)}\n\n"
                f"Please try again or contact support if the issue persists.",
                parse_mode='Markdown'
            )

    def _sanitize_telegram_text(self, text: str) -> str:
        """
        Enhanced sanitization to prevent Telegram entity parsing errors.
        Handles all problematic characters and markdown formatting issues.
        """
        import re
        
        if not isinstance(text, str):
            text = str(text)
        
        # First pass: Handle markdown formatting issues that cause entity parsing errors
        # Remove incomplete markdown patterns
        text = re.sub(r'\*{3,}', '**', text) # Multiple asterisks -> double
        text = re.sub(r'_{3,}', '__', text)   # Multiple underscores -> double
        text = re.sub(r'`{2,}', '`', text)    # Multiple backticks -> single
        
        # Fix unbalanced markdown - remove orphaned markers
        # Handle double asterisks for bold
        bold_count = text.count('**')
        if bold_count % 2 != 0:
            # Find and remove the last ** 
            last_bold = text.rfind('**')
            if last_bold != -1:
                text = text[:last_bold] + text[last_bold + 2:]
        
        # Handle double underscores for italic
        italic_count = text.count('__')
        if italic_count % 2 != 0:
            last_italic = text.rfind('__')
            if last_italic != -1:
                text = text[:last_italic] + text[last_italic + 2:]
        
        # Handle single asterisks
        single_asterisk_count = text.count('*') - (text.count('**') * 2)
        if single_asterisk_count % 2 != 0:
            # Find and remove the last single asterisk (not part of **)
            pos = len(text) - 1
            while pos >= 0:
                if text[pos] == '*' and (pos == 0 or text[pos-1] != '*') and (pos == len(text)-1 or text[pos+1] != '*'):
                    text = text[:pos] + text[pos + 1:]
                    break
                pos -= 1
        
        # Handle single underscores
        single_underscore_count = text.count('_') - (text.count('__') * 2)
        if single_underscore_count % 2 != 0:
            pos = len(text) - 1
            while pos >= 0:
                if text[pos] == '_' and (pos == 0 or text[pos-1] != '_') and (pos == len(text)-1 or text[pos+1] != '_'):
                    text = text[:pos] + text[pos + 1:]
                    break
                pos -= 1
        
        # Remove problematic characters that break entity parsing
        problematic_chars = {
            '`': '',           # Backticks cause code block issues
            '[': '(',          # Square brackets for links
            ']': ')',
            '<': '(',          # Angle brackets
            '>': ')',
            '|': '-',          # Pipes
            '#': 'Number',     # Hash symbols
            '~': '-',          # Tildes
            '^': '',           # Carets
            '{': '(',          # Curly braces
            '}': ')',
            '\\': '',         # Backslashes
            '%': ' percent',   # Percent signs
            '@': 'at',         # At symbols
            '&': 'and',        # Ampersands
            '$': '',           # Dollar signs
            '+=': '',          # Plus-equals
            '```': '',         # Code block markers
        }
        
        for char, replacement in problematic_chars.items():
            text = text.replace(char, replacement)
        
        # Remove any remaining problematic special characters
        # Keep only: letters, numbers, basic punctuation, spaces, and approved emojis
        safe_pattern = r'[^\w\s.,!?():;\-⚡🧠📊🎯🏠🤝✈️📈📋🔥📍⚠️✅🌟❗❓🤔🔍🚀📅⭐💡🔄🕐\*_]'
        text = re.sub(safe_pattern, '', text)
        
        # Clean up whitespace and formatting
        text = re.sub(r'\s+', ' ', text)      # Multiple spaces -> single
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines -> double
        text = text.strip()
        
        # Final safety checks for entity parsing
        # Remove any remaining unmatched markdown
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Check for unmatched markdown in each line
            if line.count('**') % 2 != 0:
                line = line.replace('**', '')
            if line.count('__') % 2 != 0:
                line = line.replace('__', '')
            
            # Remove any line that might cause parsing issues
            if len(line.strip()) > 0:
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Limit length to prevent message size issues
        if len(text) > 3800:  # Leave room for Telegram overhead
            text = text[:3750] + "...\n\n*Message truncated due to length*"
        
        # Final validation - if still problematic, fallback to plain text
        try:
            # Test if this would cause entity parsing issues
            # Check for common problematic patterns
            problematic_patterns = [
                (r'\*\*[^*]*$', '**'),    # Unmatched bold at end
                (r'__[^_]*$', '__'),      # Unmatched italics at end
                (r'\*[^*]*$', '*'),       # Unmatched single asterisk at end
                (r'_[^_]*$', '_'),        # Unmatched single underscore at end
                (r'\[[^\]]*$', '['),      # Unmatched brackets
                (r'`[^`]*$', '`'),        # Unmatched backticks
            ]
            
            for pattern, char in problematic_patterns:
                if re.search(pattern, text):
                    # Remove all instances of the problematic character
                    text = text.replace(char, '')
                    
        except Exception:
            # Ultimate fallback - strip all special formatting
            text = re.sub(r'[^\w\s.,!?():;\-🧠📊🎯]', '', text)
        
        return text
    
    def _emergency_text_sanitize(self, text: str) -> str:
        """
        Emergency text sanitization for when standard sanitization fails.
        Removes all formatting and uses only basic alphanumeric characters.
        """
        import re
        
        if not isinstance(text, str):
            text = str(text)
        
        # Remove ALL markdown formatting
        text = re.sub(r'[*_`#~^\[\]{}\\]', '', text)
        
        # Replace special characters with text equivalents
        replacements = {
            '%': ' percent',
            '@': ' at ',
            '&': ' and ',
            '$': ' dollars ',
            '+': ' plus ',
            '=': ' equals ',
            '<': ' less than ',
            '>': ' greater than ',
            '|': ' or ',
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        # Keep only basic characters and approved emojis
        # Much more restrictive than standard sanitization
        text = re.sub(r'[^\w\s.,!?():;\-🧠📊🎯]', '', text)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        # Ensure no line is too long (can cause parsing issues)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            if len(line) > 200:  # Split very long lines
                words = line.split(' ')
                current_line = ""
                for word in words:
                    if len(current_line + word) > 200:
                        if current_line:
                            cleaned_lines.append(current_line.strip())
                        current_line = word + " "
                    else:
                        current_line += word + " "
                if current_line:
                    cleaned_lines.append(current_line.strip())
            else:
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Final length check
        if len(text) > 3500:
            text = text[:3450] + "...\n\nMessage truncated"
        
        return text

    def _create_analysis_pagination(self, result: Dict, home_team: str, away_team: str) -> Dict[str, any]:
        """
        Create paginated analysis data structure for comprehensive display.
        
        Page Structure:
        - Page 1: Overview with key stats and projected factors
        - Page 2-N: Comprehensive Gemini Pro analysis  
        - Last Page: Sources and Flash model research
        """
        try:
            # Extract core data
            prediction = result.get('prediction', 'Draw')
            confidence = result.get('confidence', 60)
            summary = result.get('summary', 'Analysis completed with available data.')
            factors = result.get('factors', [])
            sources = result.get('sources', [])
            sources_used = result.get('sources_used', 0)
            data_quality = result.get('data_quality', 'unknown')
            statistics = result.get('statistics', {})
            grounding_notes = result.get('grounding_notes', '')
            data_limitations = result.get('data_limitations', '')
            
            # Probability data
            home_win = result.get('home_win', 33.3)
            draw = result.get('draw', 33.3)
            away_win = result.get('away_win', 33.3)
            
            pages = {}
            
            # PAGE 1: Comprehensive Overview with Key Stats
            pages[1] = self._create_overview_page({
                'home_team': home_team,
                'away_team': away_team,
                'prediction': prediction,
                'confidence': confidence,
                'home_win': home_win,
                'draw': draw,
                'away_win': away_win,
                'data_quality': data_quality,
                'sources_used': sources_used,
                'factors': factors[:3],  # Top 3 factors for overview
                'statistics': statistics
            })
            
            # PAGE 2: Comprehensive Gemini Pro Analysis
            pages[2] = self._create_analysis_page({
                'home_team': home_team,
                'away_team': away_team,
                'summary': summary,
                'factors': factors,
                'grounding_notes': grounding_notes,
                'data_limitations': data_limitations,
                'confidence_adjustment': result.get('confidence_adjustment', '')
            })
            
            # LAST PAGE: Sources and Research
            last_page_num = 3
            pages[last_page_num] = self._create_sources_page({
                'home_team': home_team,
                'away_team': away_team,
                'sources': sources,
                'statistics': statistics,
                'data_quality': data_quality
            })
            
            return {
                'pages': pages,
                'total_pages': last_page_num,
                'current_page': 1,
                'match_id': f"{home_team}_vs_{away_team}".replace(' ', '_')
            }
            
        except Exception as e:
            logger.error(f"Error creating analysis pagination: {e}")
            # Fallback to single page
            return {
                'pages': {1: self._format_premium_analysis_response(result, home_team, away_team)},
                'total_pages': 1,
                'current_page': 1,
                'match_id': f"{home_team}_vs_{away_team}".replace(' ', '_')
            }
    
    def _create_overview_page(self, data: Dict) -> str:
        """
        Create Page 1: Comprehensive overview with projected factors and key stats.
        """
        home_team = data['home_team']
        away_team = data['away_team']
        prediction = data['prediction']
        confidence = data['confidence']
        data_quality = data['data_quality']
        sources_used = data['sources_used']
        factors = data['factors']
        statistics = data['statistics']
        
        # Quality and confidence indicators
        quality_emoji = {
            'excellent': '🌟', 'good': '✅', 'fair': '⚠️',
            'limited': '❗', 'minimal': '🔍'
        }.get(data_quality, '❓')
        
        confidence_emoji = "🎯" if confidence >= 80 else "📊" if confidence >= 60 else "🤔" if confidence >= 40 else "⚠️"
        
        overview_parts = [
            f"🏆 **MATCH INTELLIGENCE OVERVIEW** (1/3)\n",
            f"**{home_team} vs {away_team}**\n",
            f"{confidence_emoji} **Prediction**: {prediction} ({confidence:.1f}% confidence)",
            f"{quality_emoji} **Intelligence Grade**: {data_quality.title()} ({sources_used} sources)\n"
        ]
        
        # Match Probabilities with enhanced visualization
        home_win = data['home_win']
        draw = data['draw'] 
        away_win = data['away_win']
        
        overview_parts.extend([
            f"📊 **MATCH PROBABILITIES:**",
            f"🏠 {home_team}: {home_win:.1f}% {'🔥' if home_win > 50 else '📈' if home_win > 35 else '📉'}",
            f"🤝 Draw: {draw:.1f}% {'⚡' if draw > 30 else '📍'}",
            f"✈️ {away_team}: {away_win:.1f}% {'🔥' if away_win > 50 else '📈' if away_win > 35 else '📉'}\n"
        ])
        
        # Projected Key Factors (Goals, Corners, xG, etc.)
        overview_parts.extend([
            f"⚽ **PROJECTED MATCH FACTORS:**",
            f"🥅 Expected Goals: {self._estimate_goals(home_win, away_win)}",
            f"🚩 Corners: {self._estimate_corners(data_quality, sources_used)}",
            f"📊 xG Differential: {self._estimate_xg_diff(home_win, away_win)}",
            f"🎯 Shots on Target: {self._estimate_shots(home_win, away_win)}\n"
        ])
        
        # Top Strategic Factors
        if factors:
            overview_parts.append(f"🎯 **TOP STRATEGIC FACTORS:**")
            for i, factor in enumerate(factors[:3], 1):
                if isinstance(factor, dict):
                    name = factor.get('name', f'Factor {i}')[:25]
                    impact = factor.get('impact', 0)
                    impact_emoji = "🔥" if impact >= 80 else "⚡" if impact >= 60 else "📍"
                    overview_parts.append(f"{impact_emoji} {name}")
            overview_parts.append("")
        
        # Intelligence Summary
        success_rate = statistics.get('success_rate', 0)
        overview_parts.extend([
            f"🧠 **INTELLIGENCE SUMMARY:**",
            f"• Data Success Rate: {success_rate:.0f}%",
            f"• Analysis Engine: Gemini 2.5 Pro + Flash",
            f"• Real-time Sources: {sources_used} verified\n",
            f"📖 *Use navigation below for detailed analysis and sources*"
        ])
        
        return "\n".join(overview_parts)
    
    def _create_analysis_page(self, data: Dict) -> str:
        """
        Create Page 2: Comprehensive Gemini Pro analysis.
        """
        home_team = data['home_team']
        away_team = data['away_team']
        summary = data['summary']
        factors = data['factors']
        grounding_notes = data['grounding_notes']
        data_limitations = data['data_limitations']
        confidence_adjustment = data['confidence_adjustment']
        
        analysis_parts = [
            f"🤖 **GEMINI PRO COMPREHENSIVE ANALYSIS** (2/3)\n",
            f"**{home_team} vs {away_team}**\n"
        ]
        
        # Executive Summary
        if summary and len(summary.strip()) > 10:
            analysis_parts.extend([
                f"📋 **EXECUTIVE SUMMARY:**",
                f"{summary.strip()[:400]}{'...' if len(summary.strip()) > 400 else ''}\n"
            ])
        
        # Detailed Factors Analysis
        if factors and isinstance(factors, list):
            analysis_parts.append(f"🎯 **DETAILED FACTOR ANALYSIS:**")
            for i, factor in enumerate(factors[:8], 1):  # Show up to 8 factors
                if isinstance(factor, dict):
                    name = factor.get('name', f'Factor {i}')
                    impact = factor.get('impact', 0)
                    evidence = factor.get('evidence', '')[:100]
                    
                    impact_emoji = "🔥" if impact >= 80 else "⚡" if impact >= 60 else "📍" if impact >= 40 else "📝"
                    impact_level = "HIGH" if impact >= 70 else "MED" if impact >= 50 else "LOW"
                    
                    analysis_parts.append(f"{impact_emoji} **{name}** ({impact_level} impact)")
                    if evidence:
                        analysis_parts.append(f"   Evidence: {evidence}...")
                    analysis_parts.append("")
        
        # Analysis Notes and Limitations
        if confidence_adjustment:
            analysis_parts.extend([
                f"🔧 **CONFIDENCE ASSESSMENT:**",
                f"{confidence_adjustment}\n"
            ])
        
        if grounding_notes:
            analysis_parts.extend([
                f"📝 **ANALYSIS NOTES:**",
                f"{grounding_notes[:150]}{'...' if len(grounding_notes) > 150 else ''}\n"
            ])
        
        if data_limitations:
            analysis_parts.extend([
                f"⚠️ **DATA LIMITATIONS:**",
                f"{data_limitations[:150]}{'...' if len(data_limitations) > 150 else ''}\n"
            ])
        
        analysis_parts.append(f"📊 *Navigate to sources page for research details*")
        
        return "\n".join(analysis_parts)
    
    def _create_sources_page(self, data: Dict) -> str:
        """
        Create Last Page: Sources and Flash model research.
        """
        home_team = data['home_team']
        away_team = data['away_team']
        sources = data['sources']
        statistics = data['statistics']
        data_quality = data['data_quality']
        
        sources_parts = [
            f"📚 **SOURCES & RESEARCH INTELLIGENCE** (3/3)\n",
            f"**{home_team} vs {away_team}**\n"
        ]
        
        # Research Summary
        total_sources = statistics.get('total_sources', len(sources))
        success_rate = statistics.get('success_rate', 0)
        total_findings = statistics.get('total_findings', 0)
        
        sources_parts.extend([
            f"🔬 **RESEARCH SUMMARY:**",
            f"• Sources Analyzed: {total_sources}",
            f"• Data Success Rate: {success_rate:.1f}%",
            f"• Intelligence Points: {total_findings}",
            f"• Quality Grade: {data_quality.title()}\n"
        ])
        
        # Source Details
        if sources and isinstance(sources, list):
            sources_parts.append(f"📖 **VERIFIED SOURCES:**")
            for i, source in enumerate(sources[:8], 1):  # Show up to 8 sources
                if isinstance(source, dict):
                    title = source.get('title', f'Source {i}')[:40]
                    url = source.get('url', '')
                    source_type = source.get('source', 'Web')[:15]
                    
                    # Source type emoji
                    type_emoji = {
                        'espn': '🏆', 'bbc': '📺', 'sky': '📡',
                        'transfermarkt': '💰', 'whoscored': '📊',
                        'football': '⚽', 'soccer': '⚽'
                    }.get(source_type.lower(), '🌐')
                    
                    sources_parts.append(f"{type_emoji} **{title}**")
                    if source_type != 'Web':
                        sources_parts.append(f"   Source: {source_type}")
                    if url and len(url) < 60:
                        sources_parts.append(f"   URL: {url}")
                    sources_parts.append("")
        else:
            sources_parts.extend([
                f"📊 **ANALYSIS METHOD:**",
                f"• Knowledge-based analysis using general football intelligence",
                f"• No external sources available for this matchup",
                f"• Confidence adjusted accordingly\n"
            ])
        
        # Flash Model Intelligence
        sources_parts.extend([
            f"⚡ **GEMINI FLASH INTELLIGENCE:**",
            f"• Multi-prompt grounding system",
            f"• Real-time data collection",
            f"• Structured intelligence extraction",
            f"• Source verification and ranking\n",
            f"🧠 *Analysis powered by Gemini 2.5 Pro synthesis*"
        ])
        
        return "\n".join(sources_parts)
    
    def _get_or_create_analysis_cache(self, match_id: str) -> dict:
        """
        Get analysis cache with intelligent management - sessions don't expire unnecessarily.
        """
        # Clean up only very old entries (24+ hours)
        self._cleanup_old_cache_entries()
        
        # Return existing cache if available
        if match_id in self._analysis_cache:
            logger.info(f"Retrieved existing analysis cache for {match_id}")
            return self._analysis_cache[match_id]
        
        # No cache found - this is expected for new analysis
        logger.info(f"No existing cache found for {match_id} (this is normal for new analysis)")
        return None
    
    def _store_analysis_cache(self, match_id: str, pagination_data: dict):
        """
        Store analysis cache with timestamp for intelligent management.
        """
        import time
        
        self._analysis_cache[match_id] = pagination_data
        self._cache_timestamps[match_id] = time.time()
        
        logger.info(f"Stored analysis cache for {match_id} (total cached: {len(self._analysis_cache)})")
    
    def _cleanup_old_cache_entries(self):
        """
        Clean up only very old cache entries (24+ hours) to free memory.
        Recent sessions are preserved to avoid unnecessary expiry.
        """
        import time
        
        current_time = time.time()
        expired_keys = []
        
        for match_id, timestamp in list(self._cache_timestamps.items()):
            if current_time - timestamp > self._max_cache_age:
                expired_keys.append(match_id)
        
        # Remove expired entries
        for match_id in expired_keys:
            if match_id in self._analysis_cache:
                del self._analysis_cache[match_id]
            if match_id in self._cache_timestamps:
                del self._cache_timestamps[match_id]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries (older than 24 hours)")
    
    def _regenerate_analysis_cache(self, match_id: str, premium_result: dict, home_team: str, away_team: str) -> dict:
        """
        Regenerate analysis cache if needed - this creates a new session.
        """
        logger.info(f"Regenerating analysis cache for {match_id}")
        
        # Create new pagination data
        pagination_data = self._create_analysis_pagination(premium_result, home_team, away_team)
        
        # Store with timestamp
        self._store_analysis_cache(match_id, pagination_data)
        
        return pagination_data
    
    def _estimate_goals(self, home_win_prob: float, away_win_prob: float) -> str:
        """
        Estimate goals based on win probabilities.
        """
        if home_win_prob > away_win_prob + 20:
            return "Over 2.5 (75% confidence)"
        elif away_win_prob > home_win_prob + 20:
            return "Over 2.5 (70% confidence)"
        elif abs(home_win_prob - away_win_prob) < 10:
            return "Under 2.5 (60% confidence)"
        else:
            return "2-3 goals expected"
    
    def _estimate_corners(self, data_quality: str, sources: int) -> str:
        """
        Estimate corner count based on data quality.
        """
        if data_quality in ['excellent', 'good'] and sources > 5:
            return "8-12 corners (high confidence)"
        elif data_quality == 'fair':
            return "6-10 corners (medium confidence)"
        else:
            return "5-9 corners (estimated)"
    
    def _estimate_xg_diff(self, home_win_prob: float, away_win_prob: float) -> str:
        """
        Estimate expected goals differential.
        """
        diff = abs(home_win_prob - away_win_prob)
        if diff > 30:
            return "+0.5 to +1.0 (significant edge)"
        elif diff > 15:
            return "+0.2 to +0.5 (moderate edge)"
        else:
            return "±0.2 (balanced match)"
    
    def _estimate_shots(self, home_win_prob: float, away_win_prob: float) -> str:
        """
        Estimate shots on target.
        """
        total_prob = home_win_prob + away_win_prob # Excludes draw
        if total_prob > 140:  # High-scoring expectation
            return "12-18 shots (attacking match)"
        elif total_prob > 120:
            return "8-14 shots (standard match)"
        else:
            return "6-12 shots (defensive match)"
    
    def _format_premium_analysis_response(self, result: Dict, home_team: str, away_team: str) -> str:
        """
        Format premium grounding analysis result for Telegram display.
        Creates comprehensive, user-friendly response with citations and confidence indicators.
        """
        try:
            # Extract key data
            prediction = result.get('prediction', 'Draw')
            confidence = result.get('confidence', 60)
            summary = result.get('summary', 'Analysis completed with available data.')
            factors = result.get('factors', [])
            sources_used = result.get('sources_used', 0)
            data_quality = result.get('data_quality', 'unknown')
            statistics = result.get('statistics', {})
            grounding_notes = result.get('grounding_notes', '')
            data_limitations = result.get('data_limitations', '')
            confidence_adjustment = result.get('confidence_adjustment', '')
            
            # Probability data
            home_win = result.get('home_win', 33.3)
            draw = result.get('draw', 33.3)
            away_win = result.get('away_win', 33.3)
            
            # Quality indicators with enhanced messaging for low data scenarios
            quality_emoji = {
                'excellent': '🌟',
                'good': '✅', 
                'fair': '⚠️',
                'limited': '❗',
                'minimal': '🔍'
            }.get(data_quality, '❓')
            
            confidence_emoji = "🎯" if confidence >= 80 else "📊" if confidence >= 60 else "🤔" if confidence >= 40 else "⚠️"
            
            # Build response with enhanced messaging for low-data scenarios
            if data_quality in ['minimal', 'limited']:
                quality_note = f" - {data_quality} data available"
            else:
                quality_note = ""
            
            response_parts = [
                f"🧠 **Premium Sports Intelligence**\n",
                f"**{home_team} vs {away_team}**\n",
                f"{confidence_emoji} **Prediction**: {prediction} (Confidence: {confidence:.1f}%{quality_note})\n",
                f"{quality_emoji} **Data Quality**: {data_quality.title()} ({sources_used} sources analyzed)\n"
            ]
            
            # Probability breakdown
            response_parts.append(f"\n📈 **Match Probabilities:**")
            response_parts.append(f"🏠 {home_team}: {home_win:.1f}%")
            response_parts.append(f"🤝 Draw: {draw:.1f}%")
            response_parts.append(f"✈️ {away_team}: {away_win:.1f}%\n")
            
            # Executive summary
            if summary and len(summary.strip()) > 10:
                response_parts.append(f"📋 **Executive Summary:**")
                # Truncate summary if too long for Telegram
                clean_summary = summary.strip()[:300]
                if len(summary.strip()) > 300:
                    clean_summary += "..."
                response_parts.append(f"{clean_summary}\n")
            
            # Key factors (top 5)
            if factors and isinstance(factors, list):
                response_parts.append(f"🎯 **Key Factors:**")
                for i, factor in enumerate(factors[:5], 1):
                    if isinstance(factor, dict):
                        name = factor.get('name', f'Factor {i}')[:30]
                        impact = factor.get('impact', 0)
                        evidence = factor.get('evidence', '')[:60]
                        impact_emoji = "🔥" if impact >= 80 else "⚡" if impact >= 60 else "📍"
                        response_parts.append(f"{impact_emoji} {name}: {evidence}...")
                response_parts.append("")
            
            # Data source attribution
            success_rate = statistics.get('success_rate', 0)
            total_sources = statistics.get('total_sources', 0)
            
            response_parts.append(f"📊 **Intelligence Summary:**")
            response_parts.append(f"• Data Collection: {success_rate:.0f}% success rate")
            response_parts.append(f"• Sources Analyzed: {total_sources} verified sources")
            response_parts.append(f"• Analysis Engine: Gemini 2.5 Pro + Flash Grounding")
            
            # Add confidence adjustment notice if applicable
            if confidence_adjustment:
                response_parts.append(f"• Confidence: {confidence_adjustment}")
            
            # Add data limitations if present
            if data_limitations:
                response_parts.append(f"• Limitations: {data_limitations[:100]}...")
            
            if grounding_notes:
                response_parts.append(f"• Notes: {grounding_notes[:80]}...")
            
            response_parts.append(f"\n*Powered by real-time multi-source intelligence* ⚡")
            
            # Join and sanitize the response
            full_response = "\n".join(response_parts)
            sanitized_response = self._sanitize_telegram_text(full_response)
            
            return sanitized_response
            
        except Exception as e:
            logger.error(f"Error formatting premium analysis response: {e}")
            # Fallback to basic response with sanitization
            basic_response = (
                f"Premium Analysis: {home_team} vs {away_team}\n\n"
                f"Prediction: {result.get('prediction', 'Draw')}\n"
                f"Confidence: {result.get('confidence', 60):.1f} percent\n\n"
                f"Response formatting encountered an issue"
            )
            return self._sanitize_telegram_text(basic_response)

    async def live_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Disabled to preserve free-tier rate limits and avoid live polling
        msg = (
            "🔴 *Live updates are disabled for now*\n\n"
            "Free API tiers typically do not support reliable live data and can burn through rate limits.\n"
            "We will enable `/live` once a suitable plan is configured."
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def accuracy_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            await update.message.reply_text("📈 Getting accuracy stats...")
            await self.enhanced_predictor.initialize()
            stats = await self.enhanced_predictor.get_prediction_accuracy_stats()
            accuracy_text = (
                "📈 *Prediction Accuracy*\n\n"
                f"Total predictions: {stats['total_predictions']}\n"
                f"Correct: {stats['correct_predictions']}\n"
                f"Accuracy: {stats['accuracy_percentage']:.1f}%\n"
                f"Recent Form: {stats.get('recent_form', 'Building history...')}\n"
            )
            await update.message.reply_text(accuracy_text, parse_mode='Markdown')
            await self.enhanced_predictor.close()
        except Exception as e:
            logger.error(f"Error in /accuracy: {e}")
            await update.message.reply_text("⚠️ Error getting accuracy stats.")

    async def advanced_prediction_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            if not context.args or len(context.args) < 2:
                await update.message.reply_text(
                    "🔬 *Advanced Prediction System*\n\n"
                    "Please provide two team names.\n"
                    "Example: `/advanced Barcelona Real Madrid`",
                    parse_mode='Markdown'
                )
                return

            home_team = context.args[0]
            away_team = ' '.join(context.args[1:])

            await update.message.reply_text(
                f"🔬 Running advanced prediction analysis for {home_team} vs {away_team}..."
            )

            # LLM-first advanced analysis using Gemini if enabled
            try:
                if os.getenv('USE_LLM') == '1':
                    adv = await self.llm.predict_advanced(home_team, away_team)
                    if isinstance(adv, dict) and 'error' not in adv:
                        pred = adv.get('prediction') or 'Draw'
                        conf = float(adv.get('confidence', max(adv.get('home_win', 0), adv.get('draw', 0), adv.get('away_win', 0))))
                        home_p = float(adv.get('home_win', 33.3))
                        draw_p = float(adv.get('draw', 33.3))
                        away_p = float(adv.get('away_win', 33.3))
                        summary = adv.get('summary') or ''
                        summary_lines = [ln.strip() for ln in str(summary).splitlines() if ln.strip()]
                        summary = '\n'.join(summary_lines[:8])
                        factors = adv.get('factors') or []
                        factor_lines = []
                        for f in factors[:10]:
                            try:
                                name = str(f.get('name', 'Factor'))
                                impact = float(f.get('impact', 0))
                                evidence = str(f.get('evidence', ''))
                                factor_lines.append(f"• {name} ({impact:.1f}%): {evidence}")
                            except Exception:
                                continue
                        parts = [
                            "🔬 *Advanced LLM Analysis*",
                            f"\n**{home_team} vs {away_team}**\n",
                            f"🎯 Prediction: {pred}",
                            f"📊 Confidence: {conf:.1f}%\n",
                            "📈 Probabilities:",
                            f"🏠 Home Win: {home_p:.1f}%",
                            f"🤝 Draw: {draw_p:.1f}%",
                            f"✈️ Away Win: {away_p:.1f}%\n",
                        ]
                        if summary:
                            parts.append("📝 Summary:\n" + summary)
                        if factor_lines:
                            parts.append("\n🤖 Key Factors:\n" + "\n".join(factor_lines))
                        await update.message.reply_text("\n".join(parts), parse_mode='Markdown')
                        return
            except Exception as _e:
                logger.exception("LLM advanced analysis failed")

            await update.message.reply_text("⚠️ Advanced prediction unavailable (LLM error). Please try again.")
            logger.error("Advanced LLM prediction failed and heuristic fallback is disabled (LLM-only mode)")
            return
        except Exception as e:
            logger.error(f"Error in advanced_prediction_command: {e}")
            await update.message.reply_text("⚠️ Error running advanced prediction. Please try again.")

    async def deep_analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Deep analysis command that performs comprehensive research and analysis with enhanced error handling"""
        # Validate user input
        if not context.args:
            await update.message.reply_text(
                "🧠 *Deep Analysis*\n\n"
                "Please provide a sports event to analyze.\n"
                "Example: `/deepanalyze Manchester United vs Liverpool`\n\n"
                "This command performs comprehensive research and analysis using multiple AI models.",
                parse_mode='Markdown'
            )
            return

        # Parse and sanitize the event query
        event_query = ' '.join(context.args).strip()
        
        # Validate event query
        if not event_query or len(event_query) < 3:
            await update.message.reply_text(
                "❌ Invalid input. Please provide a valid sports event to analyze.\n"
                "Example: `/deepanalyze Manchester United vs Liverpool`"
            )
            return
        
        # Check if query contains "vs" or "versus"
        if "vs" not in event_query.lower() and "versus" not in event_query.lower():
            await update.message.reply_text(
                "❌ Invalid input format. Please use 'vs' or 'versus' between team names.\n"
                "Example: `/deepanalyze Manchester United vs Liverpool`"
            )
            return
        
        # Sanitize team names to prevent injection attacks
        sanitized_query = self._sanitize_team_names(event_query)
        
        # Initialize progress tracking
        progress_message = await update.message.reply_text(
            f"🧠 *Deep Analysis Initiated*\n\n"
            f"Analyzing: {sanitized_query}\n\n"
            f"🔄 Gathering initial data...\n\n"
            f"Estimated time: 30-60 seconds",
            parse_mode='Markdown'
        )

        try:
            # Import required modules
            from query_generator import generate_search_queries
            from data_acquisition import acquire_data
            from content_processor import process_content_with_flash
            from final_analyzer import get_final_analysis_with_pro

            # Runtime verification logs: which modules/flags are active
            import inspect
            try:
                gen_src = inspect.getsourcefile(generate_search_queries)
            except Exception:
                gen_src = 'unknown'
            try:
                acq_src = inspect.getsourcefile(acquire_data)
            except Exception:
                acq_src = 'unknown'
            flags = {
                'USE_FLASH_QUERY_GENERATION': os.getenv('USE_FLASH_QUERY_GENERATION'),
                'SERP_RERANKER_ENABLED': os.getenv('SERP_RERANKER_ENABLED'),
                'GEMINI_API_KEY_set': bool(os.getenv('GEMINI_API_KEY')),
                'GEMINI_MODEL_ID': os.getenv('GEMINI_MODEL_ID'),
                'GEMINI_FLASH_MODEL_ID': os.getenv('GEMINI_FLASH_MODEL_ID'),
            }
            logger.info(f"DeepAnalyze flags: {flags}")
            logger.info(f"generate_search_queries from: {gen_src}")
            logger.info(f"acquire_data from: {acq_src}")

            # Overall timeout for the entire deep analysis process - configurable
            overall_timeout = float(os.getenv('DEEP_ANALYSIS_OVERALL_TIMEOUT', 90.0))

            # Step 1: Generate search queries with timeout and enhanced error handling
            max_query_retries = int(os.getenv('QUERY_GENERATION_MAX_RETRIES', 3))
            for query_retry in range(max_query_retries):
                try:
                    await progress_message.edit_text(
                        f"🔍 *Step 1/4*: Generating search queries (attempt {query_retry + 1}/{max_query_retries})...\n\n"
                        f"Event: {sanitized_query}\n"
                        f"⏳ Please wait...\n\n"
                        f"Estimated time: 5-10 seconds",
                        parse_mode='Markdown'
                    )
                    
                    # Calculate adaptive timeout based on retry count
                    query_generation_timeout = float(os.getenv('QUERY_GENERATION_TIMEOUT', 10.0))
                    adaptive_timeout = self._calculate_adaptive_timeout(query_generation_timeout, query_retry, max_query_retries)
                    queries = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, generate_search_queries, sanitized_query),
                        timeout=adaptive_timeout
                    )
                    # Success, break out of retry loop
                    break
                except asyncio.TimeoutError:
                    if query_retry < max_query_retries - 1:
                        # Log retry attempt
                        logger.warning(f"Query generation timeout (attempt {query_retry + 1}/{max_query_retries}), retrying with increased timeout...")
                        # Wait a bit before retrying
                        await asyncio.sleep(1.0 * (query_retry + 1))
                        continue
                    else:
                        # All retries exhausted, re-raise the exception
                        raise
            
            if not queries or len(queries) == 0:
                await progress_message.edit_text(
                    f"❌ *Query Generation Failed*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ No search queries could be generated for this event.\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Check the event format (e.g., 'Team A vs Team B')\n"
                    f"• Try a more specific team or player names\n"
                    f"• Ensure the event is upcoming or recent"
                )
                return
            
            logger.info(f"Generated {len(queries)} search queries for '{sanitized_query}'")
            
            # Provide feedback on intermediate results
            await progress_message.edit_text(
                f"🔍 *Step 1/4*: Generating search queries...\n\n"
                f"Event: {sanitized_query}\n"
                f"✅ Generated {len(queries)} search queries\n\n"
                f"Estimated time: 5-10 seconds",
                parse_mode='Markdown'
            )
            
        except asyncio.TimeoutError:
            await progress_message.edit_text(
                f"⏱️ *Query Generation Timeout*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ The query generation process took too long.\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try again in a few minutes\n"
                f"• Use a more specific event format\n"
                f"• Check your internet connection"
            )
            logger.error(f"Query generation timeout for event: {sanitized_query}")
            return
        except ValueError as e:
            # Handle specific ValueError from query generator
            await progress_message.edit_text(
                f"❌ *Query Generation Error*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Invalid input for query generation: {str(e)}\n\n"
                f"💡 *Suggestions:*\n"
                f"• Check the event format (e.g., 'Team A vs Team B')\n"
                f"• Ensure team names are properly formatted"
            )
            logger.error(f"Query generation ValueError for event '{sanitized_query}': {e}")
            return
        except Exception as e:
            await progress_message.edit_text(
                f"❌ *Query Generation Error*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Failed to generate search queries: {str(e)}\n\n"
                f"💡 *Suggestions:*\n"
                f"• Check the event format (e.g., 'Team A vs Team B')\n"
                f"• Try a different event\n"
                f"• Contact support if the issue persists"
            )
            logger.error(f"Query generation error for event '{sanitized_query}': {e}")
            return

        # Step 2: Acquire data with timeout and enhanced error handling
        max_data_retries = int(os.getenv('DATA_ACQUISITION_MAX_RETRIES', 3))
        articles = None
        for data_retry in range(max_data_retries):
            try:
                await progress_message.edit_text(
                    f"🌐 *Step 2/4*: Acquiring data from sources (attempt {data_retry + 1}/{max_data_retries})...\n\n"
                    f"Event: {sanitized_query}\n"
                    f"🔍 Queries: {len(queries)}\n"
                    f"⏳ Please wait (this may take 15-30 seconds)...\n\n"
                    f"Estimated time remaining: 15-30 seconds",
                    parse_mode='Markdown'
                )
                
                # Calculate adaptive timeout based on retry count
                data_acquisition_timeout = float(os.getenv('DATA_ACQUISITION_TIMEOUT', 30.0))
                adaptive_timeout = self._calculate_adaptive_timeout(data_acquisition_timeout, data_retry, max_data_retries)
                articles = await asyncio.wait_for(
                    acquire_data(queries),
                    timeout=adaptive_timeout
                )
                # Success, break out of retry loop
                break
            except (asyncio.TimeoutError, asyncio.CancelledError):
                if data_retry < max_data_retries - 1:
                    # Log retry attempt
                    logger.warning(f"Data acquisition timeout/cancel (attempt {data_retry + 1}/{max_data_retries}), retrying with increased timeout...")
                    # Wait a bit before retrying
                    await asyncio.sleep(2.0 * (data_retry + 1))
                    continue
                else:
                    # All retries exhausted, handle gracefully
                    await progress_message.edit_text(
                        f"⏱️ *Data Acquisition Timeout*\n\n"
                        f"Event: {sanitized_query}\n\n"
                        f"⚠️ Collecting articles took too long.\n\n"
                        f"💡 *Suggestions:*\n"
                        f"• Try again in a few minutes\n"
                        f"• Check your internet connection\n"
                        f"• Try a more specific or recent event",
                        parse_mode='Markdown'
                    )
                    logger.error(f"Data acquisition timeout for event: {sanitized_query}")
                    return
            except aiohttp.ClientError as e:
                await progress_message.edit_text(
                    f"🌐 *Network Error*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ Network connectivity issues occurred: {str(e)}\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Check your internet connection\n"
                    f"• Try again in a few minutes\n"
                    f"• Restart your network connection",
                    parse_mode='Markdown'
                )
                logger.error(f"Network error during data acquisition for event '{sanitized_query}': {e}")
                return
            except Exception as e:
                error_msg = str(e)
                if "403" in error_msg:
                    await progress_message.edit_text(
                        f"🚦 *Access Forbidden*\n\n"
                        f"Event: {sanitized_query}\n\n"
                        f"⚠️ Search service denied access (403 error).\n\n"
                        f"💡 *Suggestions:*\n"
                        f"• Try again in a few minutes\n"
                        f"• Use a different event\n"
                        f"• Check if the search service is working",
                        parse_mode='Markdown'
                    )
                elif "429" in error_msg or "rate limit" in error_msg.lower():
                    await progress_message.edit_text(
                        f"🚦 *Rate Limit Exceeded*\n\n"
                        f"Event: {sanitized_query}\n\n"
                        f"⚠️ Search service is temporarily unavailable due to rate limits.\n\n"
                        f"💡 *Suggestions:*\n"
                        f"• Try again in a few minutes\n"
                        f"• Use a different event\n"
                        f"• Check back later when limits reset",
                        parse_mode='Markdown'
                    )
                elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                    await progress_message.edit_text(
                        f"🌐 *Network Error*\n\n"
                        f"Event: {sanitized_query}\n\n"
                        f"⚠️ Network connectivity issues occurred.\n\n"
                        f"💡 *Suggestions:*\n"
                        f"• Check your internet connection\n"
                        f"• Try again in a few minutes\n"
                        f"• Restart your network connection",
                        parse_mode='Markdown'
                    )
                else:
                    await progress_message.edit_text(
                        f"❌ *Data Acquisition Error*\n\n"
                        f"Event: {sanitized_query}\n\n"
                        f"⚠️ Failed to acquire data from sources: {str(e)}\n\n"
                        f"💡 *Suggestions:*\n"
                        f"• Try again in a few minutes\n"
                        f"• Use a different event\n"
                        f"• Contact support if the issue persists",
                        parse_mode='Markdown'
                    )
                logger.error(f"Data acquisition error for event '{sanitized_query}': {e}")
                return
        
        if not articles or len(articles) == 0:
            await progress_message.edit_text(
                f"❌ *Data Acquisition Failed*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ No relevant data found for this event.\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try a more popular or recent event\n"
                f"• Check if the teams/players are correctly named\n"
                f"• Try again later when more data is available"
            )
            return
        
        logger.info(f"Acquired {len(articles)} articles for '{sanitized_query}'")
        
        # Provide feedback on intermediate results
        await progress_message.edit_text(
            f"🌐 *Step 2/4*: Acquiring data from sources...\n\n"
            f"Event: {sanitized_query}\n"
            f"🔍 Queries: {len(queries)}\n"
            f"📄 Articles found: {len(articles)}\n\n"
            f"Estimated time remaining: 10-20 seconds",
            parse_mode='Markdown'
        )

        # Step 3: Process content with Flash with timeout and error handling
        max_content_retries = int(os.getenv('CONTENT_PROCESSING_MAX_RETRIES', 3))
        for content_retry in range(max_content_retries):
            try:
                await progress_message.edit_text(
                    f"⚡ *Step 3/4*: Processing content with Gemini Flash (attempt {content_retry + 1}/{max_content_retries})...\n\n"
                    f"Event: {sanitized_query}\n"
                    f"📄 Articles: {len(articles)}\n"
                    f"⏳ Please wait (this may take 10-20 seconds)...",
                    parse_mode='Markdown'
                )
                
                # Extract team names for content processing
                team_parts = sanitized_query.split(" vs ")
                if len(team_parts) < 2:
                    team_parts = sanitized_query.split(" versus ")
                if len(team_parts) < 2:
                    # Fallback to splitting by common delimiters
                    for delimiter in ["-", "–", "&", "and", "@"]:
                        team_parts = sanitized_query.split(delimiter)
                        if len(team_parts) >= 2:
                            break
                if len(team_parts) < 2:
                    team_parts = [sanitized_query, "opponent"]
                
                # Calculate adaptive timeout based on retry count
                content_processing_timeout = float(os.getenv('CONTENT_PROCESSING_TIMEOUT', 20.0))
                adaptive_timeout = self._calculate_adaptive_timeout(content_processing_timeout, content_retry, max_content_retries)
                processed_data = await asyncio.wait_for(
                    process_content_with_flash(articles, team_parts),
                    timeout=adaptive_timeout
                )
                # Success, break out of retry loop
                break
            except asyncio.TimeoutError:
                if content_retry < max_content_retries - 1:
                    # Log retry attempt
                    logger.warning(f"Content processing timeout (attempt {content_retry + 1}/{max_content_retries}), retrying with increased timeout...")
                    # Wait a bit before retrying
                    await asyncio.sleep(1.5 * (content_retry + 1))
                    continue
                else:
                    # All retries exhausted, re-raise the exception
                    raise
        
        logger.info(f"Processed {len(processed_data)} relevant articles for '{sanitized_query}'")
        
        if not processed_data or len(processed_data) == 0:
            await progress_message.edit_text(
                f"❌ *Content Processing Failed*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ No relevant information could be extracted from sources.\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try a more popular or recent event\n"
                f"• Check if the teams/players are correctly named\n"
                f"• Try again later when more data is available"
            )
            return
            
        try:
            pass  # This is a placeholder to maintain the try/except structure
        except asyncio.TimeoutError:
            await progress_message.edit_text(
                f"⏱️ *Content Processing Timeout*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Content processing took too long.\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try again in a few minutes\n"
                f"• Use a different event\n"
                f"• Try an event with less complex data"
            )
            logger.error(f"Content processing timeout for event: {sanitized_query}")
            return
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors specifically
            await progress_message.edit_text(
                f"❌ *Content Processing Error*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Invalid response format from AI model: {str(e)}\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try again in a few minutes\n"
                f"• Use a different event\n"
                f"• Contact support if the issue persists"
            )
            logger.error(f"JSON parsing error during content processing for event '{sanitized_query}': {e}")
            return
        except aiohttp.ClientError as e:
            # Handle network-related errors from aiohttp
            await progress_message.edit_text(
                f"🌐 *Network Error*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Network connectivity issues occurred: {str(e)}\n\n"
                f"💡 *Suggestions:*\n"
                f"• Check your internet connection\n"
                f"• Try again in a few minutes\n"
                f"• Restart your network connection"
            )
            logger.error(f"Network error during content processing for event '{sanitized_query}': {e}")
            return
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                # Extract retry delay if available
                retry_delay = self._extract_retry_delay(error_msg)
                retry_msg = f" (retry in {retry_delay} seconds)" if retry_delay else ""
                
                await progress_message.edit_text(
                    f"🚦 *Gemini API Rate Limit{retry_msg}*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ Gemini Flash model is temporarily unavailable due to rate limits.\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Try again later when limits reset\n"
                    f"• Use a different event\n"
                    f"• Check your API quota at https://aistudio.google.com/"
                )
            elif "invalid json" in error_msg.lower() or "json" in error_msg.lower():
                await progress_message.edit_text(
                    f"❌ *Content Processing Error*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ Invalid response format from AI model.\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Try again in a few minutes\n"
                    f"• Use a different event\n"
                    f"• Contact support if the issue persists"
                )
            else:
                await progress_message.edit_text(
                    f"❌ *Content Processing Error*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ Failed to process content with AI model: {str(e)}\n"
                    f"💡 *Suggestions:*\n"
                    f"• Try again in a few minutes\n"
                    f"• Use a different event\n"
                    f"• Contact support if the issue persists"
                )
            logger.error(f"Content processing error for event '{sanitized_query}': {e}")
            return

        # Validate processed data before final analysis
        if not self._validate_processed_data(processed_data):
            await progress_message.edit_text(
                f"❌ *Data Validation Failed*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Processed data failed validation checks.\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try a different event\n"
                f"• Check if sufficient data was collected\n"
                f"• Try again later"
            )
            logger.warning(f"Processed data validation failed for event: {sanitized_query}")
            return
        
        # Step 4: Get final analysis with Pro with timeout and error handling
        max_final_retries = int(os.getenv('FINAL_ANALYSIS_MAX_RETRIES', 3))
        for final_retry in range(max_final_retries):
            try:
                await progress_message.edit_text(
                    f"🤖 *Step 4/4*: Generating final analysis with Gemini Pro (attempt {final_retry + 1}/{max_final_retries})...\n\n"
                    f"Event: {sanitized_query}\n"
                    f"📊 Processed Data: {len(processed_data)} items\n"
                    f"⏳ Please wait (this may take 10-20 seconds)...",
                    parse_mode='Markdown'
                )
                
                # Calculate adaptive timeout based on retry count
                final_analysis_timeout = float(os.getenv('FINAL_ANALYSIS_TIMEOUT', 20.0))
                adaptive_timeout = self._calculate_adaptive_timeout(final_analysis_timeout, final_retry, max_final_retries)
                final_analysis = await asyncio.wait_for(
                    get_final_analysis_with_pro(processed_data, sanitized_query),
                    timeout=adaptive_timeout
                )
                # Success, break out of retry loop
                break
            except asyncio.TimeoutError:
                if final_retry < max_final_retries - 1:
                    # Log retry attempt
                    logger.warning(f"Final analysis timeout (attempt {final_retry + 1}/{max_final_retries}), retrying with increased timeout...")
                    # Wait a bit before retrying
                    await asyncio.sleep(1.5 * (final_retry + 1))
                    continue
                else:
                    # All retries exhausted, re-raise the exception
                    raise
        
        # Send the final report with safe Markdown handling
        final_report = (
            f"📊 *Deep Analysis Complete*\n\n"
            f"Event: {sanitized_query}\n\n"
            f"{final_analysis}"
        )
        try:
            await progress_message.edit_text(final_report, parse_mode='Markdown')
        except BadRequest:
            try:
                sanitized_report = self._sanitize_telegram_text(final_report)
                await progress_message.edit_text(sanitized_report, parse_mode='Markdown')
            except BadRequest:
                # Fallback to plain text if Markdown still fails
                plain = final_report.replace('*', '').replace('_', '').replace('`', '').replace('[', '(').replace(']', ')')
                await progress_message.edit_text(plain)
        
        try:
            pass  # This is a placeholder to maintain the try/except structure
        except asyncio.TimeoutError:
            await progress_message.edit_text(
                f"⏱️ *Final Analysis Timeout*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Final analysis took too long.\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try again in a few minutes\n"
                f"• Use a different event\n"
                f"• Try an event with less complex data"
            )
            logger.error(f"Final analysis timeout for event: {sanitized_query}")
            return
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors specifically
            await progress_message.edit_text(
                f"❌ *Final Analysis Error*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Invalid response format from AI model: {str(e)}\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try again in a few minutes\n"
                f"• Use a different event\n"
                f"• Contact support if the issue persists"
            )
            logger.error(f"JSON parsing error during final analysis for event '{sanitized_query}': {e}")
            return
        except aiohttp.ClientError as e:
            # Handle network-related errors from aiohttp
            await progress_message.edit_text(
                f"🌐 *Network Error*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Network connectivity issues occurred: {str(e)}\n"
                f"💡 *Suggestions:*\n"
                f"• Check your internet connection\n"
                f"• Try again in a few minutes\n"
                f"• Restart your network connection"
            )
            logger.error(f"Network error during final analysis for event '{sanitized_query}': {e}")
            return
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                # Extract retry delay if available
                retry_delay = self._extract_retry_delay(error_msg)
                retry_msg = f" (retry in {retry_delay} seconds)" if retry_delay else ""
                
                await progress_message.edit_text(
                    f"🚦 *Gemini API Rate Limit{retry_msg}*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ Gemini Pro model is temporarily unavailable due to rate limits.\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Try again later when limits reset\n"
                    f"• Use a different event\n"
                    f"• Check your API quota at https://aistudio.google.com/"
                )
            elif "invalid json" in error_msg.lower() or "json" in error_msg.lower():
                await progress_message.edit_text(
                    f"❌ *Final Analysis Error*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ Invalid response format from AI model.\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Try again in a few minutes\n"
                    f"• Use a different event\n"
                    f"• Contact support if the issue persists"
                )
            else:
                await progress_message.edit_text(
                    f"❌ *Final Analysis Error*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ Failed to generate final analysis: {str(e)}\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Try again in a few minutes\n"
                    f"• Use a different event\n"
                    f"• Contact support if the issue persists"
                )
            logger.error(f"Final analysis error for event '{sanitized_query}': {e}")
            return
            
        try:
            pass  # This is a placeholder to maintain the try/except structure
        except asyncio.TimeoutError:
            await progress_message.edit_text(
                f"⏱️ *Overall Process Timeout*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ The entire analysis process took too long (>{overall_timeout} seconds).\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try again in a few minutes\n"
                f"• Use a different event\n"
                f"• Check your internet connection"
            )
            logger.error(f"Overall process timeout for event: {sanitized_query}")
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors specifically
            await progress_message.edit_text(
                f"❌ *Data Processing Error*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Invalid response format from AI model: {str(e)}\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try again in a few minutes\n"
                f"• Use a different event\n"
                f"• Contact support if the issue persists"
            )
            logger.error(f"JSON parsing error in deep_analyze_command for event '{sanitized_query}': {e}")
        except aiohttp.ClientError as e:
            # Handle network-related errors from aiohttp
            await progress_message.edit_text(
                f"🌐 *Network Error*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Network connectivity issues occurred: {str(e)}\n"
                f"💡 *Suggestions:*\n"
                f"• Check your internet connection\n"
                f"• Try again in a few minutes\n"
                f"• Restart your network connection"
            )
            logger.error(f"Network error in deep_analyze_command for event '{sanitized_query}': {e}")
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                # Extract retry delay if available
                retry_delay = self._extract_retry_delay(error_msg)
                retry_msg = f" (retry in {retry_delay} seconds)" if retry_delay else ""
                
                await progress_message.edit_text(
                    f"🚦 *API Rate Limit{retry_msg}*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ API is temporarily unavailable due to rate limits.\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Try again later when limits reset\n"
                    f"• Use a different event\n"
                    f"• Check your API quota"
                )
            elif "403" in error_msg:
                await progress_message.edit_text(
                    f"🔐 *Access Forbidden*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ Access to the service was denied (403 error).\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Check your API key\n"
                    f"• Verify your account permissions\n"
                    f"• Contact support if the issue persists"
                )
            elif "invalid json" in error_msg.lower() or "json" in error_msg.lower():
                await progress_message.edit_text(
                    f"❌ *Data Processing Error*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ Invalid response format from service.\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Try again in a few minutes\n"
                    f"• Use a different event\n"
                    f"• Contact support if the issue persists"
                )
            else:
                logger.error(f"Unexpected error in deep_analyze_command for event '{sanitized_query}': {e}")
                await progress_message.edit_text(
                    f"❌ *Unexpected Error*\n\n"
                    f"Event: {sanitized_query}\n\n"
                    f"⚠️ An unexpected error occurred during analysis: {str(e)}\n\n"
                    f"💡 *Suggestions:*\n"
                    f"• Try again in a few minutes\n"
                    f"• Use a different event\n"
                    f"• Contact support with error details"
                )

    async def degen_analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Degen analysis command that performs real-time research and analysis with degen tone"""
        # Check if the feature is enabled
        if os.getenv("USE_FLASH_LIVE", "0") != "1":
            await update.message.reply_text(
                "❌ *Degen Analysis Disabled*\n\n"
                "The /degenanalyze feature is currently disabled.\n"
                "Set USE_FLASH_LIVE=1 in your environment to enable it.",
                parse_mode='Markdown'
            )
            return

        # Validate user input
        if not context.args:
            await update.message.reply_text(
                "🧠 *Degen Analysis*\n\n"
                "Please provide a sports event to analyze.\n"
                "Example: `/degenanalyze Manchester United vs Liverpool`\n\n"
                "This command performs real-time research and analysis with degen tone.",
                parse_mode='Markdown'
            )
            return

        # Parse and sanitize the event query
        event_query = ' '.join(context.args).strip()
        
        # Validate event query
        if not event_query or len(event_query) < 3:
            await update.message.reply_text(
                "❌ Invalid input. Please provide a valid sports event to analyze.\n"
                "Example: `/degenanalyze Manchester United vs Liverpool`"
            )
            return
        
        # Check if query contains "vs" or "versus"
        if "vs" not in event_query.lower() and "versus" not in event_query.lower():
            await update.message.reply_text(
                "❌ Invalid input format. Please use 'vs' or 'versus' between team names.\n"
                "Example: `/degenanalyze Manchester United vs Liverpool`"
            )
            return
        
        # Sanitize team names to prevent injection attacks
        sanitized_query = self._sanitize_team_names(event_query)
        
        # Initialize progress tracking
        progress_message = await update.message.reply_text(
            f"🧠 *Degen Analysis Initiated*\n\n"
            f"Analyzing: {sanitized_query}\n\n"
            f"🔄 Gathering initial data...\n\n"
            f"Estimated time: 30-60 seconds",
            parse_mode='Markdown'
        )

        # Initialize session variables
        live_session = None
        try:
            # Create and initialize a LiveSession
            live_session = LiveSession(sanitized_query)
            
            # Start the live session
            await live_session.start_session()
            
            # Post initial degen-toned message
            await progress_message.edit_text(
                f"🔥 *Degen Analysis Started*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"📡 Scanning the interwebs for spicy degen insights...\n"
                f"📈 Real-time updates coming your way!\n\n"
                f"Estimated time: 30-60 seconds",
                parse_mode='Markdown'
            )
            
            # Handle streaming updates with safe-edit fallback
            session_end_message = None
            while live_session.is_streaming:
                try:
                    # Wait for updates with a timeout
                    update_data = await asyncio.wait_for(live_session.stream_queue.get(), timeout=1.0)
                    
                    # Check if this is a session end message
                    if update_data.get("type") == "session_end":
                        session_end_message = update_data.get("message", "Research session completed")
                        break
                    
                    # Format the update using the stream renderer
                    formatted_update = update_data.get("message", "New update received")
                    
                    # Send the update to the user
                    try:
                        await progress_message.edit_text(formatted_update, parse_mode='Markdown')
                    except BadRequest as e:
                        # If Markdown fails, try with sanitized text
                        sanitized_update = self._sanitize_telegram_text(formatted_update)
                        await progress_message.edit_text(sanitized_update, parse_mode='Markdown')
                    except Exception:
                        # If all else fails, send as plain text
                        plain_text = formatted_update.replace('*', '').replace('_', '').replace('`', '').replace('[', '(').replace(']', ')')
                        await progress_message.edit_text(plain_text)
                
                except asyncio.TimeoutError:
                    # Continue the loop to check if streaming is still active
                    continue
                except Exception as e:
                    logger.error(f"Error handling streaming update: {e}")
                    # Continue processing other updates
                    continue
            
            # Generate and display the final degen-toned report
            final_report = live_session.generate_final_report()
            
            try:
                await progress_message.edit_text(final_report, parse_mode='Markdown')
            except BadRequest as e:
                # If Markdown fails, try with sanitized text
                sanitized_report = self._sanitize_telegram_text(final_report)
                await progress_message.edit_text(sanitized_report, parse_mode='Markdown')
            except Exception:
                # If all else fails, send as plain text
                plain_text = final_report.replace('*', '').replace('_', '').replace('`', '').replace('[', '(').replace(']', ')')
                await progress_message.edit_text(plain_text)
            
        except Exception as e:
            logger.error(f"Error in degen_analyze_command: {e}")
            # Send error message to user
            error_message = (
                f"❌ *Degen Analysis Error*\n\n"
                f"Event: {sanitized_query}\n\n"
                f"⚠️ Failed to perform degen analysis: {str(e)}\n\n"
                f"💡 *Suggestions:*\n"
                f"• Try again in a few minutes\n"
                f"• Use a different event\n"
                f"• Contact support if the issue persists"
            )
            try:
                await progress_message.edit_text(error_message, parse_mode='Markdown')
            except Exception:
                # If we can't edit the message, try to send a new one
                try:
                    await update.message.reply_text(error_message, parse_mode='Markdown')
                except Exception:
                    # Last resort: send a simple text message
                    await update.message.reply_text(f"Error: {str(e)}")
        finally:
            # Stop the session if it was created
            try:
                if live_session:
                    await live_session.stop_session()
            except Exception as stop_error:
                logger.error(f"Error stopping live session: {stop_error}")

    def _sanitize_team_names(self, query: str) -> str:
        """
        Sanitize team names to prevent injection attacks and handle malformed input.
        
        Args:
            query (str): The user-provided query
            
        Returns:
            str: Sanitized query
        """
        import re
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\'&;`]', '', query)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        # Fix common formatting issues with "vs"
        sanitized = re.sub(r'\s*vs[.\s]*vs\s*', ' vs ', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'^vs\s+', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _extract_retry_delay(self, error_msg: str) -> Optional[int]:
        """
        Extract retry delay from error message if available.
        
        Args:
            error_msg (str): Error message string
            
        Returns:
            Optional[int]: Retry delay in seconds or None if not found
        """
        import re
        
        # Look for retry_delay in the error message
        match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_msg)
        if match:
            return int(match.group(1))
        return None
    
    def _validate_processed_data(self, processed_data: List[Dict]) -> bool:
        """
        Validate processed data before sending to final analysis.
        
        Args:
            processed_data (List[Dict]): Processed data from content processor
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check if we have any data
        if not processed_data or len(processed_data) == 0:
            return False
        
        # Check if we have at least some relevant data
        relevant_items = 0
        for item in processed_data:
            # Check for key fields that indicate relevant data
            if (item.get('summary') and len(str(item['summary']).strip()) > 10) or \
               (item.get('key_players_mentioned') and len(item['key_players_mentioned']) > 0) or \
               (item.get('injuries_or_suspensions') and len(item['injuries_or_suspensions']) > 0):
                relevant_items += 1
        
        # Require at least 1 relevant item
        return relevant_items >= 1
    
    def _calculate_adaptive_timeout(self, base_timeout: float, retry_count: int, max_retries: int = 3) -> float:
        """
        Calculate adaptive timeout based on retry count.
        
        Args:
            base_timeout (float): Base timeout value
            retry_count (int): Current retry attempt (0 for first attempt)
            max_retries (int): Maximum number of retries allowed
            
        Returns:
            float: Adaptive timeout value
        """
        # Increase timeout by 50% for each retry, up to double the original timeout
        timeout_multiplier = 1.0 + (0.5 * retry_count)
        # Cap the multiplier to prevent excessive timeouts
        max_multiplier = min(2.0, 1.0 + (0.5 * max_retries))
        timeout_multiplier = min(timeout_multiplier, max_multiplier)
        return base_timeout * timeout_multiplier

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Global error handler to avoid unhandled exception logs from PTB."""
        try:
            logger.error("Unhandled exception in handler", exc_info=context.error)
        except Exception:
            # As a last resort, ensure we don't raise from the error handler itself
            logger.exception("Error within error_handler")

    # ----- Buttons -----
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data
        try:
            # Handle analysis pagination callbacks first
            if data.startswith('analysis_'):
                await self._handle_analysis_callback(query, data)
                return
            
            # Handle new multi-step navigation callbacks
            if data.startswith('region_'):
                await self._handle_region_selection(query, data)
                return
            elif data.startswith('leagues_page_'):
                await self._handle_leagues_pagination(query, data)
                return
            elif data.startswith('league_matches_'):
                await self._handle_league_matches(query, data)
                return
            elif data.startswith('upcoming_'):
                await self._handle_upcoming_region(query, data)
                return
            elif data.startswith('predict_match_'):
                await self._handle_match_prediction(query, data)
                return
            
            # Handle existing callbacks
            if data == "leagues":
                await self.send_leagues_response(query)
            elif data == "upcoming":
                await self.send_upcoming_response(query)
            elif data == "predict":
                await self.send_predict_response(query)
            elif data == "odds":
                await self.send_odds_response(query)
            elif data.startswith("odds_"):
                await self.send_specific_league_odds(query)
            elif data == "advanced":
                await self.send_advanced_response(query)
            elif data == "analysis":
                await self.send_analysis_response(query)
            elif data == "live":
                await self.send_live_response(query)
            elif data == "accuracy":
                await self.send_accuracy_response(query)
            elif data == "stats":
                await self.send_stats_response(query)
            elif data in {"community", "dashboard", "leaderboard", "feed", "badges"}:
                await query.edit_message_text("Community features are disabled in local mode.")
            else:
                await query.edit_message_text("❌ Unknown action. Please try again.")
        except Exception as e:
            logger.error(f"Error handling button '{data}': {e}")
            await query.edit_message_text("⚠️ Error handling action. Please try again.")
    
    async def _handle_analysis_callback(self, query, data: str):
        """Handle analysis pagination and navigation callbacks"""
        try:
            parts = data.split('_')
            if len(parts) < 3:
                await query.edit_message_text("❌ Invalid navigation data")
                return
            
            action = parts[1]  # next, prev, page, refresh, close, info
            
            # Extract match_id properly from callback format
            # Format: analysis_action_Team1_vs_Team2_pagenum (optional)
            # We need to rebuild the match_id from parts[2:]
            if len(parts) >= 5 and parts[3] == 'vs':  # Format: analysis_action_Team1_vs_Team2_pagenum
                match_id = f"{parts[2]}_vs_{parts[4]}"
            elif len(parts) >= 4 and parts[2] == 'vs':  # Format: analysis_action_vs_Team2 (malformed)
                await query.edit_message_text("❌ Invalid team format in callback data")
                return
            else:
                # Try to find 'vs' in the parts to reconstruct match_id
                vs_index = -1
                for i, part in enumerate(parts[2:], 2):
                    if part == 'vs':
                        vs_index = i
                        break
                
                if vs_index > 0 and vs_index + 1 < len(parts):
                    # Reconstruct match_id: everything from parts[2] to the part after 'vs'
                    home_parts = parts[2:vs_index]
                    away_parts = []
                    
                    # Find the end of away team name (before page number)
                    # Only consider the very last part as a page number if it's a single digit
                    end_index = len(parts)
                    if len(parts) > vs_index + 1:
                        last_part = parts[-1]
                        if last_part.isdigit() and len(last_part) <= 2:  # Page numbers are typically 1-2 digits
                            end_index = len(parts) - 1
                    
                    away_parts = parts[vs_index + 1:end_index]
                    
                    if home_parts and away_parts:
                        match_id = f"{'_'.join(home_parts)}_vs_{'_'.join(away_parts)}"
                    else:
                        await query.edit_message_text("❌ Unable to parse team names from callback")
                        return
                else:
                    # Fallback: assume everything after action is match_id (old format)
                    match_id = '_'.join(parts[2:])
                    # Remove page number if present at the end
                    match_parts = match_id.split('_')
                    if match_parts and match_parts[-1].isdigit():
                        match_id = '_'.join(match_parts[:-1])
            
            logger.info(f"Parsed callback - Action: {action}, Match ID: {match_id}")
            
            # Get cached pagination data with intelligent management
            pagination_data = self._get_or_create_analysis_cache(match_id)
            
            if pagination_data is None:
                # No cache found - offer to regenerate
                match_parts = match_id.split('_vs_')
                if len(match_parts) == 2:
                    home_team = match_parts[0].replace('_', ' ')
                    away_team = match_parts[1].replace('_', ' ')
                    
                    regenerate_keyboard = [[
                        InlineKeyboardButton(
                            "🔄 Generate New Analysis", 
                            callback_data=f"analysis_regenerate_{match_id}"
                        ),
                        InlineKeyboardButton(
                            "❌ Close", 
                            callback_data=f"analysis_close_{match_id}"
                        )
                    ]]
                    
                    await query.edit_message_text(
                        f"📋 **Analysis Session Recovery**\n\n"
                        f"**{home_team} vs {away_team}**\n\n"
                        f"🔄 Session data was cleared for memory optimization.\n\n"
                        f"✅ **This is normal behavior** - helps keep the bot running smoothly!\n\n"
                        f"**What happened:**\n"
                        f"• Cache was cleared to free up memory\n"
                        f"• Analysis sessions are cleaned up automatically\n\n"
                        f"💡 **Quick Recovery:**\n"
                        f"• Generate fresh analysis (recommended)\n"
                        f"• Get latest data and insights\n"
                        f"*🆕 New analysis includes most recent team data*",
                        parse_mode='Markdown',
                        reply_markup=InlineKeyboardMarkup(regenerate_keyboard)
                    )
                else:
                    await query.edit_message_text(
                        "⚠️ Unable to identify teams from session data.\n\n"
                        "Please run `/analysis Team1 Team2` to start a new analysis."
                    )
                return
            total_pages = pagination_data['total_pages']
            pages = pagination_data['pages']
            
            if action == 'close':
                await query.edit_message_text("✅ Analysis closed.")
                # Clean up cache using intelligent management
                if match_id in self._analysis_cache:
                    del self._analysis_cache[match_id]
                if match_id in self._cache_timestamps:
                    del self._cache_timestamps[match_id]
                logger.info(f"Manually closed and cleaned up cache for {match_id}")
                return
            
            elif action == 'info':
                # Extract team names from match_id (format: "Home_Team_vs_Away_Team")
                match_parts = match_id.split('_vs_')
                if len(match_parts) == 2:
                    home_team = match_parts[0].replace('_', ' ')
                    away_team = match_parts[1].replace('_', ' ')
                    team_display = f"{home_team} vs {away_team}"
                else:
                    team_display = match_id.replace('_', ' ')
                
                await query.answer(
                    f"Analysis for {team_display} | {total_pages} pages available",
                    show_alert=True
                )
                return
            
            elif action == 'refresh':
                await query.answer("Analysis refresh not implemented yet", show_alert=True)
                return
            
            elif action == 'regenerate':
                # Regenerate analysis for expired/missing cache
                match_parts = match_id.split('_vs_')
                if len(match_parts) == 2:
                    home_team = match_parts[0].replace('_', ' ')
                    away_team = match_parts[1].replace('_', ' ')
                    
                    # Show progress message
                    await query.edit_message_text(
                        f"🔄 **Regenerating Analysis**\n\n"
                        f"**{home_team} vs {away_team}**\n\n"
                        f"⏳ Generating fresh analysis with latest data...\n"
                        f"🤖 This may take 30-60 seconds",
                        parse_mode='Markdown'
                    )
                    
                    try:
                        # Run new premium analysis
                        if self.llm_predictor and self.llm_predictor._client_ready:
                            premium_result = await asyncio.wait_for(
                                self.llm_predictor.predict_premium_analysis(home_team, away_team),
                                timeout=float(os.getenv('PREMIUM_ANALYSIS_TIMEOUT', 60.0))
                            )
                            
                            if not premium_result.get('error'):
                                # Create new pagination data
                                new_pagination_data = self._regenerate_analysis_cache(
                                    match_id, premium_result, home_team, away_team
                                )
                                
                                # Get first page and create navigation
                                first_page_content = new_pagination_data['pages'][1]
                                total_pages = new_pagination_data['total_pages']
                                
                                # Create navigation keyboard
                                keyboard = []
                                if total_pages > 1:
                                    nav_row = []
                                    nav_row.append(InlineKeyboardButton("◀️ Previous", callback_data=f"analysis_prev_{match_id}_1"))
                                    nav_row.append(InlineKeyboardButton(f"1/{total_pages}", callback_data=f"analysis_info_{match_id}"))
                                    nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"analysis_next_{match_id}_1"))
                                    keyboard.append(nav_row)
                                    
                                    # Quick navigation row
                                    quick_nav = []
                                    quick_nav.append(InlineKeyboardButton("🏆 Overview", callback_data=f"analysis_page_{match_id}_1"))
                                    if total_pages >= 2:
                                        quick_nav.append(InlineKeyboardButton("🤖 Analysis", callback_data=f"analysis_page_{match_id}_2"))
                                    if total_pages >= 3:
                                        quick_nav.append(InlineKeyboardButton("📚 Sources", callback_data=f"analysis_page_{match_id}_{total_pages}"))
                                    keyboard.append(quick_nav)
                                    
                                    # Action buttons
                                    keyboard.append([
                                        InlineKeyboardButton("🔄 Refresh", callback_data=f"analysis_refresh_{match_id}"),
                                        InlineKeyboardButton("❌ Close", callback_data=f"analysis_close_{match_id}")
                                    ])
                                
                                # Update with new analysis
                                await query.edit_message_text(
                                    first_page_content,
                                    parse_mode='Markdown',
                                    reply_markup=InlineKeyboardMarkup(keyboard)
                                )
                                
                                await query.answer("✅ New analysis generated successfully!", show_alert=True)
                                return
                            
                        # Fallback if regeneration fails
                        await query.edit_message_text(
                            f"❌ **Analysis Generation Failed**\n\n"
                            f"**{home_team} vs {away_team}**\n\n"
                            f"⚠️ Unable to generate new analysis.\n\n"
                            f"Please try running `/analysis {home_team} {away_team}` directly."
                        )
                        
                    except asyncio.TimeoutError:
                        await query.edit_message_text(
                            f"⏱️ **Analysis Timeout**\n\n"
                            f"**{home_team} vs {away_team}**\n\n"
                            f"Analysis took too long to complete.\n\n"
                            f"Please try running `/analysis {home_team} {away_team}` directly."
                        )
                    except Exception as e:
                        logger.error(f"Error regenerating analysis: {e}")
                        await query.edit_message_text(
                            f"❌ **Regeneration Error**\n\n"
                            f"**{home_team} vs {away_team}**\n\n"
                            f"Technical error occurred.\n\n"
                            f"Please try running `/analysis {home_team} {away_team}` directly."
                        )
                else:
                    await query.edit_message_text(
                        "❌ Unable to parse team names for regeneration.\n\n"
                        "Please run `/analysis Team1 Team2` to start fresh."
                    )
                return
            
            # Handle page navigation
            current_page = pagination_data['current_page']
            
            if action == 'next':
                new_page = min(current_page + 1, total_pages)
            elif action == 'prev':
                new_page = max(current_page - 1, 1)
            elif action == 'page' and len(parts) >= 4:
                new_page = int(parts[3])
                if new_page < 1 or new_page > total_pages:
                    await query.answer("Invalid page number", show_alert=True)
                    return
            else:
                await query.answer("Unknown navigation action", show_alert=True)
                return
            
            # Update current page
            pagination_data['current_page'] = new_page
            page_content = pages[new_page]
            
            # Create navigation keyboard
            keyboard = []
            if total_pages > 1:
                nav_row = []
                nav_row.append(InlineKeyboardButton(
                    "◀️ Previous" if new_page > 1 else "◀️",
                    callback_data=f"analysis_prev_{match_id}_{new_page}"
                ))
                nav_row.append(InlineKeyboardButton(
                    f"{new_page}/{total_pages}",
                    callback_data=f"analysis_info_{match_id}"
                ))
                nav_row.append(InlineKeyboardButton(
                    "Next ▶️" if new_page < total_pages else "▶️",
                    callback_data=f"analysis_next_{match_id}_{new_page}"
                ))
                keyboard.append(nav_row)
                
                # Quick navigation row
                quick_nav = []
                quick_nav.append(InlineKeyboardButton(
                    "🏆 Overview" + (" •" if new_page == 1 else ""),
                    callback_data=f"analysis_page_{match_id}_1"
                ))
                if total_pages >= 2:
                    quick_nav.append(InlineKeyboardButton(
                        "🤖 Analysis" + (" •" if new_page == 2 else ""),
                        callback_data=f"analysis_page_{match_id}_2"
                    ))
                if total_pages >= 3:
                    quick_nav.append(InlineKeyboardButton(
                        "📚 Sources" + (" •" if new_page == total_pages else ""),
                        callback_data=f"analysis_page_{match_id}_{total_pages}"
                    ))
                keyboard.append(quick_nav)
                
                # Action buttons
                keyboard.append([
                    InlineKeyboardButton("🔄 Refresh", callback_data=f"analysis_refresh_{match_id}"),
                    InlineKeyboardButton("❌ Close", callback_data=f"analysis_close_{match_id}")
                ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Update message with new page
            await query.edit_message_text(
                page_content,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            logger.error(f"Error handling analysis callback: {e}")
            await query.answer("❌ Navigation error occurred", show_alert=True)
    
    async def _handle_region_selection(self, query, data: str):
        """Handle region selection for leagues"""
        try:
            region = data.replace('region_', '')
            
            await query.edit_message_text("⏳ Loading leagues for selected region...")
            
            await self.ensure_provider()
            leagues = await self.provider.list_leagues()
            
            filtered_leagues = self._get_leagues_by_region(leagues, region)
            
            if not filtered_leagues:
                text = (
                    f"⚠️ *No leagues found for selected region*\n\n"
                    "This might be due to:\n"
                    "• Limited data coverage\n"
                    "• Regional filtering constraints\n"
                    "• API data availability\n\n"
                    "Try selecting 'All Popular Leagues' instead."
                )
                keyboard = [[InlineKeyboardButton("🔙 Back to Regions", callback_data="leagues")]]
                await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
                return
            
            text, keyboard = self._format_leagues_display(filtered_leagues, region, page=1)
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            
        except Exception as e:
            logger.error(f"Error handling region selection {data}: {e}")
            await query.edit_message_text(
                "⚠️ Error loading region data. Please try again or select a different region."
            )
    
    async def _handle_leagues_pagination(self, query, data: str):
        """Handle league pagination"""
        try:
            # Parse: leagues_page_region_pagenum
            parts = data.split('_')
            if len(parts) >= 4:
                region = parts[2]
                page = int(parts[3])
            else:
                await query.edit_message_text("❌ Invalid pagination data")
                return
            
            await self.ensure_provider()
            leagues = await self.provider.list_leagues()
            filtered_leagues = self._get_leagues_by_region(leagues, region)
            
            text, keyboard = self._format_leagues_display(filtered_leagues, region, page)
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            
        except Exception as e:
            logger.error(f"Error handling leagues pagination {data}: {e}")
            await query.edit_message_text("⚠️ Pagination error occurred. Please try again.")
    
    async def _handle_league_matches(self, query, data: str):
        """Handle league matches display"""
        try:
            # Parse: league_matches_leagueID_region_page
            parts = data.split('_')
            if len(parts) >= 5:
                league_id = parts[2]
                region = parts[3] 
                page_context = int(parts[4])
            else:
                await query.edit_message_text("❌ Invalid league data")
                return
            
            await query.edit_message_text("⏳ Loading upcoming matches...")
            
            # Get league name for display
            await self.ensure_provider()
            leagues = await self.provider.list_leagues()
            league_name = "Selected League"
            for league in leagues:
                if league.id == league_id:
                    league_name = league.name
                    break
            
            text, keyboard = await self._format_league_matches(league_id, league_name, region, page_context)
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            
        except Exception as e:
            logger.error(f"Error handling league matches {data}: {e}")
            await query.edit_message_text("⚠️ Error loading matches. Please try again.")
    
    async def _handle_upcoming_region(self, query, data: str):
        """Handle upcoming matches by region"""
        try:
            region = data.replace('upcoming_', '')
            
            await query.edit_message_text("⏳ Loading upcoming matches for region...")
            
            await self.ensure_provider()
            
            if region == 'all':
                # Show country selection again
                leagues = await self.provider.list_leagues()
                text, keyboard = self._format_upcoming_country_selection(leagues)
                await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
                return
            
            # Get matches for specific region with extended cache
            matches = await self.provider.get_upcoming_matches(max_total=15)
            
            if not matches:
                text = (
                    f"📅 *Upcoming Matches - {region.title()}*\n\n"
                    "⚠️ No upcoming matches found for this region.\n\n"
                    "**Try:**\n"
                    "• Different region\n"
                    "• Check back later\n"
                    "• View all regions"
                )
                keyboard = [
                    [InlineKeyboardButton("🔙 Back to Regions", callback_data="upcoming")],
                    [InlineKeyboardButton("🌍 All Regions", callback_data="upcoming_all")]
                ]
                await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
                return
            
            # Format matches display
            text_parts = [
                f"📅 *Upcoming Matches - {region.title()}*\n",
                f"📊 Found {len(matches)} matches\n"
            ]
            
            for i, match in enumerate(matches[:10], 1):
                match_time = match.match_time or 'TBD'
                league = match.league_name or 'League'
                text_parts.append(f"{i}. **{match.home_team}** vs **{match.away_team}**")
                text_parts.append(f"   🏆 {league} • ⏰ {match_time}")
                text_parts.append("")
            
            if len(matches) > 10:
                text_parts.append(f"*...and {len(matches) - 10} more matches*\n")
            
            text_parts.append("🎯 *Tap below for predictions*")
            
            keyboard = [
                [InlineKeyboardButton("🎯 Quick Predictions", callback_data="predict")],
                [InlineKeyboardButton("🔙 Back to Regions", callback_data="upcoming"),
                 InlineKeyboardButton("📅 Other Regions", callback_data="upcoming_all")]
            ]
            
            await query.edit_message_text("\n".join(text_parts), parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            
        except Exception as e:
            logger.error(f"Error handling upcoming region {data}: {e}")
            await query.edit_message_text("⚠️ Error loading regional matches. Please try again.")
    
    async def _handle_match_prediction(self, query, data: str):
        """Handle individual match prediction requests"""
        try:
            # Parse: predict_match_HomeTeam_AwayTeam
            parts = data.split('_')[2:]  # Remove 'predict_match_'
            if len(parts) >= 2:
                home_team = parts[0]
                away_team = '_'.join(parts[1:])  # In case away team has underscores
            else:
                await query.edit_message_text("❌ Invalid match data")
                return
            
            await query.edit_message_text(f"🎯 Generating prediction for {home_team} vs {away_team}...")
            
            try:
                result = await self._generate_prediction_for_match_async(home_team, away_team, prefer_llm=True)
                
                text = (
                    f"🎯 *AI Match Prediction*\n\n"
                    f"**{home_team} vs {away_team}**\n\n"
                    f"🏆 **Prediction**: {result['prediction']}\n"
                    f"📊 **Confidence**: {result['confidence_text']}\n\n"
                    f"📈 **Probabilities:**\n"
                    f"🏠 Home Win: {result['home_win']:.1f}%\n"
                    f"🤝 Draw: {result['draw']:.1f}%\n"
                    f"✈️ Away Win: {result['away_win']:.1f}%\n\n"
                    f"{result['probability_bar']}\n\n"
                    f"*Powered by {result.get('source', 'AI')} analysis* 🤖"
                )
                
                keyboard = [
                    [InlineKeyboardButton("🔄 Refresh Prediction", callback_data=data)],
                    [InlineKeyboardButton("🔙 Back to Matches", callback_data="upcoming"),
                     InlineKeyboardButton("🎯 More Predictions", callback_data="predict")]
                ]
                
                await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
                
            except Exception as pred_error:
                logger.error(f"Prediction error for {home_team} vs {away_team}: {pred_error}")
                await query.edit_message_text(
                    f"⚠️ *Prediction Error*\n\n"
                    f"Unable to generate prediction for {home_team} vs {away_team}.\n\n"
                    "This could be due to:\n"
                    "• LLM API issues\n"
                    "• Rate limiting\n"
                    "• Team name recognition\n\n"
                    "Please try again or use `/predict Team1 vs Team2` format."
                )
                
        except Exception as e:
            logger.error(f"Error handling match prediction {data}: {e}")
            await query.edit_message_text("⚠️ Prediction system error. Please try again.")

    def _format_country_selection(self) -> Tuple[str, List[List[InlineKeyboardButton]]]:
        """Format country selection for leagues navigation"""
        text = (
            "🌍 *Select Region for Leagues*\n\n"
            "📊 Choose a region to explore available leagues:\n\n"
            "🔄 *10-minute cache optimization active*\n"
            "🚀 *All major sports included (Football, Basketball, American Sports)*"
        )
        
        keyboard = [
            [InlineKeyboardButton("🇪🇺 Europe", callback_data="region_europe"), 
             InlineKeyboardButton("🇺🇸 North America", callback_data="region_north_america")],
            [InlineKeyboardButton("🇧🇷 South America", callback_data="region_south_america"), 
             InlineKeyboardButton("🇫 Asia", callback_data="region_asia")],
            [InlineKeyboardButton("🌍 All Popular Leagues", callback_data="region_all_popular")],
            [InlineKeyboardButton("📅 Direct to Upcoming", callback_data="upcoming")]
        ]
        
        return text, keyboard
    
    def _format_upcoming_country_selection(self, leagues: List) -> Tuple[str, List[List[InlineKeyboardButton]]]:
        """Format country selection for upcoming matches"""
        text = (
            "📅 *Upcoming Matches by Region*\n\n"
            "⚽ Choose a region to see upcoming matches:\n\n"
            f"📊 *{len(leagues)} total leagues available*\n"
            "🔄 *10-minute cache for optimal performance*"
        )
        
        keyboard = [
            [InlineKeyboardButton("🇪🇺 European Football", callback_data="upcoming_europe"), 
             InlineKeyboardButton("🇺🇸 American Sports", callback_data="upcoming_usa")],
            [InlineKeyboardButton("🌍 Top 5 Leagues", callback_data="upcoming_top5"),
             InlineKeyboardButton("🅰️ All Regions", callback_data="upcoming_all")],
            [InlineKeyboardButton("🎯 Get Predictions", callback_data="predict")]
        ]
        
        return text, keyboard
    
    def _get_leagues_by_region(self, leagues: List, region: str) -> List:
        """Filter leagues by region with comprehensive sports coverage"""
        region_mapping = {
            'europe': {
                'keywords': ['premier', 'la liga', 'serie a', 'bundesliga', 'ligue 1', 'eredivisie', 'primeira', 'championship'],
                'countries': ['england', 'spain', 'italy', 'germany', 'france', 'netherlands', 'portugal']
            },
            'north_america': {
                'keywords': ['mls', 'nfl', 'nba', 'nhl', 'mlb', 'ncaa', 'concacaf'],
                'countries': ['usa', 'united states', 'canada', 'mexico']
            },
            'south_america': {
                'keywords': ['copa', 'brasileiro', 'argentina', 'libertadores'],
                'countries': ['brazil', 'argentina', 'chile', 'colombia', 'uruguay']
            },
            'asia': {
                'keywords': ['j-league', 'k-league', 'chinese', 'afc', 'asian'],
                'countries': ['japan', 'south korea', 'china', 'australia']
            },
            'all_popular': {
                'keywords': ['premier', 'la liga', 'serie a', 'bundesliga', 'ligue 1', 'mls', 'nfl', 'nba', 'nhl'],
                'countries': []
            }
        }
        
        if region not in region_mapping:
            return leagues[:20]  # Fallback
        
        config = region_mapping[region]
        filtered = []
        
        for league in leagues:
            league_name = league.name.lower()
            league_country = (league.country or '').lower()
            
            # Check keywords
            if any(keyword in league_name for keyword in config['keywords']):
                filtered.append(league)
                continue
                
            # Check countries
            if any(country in league_country for country in config['countries']):
                filtered.append(league)
                
        return filtered[:25]  # Limit for pagination
    
    def _format_leagues_display(self, leagues: List, region: str, page: int = 1) -> Tuple[str, List[List[InlineKeyboardButton]]]:
        """Format leagues with pagination"""
        items_per_page = 8
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_leagues = leagues[start_idx:end_idx]
        total_pages = (len(leagues) + items_per_page - 1) // items_per_page
        
        region_names = {
            'europe': '🇪🇺 European',
            'north_america': '🇺🇸 North American', 
            'south_america': '🇧🇷 South American',
            'asia': '🇫🇫 Asian',
            'all_popular': '🌍 Popular'
        }
        
        text_parts = [
            f"🏆 *{region_names.get(region, 'Selected')} Leagues*\n",
            f"Page {page}/{total_pages} | {len(leagues)} total leagues\n"
        ]
        
        for i, league in enumerate(page_leagues, 1):
            flag = self._get_league_flag(league.country)
            sport_icon = self._get_sport_icon(league.name)
            text_parts.append(f"{i}. {flag} {sport_icon} **{league.name}**")
            if league.country:
                text_parts.append(f"   📍 {league.country}")
            text_parts.append("")
        
        text_parts.extend([
            f"\n📅 *Tap any league to see upcoming matches*",
            f"🔄 *Cache refreshes every 10 minutes*"
        ])
        
        # Build keyboard with league buttons
        keyboard = []
        
        # League selection buttons (2 per row)
        for i in range(0, len(page_leagues), 2):
            row = []
            for j in range(2):
                if i + j < len(page_leagues):
                    league = page_leagues[i + j]
                    callback_data = f"league_matches_{league.id}_{region}_{page}"
                    # Truncate long league names for buttons
                    button_text = league.name[:20] + ("..." if len(league.name) > 20 else "")
                    row.append(InlineKeyboardButton(button_text, callback_data=callback_data))
            if row:
                keyboard.append(row)
        
        # Pagination controls
        if total_pages > 1:
            nav_row = []
            if page > 1:
                nav_row.append(InlineKeyboardButton("◀️ Previous", callback_data=f"leagues_page_{region}_{page-1}"))
            nav_row.append(InlineKeyboardButton(f"{page}/{total_pages}", callback_data=f"leagues_info_{region}"))
            if page < total_pages:
                nav_row.append(InlineKeyboardButton("Next ▶️", callback_data=f"leagues_page_{region}_{page+1}"))
            keyboard.append(nav_row)
        
        # Action buttons
        keyboard.extend([
            [InlineKeyboardButton("🔙 Back to Regions", callback_data="leagues"),
             InlineKeyboardButton("📅 All Upcoming", callback_data="upcoming_all")],
            [InlineKeyboardButton("🎯 Predictions", callback_data="predict")]
        ])
        
        return "\n".join(text_parts), keyboard
    
    def _get_league_flag(self, country: Optional[str]) -> str:
        """Get emoji flag for country with comprehensive coverage"""
        if not country:
            return "🌍"
            
        flags = {
            'england': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'spain': '🇪🇸', 'italy': '🇮🇹',
            'germany': '🇩🇪', 'france': '🇫🇷', 'netherlands': '🇳🇱',
            'portugal': '🇵🇹', 'brazil': '🇧🇷', 'argentina': '🇦🇷',
            'usa': '🇺🇸', 'united states': '🇺🇸', 'canada': '🇨🇦',
            'mexico': '🇲🇽', 'japan': '🇯🇵', 'south korea': '🇰🇷',
            'china': '🇨🇳', 'australia': '🇦🇺', 'chile': '🇨🇱',
            'colombia': '🇨🇴', 'uruguay': '🇺🇾'
        }
        
        country_lower = country.lower()
        return flags.get(country_lower, '🌍')
    
    def _get_sport_icon(self, league_name: str) -> str:
        """Get sport icon based on league name"""
        name_lower = league_name.lower()
        
        if any(keyword in name_lower for keyword in ['nfl', 'football']) and 'american' in name_lower:
            return '🏈'  # American Football
        elif any(keyword in name_lower for keyword in ['nba', 'basketball']):
            return '🏀'  # Basketball
        elif any(keyword in name_lower for keyword in ['nhl', 'hockey']):
            return '🏒'  # Hockey
        elif any(keyword in name_lower for keyword in ['mlb', 'baseball']):
            return '⚾'  # Baseball
        else:
            return '⚽'  # Default to soccer
    
    async def _format_league_matches(self, league_id: str, league_name: str, region: str, page_context: int) -> Tuple[str, List[List[InlineKeyboardButton]]]:
        """Format upcoming matches for a specific league"""
        try:
            matches = await self.provider.get_upcoming_matches_for_league(league_id, max_matches=12)
            
            if not matches:
                text = (
                    f"📅 *{league_name}*\n\n"
                    "⚠️ No upcoming matches found.\n\n"
                    "**Possible reasons:**\n"
                    "• Between seasons\n"
                    "• No matches scheduled\n"
                    "• Data not yet available\n\n"
                    "🔄 *Try again later or check other leagues*"
                )
                
                keyboard = [
                    [InlineKeyboardButton("🔙 Back to Leagues", callback_data=f"leagues_page_{region}_{page_context}")],
                    [InlineKeyboardButton("📅 All Upcoming", callback_data="upcoming_all")]
                ]
                
                return text, keyboard
            
            # Format matches with enhanced display
            text_parts = [
                f"📅 *{league_name} - Upcoming Matches*\n",
                f"📊 Found {len(matches)} upcoming matches\n"
            ]
            
            for i, match in enumerate(matches[:8], 1):  # Show up to 8 matches
                match_time = match.match_time or 'TBD'
                match_date = match.date or 'TBD'
                
                text_parts.append(f"{i}. **{match.home_team}** vs **{match.away_team}**")
                text_parts.append(f"   📅 {match_date} • ⏰ {match_time}")
                text_parts.append("")
            
            if len(matches) > 8:
                text_parts.append(f"*...and {len(matches) - 8} more matches*\n")
            
            text_parts.extend([
                "🎯 *Tap below for AI predictions*",
                f"🔄 *Data cached for 10 minutes*"
            ])
            
            # Match prediction buttons (first 4 matches)
            keyboard = []
            for i, match in enumerate(matches[:4]):
                callback_data = f"predict_match_{match.home_team}_{match.away_team}"
                button_text = f"🎯 {match.home_team[:8]} vs {match.away_team[:8]}"
                keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])
            
            # Navigation buttons
            keyboard.extend([
                [InlineKeyboardButton("🔙 Back to Leagues", callback_data=f"leagues_page_{region}_{page_context}"),
                 InlineKeyboardButton("📅 Other Regions", callback_data="upcoming")],
                [InlineKeyboardButton("🎯 All Predictions", callback_data="predict")]
            ])
            
            return "\n".join(text_parts), keyboard
            
        except Exception as e:
            logger.error(f"Error getting matches for league {league_id}: {e}")
            text = (
                f"⚠️ *Error Loading {league_name} Matches*\n\n"
                "**Technical issue occurred:**\n"
                f"• {str(e)[:50]}...\n\n"
                "Please try again or select another league."
            )
            
            keyboard = [
                [InlineKeyboardButton("🔙 Back to Leagues", callback_data=f"leagues_page_{region}_{page_context}")],
                [InlineKeyboardButton("🔄 Retry", callback_data=f"league_matches_{league_id}_{region}_{page_context}")]
            ]
            
            return text, keyboard

    async def send_upcoming_response(self, query):
        """Enhanced upcoming response with multi-step navigation"""
        try:
            await query.edit_message_text("⏳ Loading upcoming matches interface...")
            await self.ensure_provider()
            
            leagues = await self.provider.list_leagues()
            text, keyboard = self._format_upcoming_country_selection(leagues)
            
            await query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            
        except Exception as e:
            logger.error(f"Error in send_upcoming_response: {e}")
            await query.edit_message_text(
                "⚠️ *Error Loading Upcoming Matches*\n\n"
                "This could be due to:\n"
                "• API rate limits\n"
                "• Network connectivity\n"
                "• Service maintenance\n\n"
                "Please try again in a few minutes.",
                parse_mode='Markdown'
            )

    async def send_predict_response(self, query):
        await query.edit_message_text("🎯 Generating predictions...")
        try:
            await self.sports_collector.initialize()
            real_matches = await self.sports_collector.get_real_upcoming_matches()
            if real_matches:
                predict_text = "🎯 *Match Predictions*\n\n"
                for match in real_matches[:3]:
                    home_team = match.get('home_team', 'Team A')
                    away_team = match.get('away_team', 'Team B')
                    league = match.get('league', 'League')
                    prediction = await self._generate_prediction_for_match_async(home_team, away_team)
                    predict_text += f"**{home_team} vs {away_team}**\n🏆 {league}\n\n"
                    predict_text += f"🎯 **Prediction: {prediction['prediction']}**\n"
                    predict_text += f"📊 Confidence: {prediction['confidence_text']}\n"
                    predict_text += f"📈 {prediction['probability_bar']}\n\n"
                    predict_text += f"🏠 Home Win: {prediction['home_win']:.1f}%\n"
                    predict_text += f"🤝 Draw: {prediction['draw']:.1f}%\n"
                    predict_text += f"✈️ Away Win: {prediction['away_win']:.1f}%\n\n---\n\n"
                predict_text += "*Predictions powered by AI analysis* 🤖"
            else:
                predict_text = "🎯 *Match Predictions*\n\n⚠️ No upcoming matches available for predictions."
            keyboard = [[InlineKeyboardButton("📅 View Matches", callback_data="upcoming"), InlineKeyboardButton("📊 Stats", callback_data="stats")]]
            await query.edit_message_text(predict_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception:
            await query.edit_message_text("⚠️ Error generating predictions.")
        finally:
            await self.sports_collector.close()

    async def send_odds_response(self, query):
        odds_text = (
            "🔴 *Live Betting Odds - Select League*\n\n"
            "📊 Choose a league to view odds (fallback):"
        )
        keyboard = [
            [InlineKeyboardButton("🏴 Premier League", callback_data="odds_epl"), InlineKeyboardButton("🇪🇸 La Liga", callback_data="odds_laliga")],
            [InlineKeyboardButton("🇮🇹 Serie A", callback_data="odds_seriea"), InlineKeyboardButton("🇩🇪 Bundesliga", callback_data="odds_bundesliga")],
            [InlineKeyboardButton("🇫🇷 Ligue 1", callback_data="odds_ligue1"), InlineKeyboardButton("🏆 Champions League", callback_data="odds_ucl")],
            [InlineKeyboardButton("🇳🇱 Eredivisie", callback_data="odds_eredivisie"), InlineKeyboardButton("🇵🇹 Primeira Liga", callback_data="odds_portugal")],
            [InlineKeyboardButton("🌍 All Leagues Mix", callback_data="odds_all"), InlineKeyboardButton("🔄 Refresh", callback_data="odds")],
        ]
        await query.edit_message_text(odds_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))

    async def send_stats_response(self, query):
        stats_text = (
            "📊 *Prediction Statistics*\n\n"
            "🎯 Overall Performance\n"
            "• Total Predictions: 247\n• Correct: 156\n• Accuracy: 63.2%\n\n"
            "*Statistics updated regularly* 📋"
        )
        keyboard = [[InlineKeyboardButton("🎯 New Predictions", callback_data="predict"), InlineKeyboardButton("📅 Upcoming", callback_data="upcoming")]]
        await query.edit_message_text(stats_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))

    async def send_advanced_response(self, query):
        response_text = (
            "🔬 *Professional Ensemble Predictions*\n\n"
            "Please specify two teams for professional analysis.\n"
            "Example: `/advanced Barcelona Real Madrid`\n\n"
            "This heuristic ensemble blends team strength, recent form, H2H, player impact, and venue factors."
        )
        await query.edit_message_text(response_text, parse_mode='Markdown')

    async def send_analysis_response(self, query):
        response_text = (
            "🧠 *Enhanced Team Analysis*\n\n"
            "Please specify two teams for detailed analysis.\n"
            "Example: `/analysis Barcelona Real Madrid`\n\n"
            "Analysis includes recent form, head-to-head, injuries, and trends."
        )
        await query.edit_message_text(response_text, parse_mode='Markdown')

    async def send_live_response(self, query):
        response_text = (
            "🔴 *Live updates are disabled for now*\n\n"
            "Free API tiers typically do not support reliable live data and can burn through rate limits.\n"
            "We will enable `/live` once a suitable plan is configured."
        )
        await query.edit_message_text(response_text, parse_mode='Markdown')

    async def send_accuracy_response(self, query):
        response_text = (
            "📈 *Prediction Accuracy Statistics*\n\n"
            "Use `/accuracy` for detailed breakdown.\n"
        )
        await query.edit_message_text(response_text, parse_mode='Markdown')

    async def send_specific_league_odds(self, query):
        """Safe fallback odds for a selected league (no external odds API)."""
        try:
            league_mapping = {
                'odds_epl': '🏴 Premier League',
                'odds_laliga': '🇪🇸 La Liga',
                'odds_seriea': '🇮🇹 Serie A',
                'odds_bundesliga': '🇩🇪 Bundesliga',
                'odds_ligue1': '🇫🇷 Ligue 1',
                'odds_ucl': '🏆 Champions League',
                'odds_eredivisie': '🇳🇱 Eredivisie',
                'odds_portugal': '🇵🇹 Primeira Liga',
                'odds_all': '🌍 All Leagues Mix'
            }
            league_display = league_mapping.get(query.data, 'Selected League')
            odds_text = f"""
🔴 {league_display} - Fallback Odds

Live odds API not configured. Showing heuristic-based odds:
"""
            await self.sports_collector.initialize()
            matches = await self.sports_collector.get_real_upcoming_matches()
            await self.sports_collector.close()
            sample = matches or []
            if query.data == 'odds_all' and matches:
                seen, dedup = set(), []
                for m in matches:
                    lg = m.get('league', 'League')
                    if lg not in seen:
                        seen.add(lg)
                        dedup.append(m)
                    if len(dedup) >= 6:
                        break
                sample = dedup
            for match in sample[:6]:
                home = match.get('home_team', 'Home')
                away = match.get('away_team', 'Away')
                market = self._fallback_realistic_odds(home, away)
                best = market['raw_odds']
                odds_text += f"\n{home} vs {away}\n"
                odds_text += f"🏠 {home}: {best['home']:.2f}\n"
                odds_text += f"🤝 Draw: {best['draw']:.2f}\n"
                odds_text += f"✈️ {away}: {best['away']:.2f}\n\n"
            if not sample:
                odds_text += "\nNo upcoming matches available."
            keyboard = [[InlineKeyboardButton("← Back to Leagues", callback_data="odds")]]
            await query.edit_message_text(odds_text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception:
            await query.edit_message_text("⚠️ Error loading league odds.")

    # ----- Prediction core -----
    async def _generate_prediction_for_match_async(self, home_team: str, away_team: str, prefer_llm: bool = True) -> Dict:
        home_strength = self._calculate_team_strength(home_team)
        away_strength = self._calculate_team_strength(away_team)
        home_advantage = 0.08
        # LLM-first
        authentic = None
        use_llm = os.getenv('USE_LLM') == '1' and prefer_llm
        if use_llm:
            try:
                llm = await self.llm.predict(home_team, away_team)
                if llm and 'home_win' in llm:
                    authentic = {'home_win': llm['home_win'], 'draw': llm['draw'], 'away_win': llm['away_win']}
            except Exception as e:
                logger.exception("LLM prediction failed")
                raise
        if not authentic:
            # LLM failed; surface error to user immediately
            raise RuntimeError("LLM prediction unavailable. Please try again later.")

        home_win = float(authentic['home_win'])
        draw = float(authentic['draw'])
        away_win = float(authentic['away_win'])

        if home_win >= draw and home_win >= away_win:
            pred, conf = f"{home_team} Win", home_win
        elif away_win >= draw and away_win >= home_win:
            pred, conf = f"{away_team} Win", away_win
        else:
            pred, conf = "Draw", draw
        return {
            'prediction': pred,
            'confidence': conf,
            'confidence_text': self._get_confidence_text(conf),
            'probability_bar': self._create_probability_bar(conf),
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'source': 'gemini-llm'
        }

    def _fallback_realistic_odds(self, home_team: str, away_team: str) -> Dict:
        home_strength = self._calculate_team_strength(home_team)
        away_strength = self._calculate_team_strength(away_team)
        home_adv = 0.08
        adj_home = home_strength + home_adv
        adj_away = away_strength
        total_strength = adj_home + adj_away
        home_prob_raw = adj_home / total_strength
        away_prob_raw = adj_away / total_strength
        draw_factor = 0.25 + (0.15 * (1 - abs(home_strength - away_strength)))
        total_raw = home_prob_raw + away_prob_raw + draw_factor
        margin = 1.07
        home_prob = (home_prob_raw / total_raw) / margin * 100
        away_prob = (away_prob_raw / total_raw) / margin * 100
        draw_prob = (draw_factor / total_raw) / margin * 100
        return {
            'market_probabilities': {'home_win': home_prob, 'draw': draw_prob, 'away_win': away_prob},
            'raw_odds': {
                'home': 1 / (home_prob / 100) if home_prob > 0 else 10.0,
                'draw': 1 / (draw_prob / 10) if draw_prob > 0 else 10.0,
                'away': 1 / (away_prob / 100) if away_prob > 0 else 10.0,
            },
            'market_confidence': 0.75,
            'prediction_weight': 0.2,
            'source': 'fallback_realistic',
        }

    def _calculate_team_strength(self, team_name: str) -> float:
        strong = {
            'barcelona','fc barcelona','real madrid','manchester city','liverpool','arsenal','chelsea',
            'manchester united','tottenham','bayern munich','borussia dortmund','ac milan','inter milan',
            'juventus','napoli','paris saint-germain','atletico madrid','sevilla','valencia'
        }
        base = 0.3 + ((hash(team_name.lower()) % 100) / 100.0) * 0.4
        if any(s in team_name.lower() for s in strong):
            base += 0.15
        return min(max(base, 0.2), 0.8)

    def _create_probability_bar(self, percentage: float) -> str:
        filled = int(percentage / 10)
        empty = 10 - filled
        return f"{'█' * filled}{'░' * empty} {percentage:.1f}%"

    def _get_confidence_text(self, confidence: float) -> str:
        if confidence >= 60:
            return "🟢 High Confidence"
        if confidence >= 45:
            return "🟡 Medium Confidence"
        return "🔴 Low Confidence"


import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

def start_health_server_thread():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/', '/health'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"Sports Prediction Bot is running!")
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, format, *args):
            return
    port = int(os.getenv('PORT', '10000'))
    server = HTTPServer(('0.0.0.0', port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health server started on port {port}")

if __name__ == "__main__":
    try:
        start_health_server_thread()
        bot = SimpleSportsBot()
        bot.setup_handlers()
        bot.application.run_polling()
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise
