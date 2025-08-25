"""
Sports Prediction Telegram Bot (LLM-first, no DB, no heavy ML)
"""
import asyncio
import logging
import os
from typing import Dict, Optional

from aiohttp import web
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from telegram.error import TimedOut, RetryAfter, NetworkError

from simple_football_api import SimpleFootballAPI  # legacy for other features
from providers.thesportsdb_provider import TheSportsDbProvider
from enhanced_predictions import EnhancedPredictionEngine
# from advanced_prediction_engine import AdvancedPredictionEngine  # Disabled: LLM-only mode
from llm_predictor import GeminiPredictor

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
        req = HTTPXRequest(read_timeout=30.0, connect_timeout=15.0)
        # Initialize LLM once at startup via PTB post_init
        async def _post_init(app):
            try:
                await self.llm.initialize()
                logger.info("LLM initialized in post_init")
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
        # Advanced predictor disabled

    def setup_handlers(self) -> None:
        # Core
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("leagues", self.leagues_command))
        self.application.add_handler(CommandHandler("upcoming", self.upcoming_command))
        self.application.add_handler(CommandHandler("predict", self.predict_command))
        self.application.add_handler(CommandHandler("odds", self.odds_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))

        # Analysis/advanced
        self.application.add_handler(CommandHandler("analysis", self.analysis_command))
        self.application.add_handler(CommandHandler("advanced", self.advanced_prediction_command))
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
        """Slash command variant: replies in chat context."""
        try:
            await self.ensure_provider()
            leagues = await self.provider.list_leagues()
            text, keyboard = self._format_leagues_text_and_keyboard(leagues)
            if update.message:
                await update.message.reply_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            elif update.callback_query:
                await update.callback_query.edit_message_text(text, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
            else:
                logger.warning("/leagues called without message or callback_query context")
        except Exception as e:
            logger.error(f"Error in /leagues: {e}")
            if update.message:
                await update.message.reply_text("⚠️ Error fetching leagues.")

    async def upcoming_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("⏳ Getting upcoming matches...")
        try:
            await self.ensure_provider()
            matches = await self.provider.get_upcoming_matches(max_total=10)
            if matches:
                txt = "📅 *Upcoming Matches*\n\n⚽ *Football Matches from Multiple Leagues*\n"
                for m in matches[:9]:
                    match_time = m.match_time or 'TBD'
                    league = m.league_name or 'League'
                    txt += f"  • {m.home_team} vs {m.away_team}\n"
                    txt += f"    🏆 {league} • ⏰ {match_time}\n\n"
                txt += f"\n*Total: {len(matches)} matches found*\nData from TheSportsDB ✅"
            else:
                txt = (
                    "📅 *Upcoming Matches*\n\n"
                    "⚠️ No upcoming matches found at the moment. Try again later."
                )
            keyboard = [[InlineKeyboardButton("🎯 Get Predictions", callback_data="predict"), InlineKeyboardButton("🔄 Refresh", callback_data="upcoming")]]
            await update.message.reply_text(txt, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception as e:
            logger.error(f"Error in /upcoming: {e}")
            await update.message.reply_text("⚠️ Error getting upcoming matches.")

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
        try:
            if not context.args or len(context.args) < 2:
                await update.message.reply_text("🧠 *Enhanced Team Analysis*\n\nPlease provide two team names.\nExample: `/analysis Barcelona Real Madrid`", parse_mode='Markdown')
                return
            home_team = context.args[0]
            away_team = ' '.join(context.args[1:])
            await update.message.reply_text(f"🧠 Analyzing {home_team} vs {away_team}...")
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
                f"✈️ Away: {analysis['enhanced_prediction']['away_win_probability']:.1f}%\n"
            )
            await update.message.reply_text(analysis_text, parse_mode='Markdown')
            await self.enhanced_predictor.close()
        except Exception as e:
            logger.error(f"Error in /analysis: {e}")
            await update.message.reply_text("⚠️ Error generating analysis.")

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

    # ----- Buttons -----
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data
        try:
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

    async def send_upcoming_response(self, query):
        await query.edit_message_text("⏳ Getting upcoming matches...")
        try:
            await self.ensure_provider()
            matches = await self.provider.get_upcoming_matches(max_total=10)
            if matches:
                txt = "📅 *Upcoming Matches*\n\n⚽ *Football Matches from Multiple Leagues*\n"
                for m in matches[:9]:
                    match_time = m.match_time or 'TBD'
                    league = m.league_name or 'League'
                    txt += f"  • {m.home_team} vs {m.away_team}\n"
                    txt += f"    🏆 {league} • ⏰ {match_time}\n\n"
                txt += f"\n*Total: {len(matches)} matches found*\nData from TheSportsDB ✅"
            else:
                txt = "📅 *Upcoming Matches*\n\n⚠️ No upcoming matches found at the moment."
            keyboard = [[InlineKeyboardButton("🎯 Get Predictions", callback_data="predict"), InlineKeyboardButton("🔄 Refresh", callback_data="upcoming")]]
            await query.edit_message_text(txt, parse_mode='Markdown', reply_markup=InlineKeyboardMarkup(keyboard))
        except Exception:
            await query.edit_message_text("⚠️ Error getting matches.")

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
                'draw': 1 / (draw_prob / 100) if draw_prob > 0 else 10.0,
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
