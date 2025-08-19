"""
Telegram bot command handlers.
"""
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler, filters
from .responses import (
    get_start_message,
    get_help_message,
    get_leagues_message,
    get_upcoming_message,
    get_predict_message,
    get_odds_message,
    get_stats_message,
    get_advanced_message,
    get_analysis_message,
    get_live_message,
    get_accuracy_message,
    get_community_message,
    get_deepml_message,
    get_specific_league_odds_message,
    get_dashboard_message,
    get_leaderboard_message,
    get_feed_message,
    get_badges_message,
    get_deepml1_message,
    get_deepml2_message,
    get_deepml3_message,
    get_deepml4_message,
    get_deepml5_message,
    get_god_ensemble_message,
)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    text, reply_markup = get_start_message()
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    text, reply_markup = get_help_message()
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

async def leagues_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /leagues command"""
    text, reply_markup = get_leagues_message()
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

async def upcoming_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /upcoming command"""
    await update.message.reply_text("⏳ Getting live upcoming matches...")
    text, reply_markup = await get_upcoming_message()
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /predict command"""
    await update.message.reply_text("🎯 Generating predictions...")
    text, reply_markup = await get_predict_message()
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

async def odds_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /odds command"""
    text, reply_markup = get_odds_message()
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command"""
    text, reply_markup = get_stats_message()
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

async def analysis_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /analysis command for enhanced team analysis"""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("🧠 *Enhanced Team Analysis*\n\nPlease provide two team names.\nExample: `/analysis Barcelona Real Madrid`", parse_mode='Markdown')
        return

    home_team = context.args[0]
    away_team = ' '.join(context.args[1:])

    await update.message.reply_text(f"🧠 Analyzing {home_team} vs {away_team}...")
    text = await get_analysis_message(home_team, away_team)
    await update.message.reply_text(text, parse_mode='Markdown')

async def live_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /live command for live match updates"""
    await update.message.reply_text("🔴 Getting live match updates...")
    text = await get_live_message()
    await update.message.reply_text(text, parse_mode='Markdown')

async def accuracy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /accuracy command for prediction accuracy stats"""
    await update.message.reply_text("📈 Getting accuracy stats...")
    text = await get_accuracy_message()
    await update.message.reply_text(text, parse_mode='Markdown')

async def advanced_prediction_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /advanced command for professional-grade predictions"""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("🔬 *Advanced Prediction System*\n\nProfessional ensemble model with 6 prediction algorithms.\nExample: `/advanced Barcelona Real Madrid`", parse_mode='Markdown')
        return

    home_team = context.args[0]
    away_team = ' '.join(context.args[1:])

    await update.message.reply_text(f"🔬 Running advanced prediction analysis for {home_team} vs {away_team}...")
    text = await get_advanced_message(home_team, away_team)
    await update.message.reply_text(text, parse_mode='Markdown')

async def deep_ml_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /deepml command for deep learning predictions"""
    if not context.args or len(context.args) < 2:
        text, reply_markup = get_deepml_message(None, None)
        await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)
        return

    home_team = " ".join(context.args[:-1])
    away_team = context.args[-1]

    text, reply_markup = get_deepml_message(home_team, away_team)
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

async def community_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /community command to access social features Mini App"""
    text, reply_markup = get_community_message()
    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()

    if query.data == "leagues":
        text, reply_markup = get_leagues_message()
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif query.data == "upcoming":
        await query.edit_message_text("⏳ Getting live upcoming matches...")
        text, reply_markup = await get_upcoming_message()
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif query.data == "predict":
        await query.edit_message_text("🎯 Generating predictions...")
        text, reply_markup = await get_predict_message()
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif query.data == "odds":
        text, reply_markup = get_odds_message()
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif query.data.startswith("odds_"):
        text, reply_markup = await get_specific_league_odds_message(query.data)
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif query.data == "advanced":
        text, _ = get_advanced_message(None, None)
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data == "deepml":
        text, _ = get_deepml_message(None, None)
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data.startswith("deepml1_"):
        home_team, away_team = query.data.split('_')[1:]
        text = await get_deepml1_message(home_team, away_team)
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data.startswith("deepml2_"):
        home_team, away_team = query.data.split('_')[1:]
        text = await get_deepml2_message(home_team, away_team)
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data.startswith("deepml3_"):
        home_team, away_team = query.data.split('_')[1:]
        text = await get_deepml3_message(home_team, away_team)
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data.startswith("deepml4_"):
        home_team, away_team = query.data.split('_')[1:]
        text = await get_deepml4_message(home_team, away_team)
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data.startswith("deepml5_"):
        home_team, away_team = query.data.split('_')[1:]
        text = await get_deepml5_message(home_team, away_team)
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data.startswith("godensemble_"):
        home_team, away_team = query.data.split('_')[1:]
        text = await get_god_ensemble_message(home_team, away_team)
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data == "analysis":
        text = get_analysis_message(None, None)
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data == "live":
        text = await get_live_message()
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data == "accuracy":
        text = await get_accuracy_message()
        await query.edit_message_text(text, parse_mode='Markdown')
    elif query.data == "stats":
        text, reply_markup = get_stats_message()
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif query.data == "community":
        text, reply_markup = get_community_message()
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif query.data == "dashboard":
        user_id = str(query.from_user.id)
        username = query.from_user.username or query.from_user.first_name or "Anonymous"
        text, reply_markup = await get_dashboard_message(user_id, username)
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif query.data == "leaderboard":
        user_id = str(query.from_user.id)
        text, reply_markup = await get_leaderboard_message(user_id)
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif query.data == "feed":
        text, reply_markup = await get_feed_message()
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    elif query.data == "badges":
        user_id = str(query.from_user.id)
        text, reply_markup = await get_badges_message(user_id)
        await query.edit_message_text(text, parse_mode='Markdown', reply_markup=reply_markup)
    else:
        await query.edit_message_text("❌ Unknown action. Please try again.")

def setup_handlers(application):
    """Setup all bot handlers"""
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("leagues", leagues_command))
    application.add_handler(CommandHandler("upcoming", upcoming_command))
    application.add_handler(CommandHandler("predict", predict_command))
    application.add_handler(CommandHandler("odds", odds_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("analysis", analysis_command))
    application.add_handler(CommandHandler("live", live_command))
    application.add_handler(CommandHandler("accuracy", accuracy_command))
    application.add_handler(CommandHandler("advanced", advanced_prediction_command))
    application.add_handler(CommandHandler("deepml", deep_ml_command))
    application.add_handler(CommandHandler("community", community_command))

    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, help_command))
