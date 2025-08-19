"""
Telegram bot keyboards.
"""
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo

def get_main_keyboard():
    """Get the main keyboard"""
    keyboard = [
        [
            InlineKeyboardButton("📋 Leagues", callback_data="leagues"),
            InlineKeyboardButton("📅 Upcoming", callback_data="upcoming")
        ],
        [
            InlineKeyboardButton("🎯 Predictions", callback_data="predict"),
            InlineKeyboardButton("💰 Live Odds", callback_data="odds")
        ],
        [
            InlineKeyboardButton("🔬 Advanced", callback_data="advanced"),
            InlineKeyboardButton("🤖 Deep Learning", callback_data="deepml")
        ],
        [
            InlineKeyboardButton("🔴 Live", callback_data="live"),
            InlineKeyboardButton("📈 Accuracy", callback_data="accuracy")
        ],
        [
            InlineKeyboardButton("👥 Community", callback_data="community"),
            InlineKeyboardButton("📊 Stats", callback_data="stats")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_leagues_keyboard():
    """Get the keyboard for the /leagues command"""
    keyboard = [
        [
            InlineKeyboardButton("📅 Upcoming Matches", callback_data="upcoming"),
            InlineKeyboardButton("🎯 Get Predictions", callback_data="predict")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_upcoming_keyboard():
    """Get the keyboard for the /upcoming command"""
    keyboard = [
        [
            InlineKeyboardButton("🎯 Get Predictions", callback_data="predict"),
            InlineKeyboardButton("🔄 Refresh", callback_data="upcoming")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_predict_keyboard():
    """Get the keyboard for the /predict command"""
    keyboard = [
        [
            InlineKeyboardButton("📅 View Matches", callback_data="upcoming"),
            InlineKeyboardButton("📊 Stats", callback_data="stats")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_odds_keyboard():
    """Get the keyboard for the /odds command"""
    keyboard = [
        [
            InlineKeyboardButton("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League", callback_data="odds_epl"),
            InlineKeyboardButton("🇪🇸 La Liga", callback_data="odds_laliga")
        ],
        [
            InlineKeyboardButton("🇮🇹 Serie A", callback_data="odds_seriea"),
            InlineKeyboardButton("🇩🇪 Bundesliga", callback_data="odds_bundesliga")
        ],
        [
            InlineKeyboardButton("🇫🇷 Ligue 1", callback_data="odds_ligue1"),
            InlineKeyboardButton("🏆 Champions League", callback_data="odds_ucl")
        ],
        [
            InlineKeyboardButton("🇳🇱 Eredivisie", callback_data="odds_eredivisie"),
            InlineKeyboardButton("🇵🇹 Primeira Liga", callback_data="odds_portugal")
        ],
        [
            InlineKeyboardButton("🌍 All Leagues Mix", callback_data="odds_all"),
            InlineKeyboardButton("🔄 Refresh", callback_data="odds")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_stats_keyboard():
    """Get the keyboard for the /stats command"""
    keyboard = [
        [
            InlineKeyboardButton("🎯 New Predictions", callback_data="predict"),
            InlineKeyboardButton("📅 Upcoming", callback_data="upcoming")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_community_keyboard():
    """Get the keyboard for the /community command"""
    keyboard = [[
        InlineKeyboardButton(
            "🚀 Open Community Hub",
            web_app=WebAppInfo(url="https://british-pressed-leu-alpine.trycloudflare.com")
        )
    ], [
        InlineKeyboardButton("📊 Dashboard", callback_data="dashboard"),
        InlineKeyboardButton("🏆 Leaderboard", callback_data="leaderboard")
    ], [
        InlineKeyboardButton("🎯 Make Prediction", callback_data="predict"),
        InlineKeyboardButton("💰 View Odds", callback_data="odds")
    ]]
    return InlineKeyboardMarkup(keyboard)

def get_dashboard_keyboard():
    """Get the keyboard for the dashboard"""
    keyboard = [[
        InlineKeyboardButton("🎯 Make Prediction", callback_data="predict"),
        InlineKeyboardButton("🏆 View Leaderboard", callback_data="leaderboard")
    ], [
        InlineKeyboardButton("🔙 Back to Community", callback_data="community")
    ]]
    return InlineKeyboardMarkup(keyboard)

def get_leaderboard_keyboard():
    """Get the keyboard for the leaderboard"""
    keyboard = [[
        InlineKeyboardButton("📊 My Dashboard", callback_data="dashboard"),
        InlineKeyboardButton("🎯 Make Prediction", callback_data="predict")
    ], [
        InlineKeyboardButton("🔙 Back to Community", callback_data="community")
    ]]
    return InlineKeyboardMarkup(keyboard)

def get_feed_keyboard():
    """Get the keyboard for the feed"""
    keyboard = [[
        InlineKeyboardButton("🎯 Make Prediction", callback_data="predict"),
        InlineKeyboardButton("🏆 Leaderboard", callback_data="leaderboard")
    ], [
        InlineKeyboardButton("🔙 Back to Community", callback_data="community")
    ]]
    return InlineKeyboardMarkup(keyboard)

def get_badges_keyboard():
    """Get the keyboard for the badges"""
    keyboard = [[
        InlineKeyboardButton("🎯 Make Prediction", callback_data="predict"),
        InlineKeyboardButton("📊 My Dashboard", callback_data="dashboard")
    ], [
        InlineKeyboardButton("🔙 Back to Community", callback_data="community")
    ]]
    return InlineKeyboardMarkup(keyboard)

def get_deepml_keyboard(home_team, away_team):
    """Get the keyboard for the /deepml command"""
    keyboard = [
        [InlineKeyboardButton("🌳 Framework 1: Reliable Ensemble", callback_data=f"deepml1_{home_team}_{away_team}")],
        [InlineKeyboardButton("⚡ Framework 2: XGBoost Advanced", callback_data=f"deepml2_{home_team}_{away_team}")],
        [InlineKeyboardButton("🚀 Framework 3: LightGBM Pro", callback_data=f"deepml3_{home_team}_{away_team}")],
        [InlineKeyboardButton("🧠 Framework 4: TensorFlow Neural", callback_data=f"deepml4_{home_team}_{away_team}")],
        [InlineKeyboardButton("🔥 Framework 5: PyTorch LSTM", callback_data=f"deepml5_{home_team}_{away_team}")],
        [InlineKeyboardButton("⚡ GOD ENSEMBLE ⚡", callback_data=f"godensemble_{home_team}_{away_team}")],
    ]
    return InlineKeyboardMarkup(keyboard)

def get_specific_league_odds_keyboard():
    """Get the keyboard for specific league odds"""
    keyboard = [[InlineKeyboardButton("← Back to Leagues", callback_data="odds")]]
    return InlineKeyboardMarkup(keyboard)
