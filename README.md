# AI Sports Outcome Predictor (LLM-first, Local-First)

This is a lightweight, production-friendly Telegram bot for football match predictions.
It runs locally without Docker or a database, uses a simple API for fixtures, and (optionally) Google Gemini for LLM-based predictions. Heavy ML frameworks are removed; a scikit-learn fallback ensemble can be enabled later.

## Quick start

1) Create a .env file

Copy .env.example to .env and fill in values:
- TELEGRAM_BOT_TOKEN (required)
- FOOTBALL_API_KEY (optional, for upcoming fixtures via football-data.org)
- GEMINI_API_KEY + GEMINI_MODEL_ID (optional; if set, you can enable LLM predictions)

2) Install deps

- pip install -U uv
- uv sync

3) Run

- python fixed_bot.py

The bot starts and exposes a health endpoint on PORT (default 10000). Commands available:
- /start, /help, /leagues, /upcoming
- /predict (uses LLM if USE_LLM=1 and Gemini key present; otherwise heuristic)
- /analysis, /advanced, /accuracy
- /odds (fallback odds unless ODDS_API_KEY and DirectOddsAPI are implemented)

## What was cleaned/reworked

- Removed database and heavy ML by default. All related files moved to archive/.
- fixed_bot.py is the single entry point. apprunner.yaml and render.yaml updated.
- pyproject.toml trimmed to minimal deps.
- Added llm_predictor.py (Gemini wrapper) and updated .env.example.
- Kept EnhancedPredictionEngine and AdvancedPredictionEngine (heuristics). They do not require the dataset.
- Guard or remove references to missing modules. Odds paths fall back when APIs aren’t configured.

## Notes

- The dataset football_data/data/Matches.csv is not shipped. Any code paths that used it are now optional or guarded. The default run does not require it.
- Community/leaderboard features are disabled in local mode; no DB.
- If you enable Gemini, set USE_LLM=1 in .env to prefer LLM for /predict.

## Deploy

- Render: render.yaml uses python fixed_bot.py. DB is commented out.
- Replit/AppRunner: apprunner.yaml updated to fixed_bot.py.

## Next steps (optional)

- Implement a small DirectOddsAPI client if you want real odds.
- Reintroduce DB later behind a feature flag using asyncpg or a hosted Neon/Render DB.
- If you want heavy ML again, create extras for xgboost/lightgbm/torch/tf and gate imports.
