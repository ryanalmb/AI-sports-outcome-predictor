# Deploy Sports Prediction Bot

This project is now local-first and LLM-first (Gemini), with no DB and no heavy ML by default.

## Render.com (simple)
- Build Command: `pip install uv && uv sync --frozen`
- Start Command: `python fixed_bot.py`
- Environment Variables:
  - TELEGRAM_BOT_TOKEN (required)
  - FOOTBALL_API_KEY (optional; for upcoming fixtures via football-data.org)
  - GEMINI_API_KEY (optional; to enable LLM-first predictions)
  - GEMINI_MODEL_ID (optional; default: gemini-2.5-pro)
- Port: The bot runs a health endpoint on `$PORT` (defaults to 10000 locally).

## Docker (optional)
A minimal Dockerfile is provided. Example:

```
docker build -t sports-bot .
docker run -e TELEGRAM_BOT_TOKEN=xxxx -e FOOTBALL_API_KEY=xxxx -p 8080:8080 sports-bot
```

Notes:
- Database is not used in the default path. Community features are disabled.
- Heavy ML frameworks (xgboost, lightgbm, tensorflow, pytorch) are archived and not installed.
- `/predict` uses the LLM if `USE_LLM=1` and `GEMINI_API_KEY` is set; otherwise a heuristic fallback is used.
