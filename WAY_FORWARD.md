# Way Forward: Local, Lightweight, LLM-first Sports Predictor

This document outlines the architecture changes and implementation plan to:
- Remove the database requirement (no shared leaderboards/community backend) – DONE
- Run locally (no Docker required; Docker optional) – DONE
- Keep only lightweight path and introduce Gemini 2.5 Pro as the primary predictor – DONE (LLM-first)
- Optionally integrate a browser agent (browser-use) later

## 1) Target architecture (local-first)

Runtime:
- Python 3.11+
- Telegram Bot: python-telegram-bot
- Data: SimpleFootballAPI (football-data.org)
- Predictors:
  - Primary: Gemini 2.5 Pro (google-generativeai)
  - Fallback: ReliableAuthenticEnsemble (scikit-learn based, no deep frameworks)
- Storage: None by default (no Postgres). Optional local JSON file for personal stats.

Notes:
- Docker and Postgres are removed from the default path. End-users can bring their own DB later if needed.
- Community features (leaderboard, feed, badges) are disabled in local mode.

## 2) Dependency posture

Keep:
- google-generativeai (Gemini)
- python-telegram-bot
- aiohttp, requests, numpy, pandas, scikit-learn

Remove heavy:
- tensorflow/keras, torch, xgboost, lightgbm
- asyncpg (no DB)

Optional (later):
- browser-use, playwright (for a Gemini-controlled browser agent)

The project has already had heavy deps removed from `pyproject.toml` (TF/Torch/XGBoost/LightGBM/asyncpg).

## 3) Configuration

Environment (.env):
- TELEGRAM_BOT_TOKEN=...
- FOOTBALL_API_KEY=... (football-data.org)
- ODDS_API_KEY=... (optional, for live odds)
- GEMINI_API_KEY=...
- GEMINI_MODEL_ID=gemini-2.5-pro (default)

No DB variables are required anymore.

## 4) Code changes (high-level)

A. Remove DB usage and community features
- fixed_bot.py
  - Do not import DatabaseManager; set `self.db_manager = None`
  - Remove or disable commands and callbacks that require DB:
    - /community, leaderboard, dashboard, feed, badges
  - Start/help text: remove references to community and deep learning frameworks
  - Keep only: /start, /help, /leagues, /upcoming, /predict, /odds (optional), /analysis, /advanced, /accuracy, /stats
- database_manager.py can be archived or left unused.

B. Make predictors lightweight and LLM-first
- Add `llm_predictor.py` (Gemini 2.5 Pro). Use structured prompts and return calibrated probabilities with rationale.
- Wire a new `/llm_predict` command (or make `/predict` use Gemini by default via env flag `USE_LLM=1`).
- Keep LLM-first with heuristic fallback only; heavy frameworks archived and not imported.
- Archived references to heavy frameworks:
  - ml/xgboost_framework.py, ml/lightgbm_framework.py, ml/tensorflow_framework.py, ml/pytorch_lstm_framework.py (moved to archive/)
- Remove references to non-existent modules:
  - ml.direct_odds_api, ml.live_odds_collector, ml.comprehensive_authentic_predictor
  - Either implement lightweight versions or remove those call sites. Short-term: use the existing `_fallback_realistic_odds` and SimpleFootballAPI data.

C. Personal stats (optional)
- If you want local-only personal tracking, add a tiny JSON-based store, e.g. `local_stats.py`:
  - Keyed by Telegram user id: total_predictions, correct_predictions, streaks.
  - Update on each `/predict` invocation.
  - This keeps “personal stats” without any server.

## 5) Gemini 2.5 Pro integration

Create `llm_predictor.py`:
- Initialize Gemini with `GEMINI_API_KEY` and `GEMINI_MODEL_ID` (default gemini-2.5-pro)
- `predict(home_team, away_team, date=None)` returns: { home_win, draw, away_win, rationale, framework, source }
- Use temperature ~0.3–0.5, max tokens ~512
- Normalize output probabilities to sum to ~100
- Enrich prompt with context from SimpleFootballAPI if available (recent form, league)

Wire into bot:
- On startup: `self.llm = GeminiPredictor()`; `await self.llm.initialize()`
- Add `/llm_predict <Home> <Away>` handler
- Or flip default `/predict` to use Gemini first, fallback to ensemble or the existing light heuristic

## 6) Optional: Browser agent (browser-use)

When ready to automate browsing tasks (e.g., fetch odds/news):
- Install: `pip install browser-use playwright` then `playwright install chromium`
- Create `browser_agent.py` that instantiates `browser_use.Agent` with `browser_use.llm.Gemini`
- Run tasks like: “Open oddsportal.com and find odds for ‘Arsenal vs Chelsea’”
- Expose a bot command `/web_task <instruction>`
- Add safety: max_actions, timeout, domain safelist

This stays optional and off by default for local setups.

## 7) Local run (no Docker)

- Create `.env` with your keys
- Install deps: `pip install -U uv` then `uv sync` (or `pip install -r` equivalent)
- Run the bot: `python fixed_bot.py`
- Use Telegram to interact with the bot

## 8) Cleanup and follow-ups

- Remove Docker-focused docs from your onboarding path (keep files if you want for later, but mention they’re not used locally)
- Remove or gate all code that references missing datasets (football_data/data/Matches.csv). If you don’t ship that file, ensure the code degrades gracefully (use API-only features and/or LLM output)
- Update README with local-only instructions

## 9) Task checklist

- [x] Strip DB references in fixed_bot.py; disable community-related handlers and UI
- [x] Update start/help text to remove deep-learning and community references
- [x] Add `llm_predictor.py` and make `/predict` use LLM when `USE_LLM=1`
- [x] Remove heavy frameworks and dataset requirements from the default path (archived code)
- [x] Remove references to missing modules (ml.direct_odds_api, ml.live_odds_collector, ml.comprehensive_authentic_predictor)
- [x] Add GEMINI_API_KEY to `.env.example`
- [ ] Optional: add `local_stats.py` for personal tracking
- [ ] Optional: add `browser_agent.py` + `/web_task` command
- [x] Update README for local run

## 10) Risks and mitigations

- Missing dataset (Matches.csv):
  - Mitigation: LLM-first predictor, and graceful heuristics if no data
- Live odds APIs can rate-limit:
  - Mitigation: use cached or heuristic odds fallback (already present)
- Gemini JSON parsing robustness:
  - Mitigation: normalize and clamp probabilities; guard JSON parsing with try/except
- Browser agent stability:
  - Mitigation: headless Chromium, max_actions, domain safelist

---

This plan makes the project easy to run locally with minimal dependencies, uses Gemini 2.5 Pro for high-quality predictions, and keeps a small ML fallback without requiring a database or Docker.
