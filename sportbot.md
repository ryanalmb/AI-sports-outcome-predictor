# Sports Prediction Bot (LLM‑first) – Architecture and Operations Guide

This document gives engineers and agents a complete understanding of the bot’s architecture, design decisions, runtime behavior, and extensibility. It reflects the current codebase after recent changes that make Gemini the sole prediction source for `/predict`.

Contents
- High‑level overview
- Core principles and goals
- Repository layout and major modules
- Runtime architecture
- Command surface and UX
- LLM predictor design (Gemini) and error handling
- Football data collection (football‑data.org)
- Heuristic engines (for analysis and advanced commands)
- Configuration and environment variables
- Logging and observability
- Deployment and operations
- Health checks
- Security considerations
- Extensibility guidelines
- Known limitations and roadmap
- Troubleshooting checklist

---

## High‑level overview

- Purpose: A lightweight, production‑friendly Telegram bot for football match predictions and insights.
- Design: LLM‑first for predictions (Gemini). No database. Minimal dependencies.
- Data: Live fixtures from football‑data.org. Optional engines provide heuristic analysis for advanced/analysis commands.
- Result: Easy to run locally and deploy. Robust JSON handling and clear logging.

## Core principles and goals

- LLM‑first: Gemini response is authoritative for `/predict`. No heuristics or market blending in that path.
- Minimal deps: Python 3.11+, `python-telegram-bot`, `aiohttp`, `google-generativeai`, `numpy`, `pandas`, `scikit-learn` (for optional engines).
- No DB: Community stats/leaderboards are out of scope by default.
- Resilience: Strong JSON parsing, tolerant numeric coercion, clear errors for users on failure.
- Observability: Single init, structured logging, optional debug toggles to view raw LLM snippets.

## Repository layout and major modules

Top‑level files:
- `fixed_bot.py` – Main Telegram bot entry point; handlers, health server, and integration glue.
- `llm_predictor.py` – Gemini LLM wrapper. JSON‑only outputs, robust parsing, and strict error behavior.
- `simple_football_api.py` – Live upcoming fixtures from football‑data.org (no DB).
- `enhanced_predictions.py` – “Enhanced analysis” engine (live updates, historical tracking in memory, head‑to‑head/form simulations).
- `advanced_prediction_engine.py` – “Advanced/Heuristic ensemble” engine (explainable factors). Not used by `/predict`.
- `league_mixer.py` – Mixes matches across leagues for a diverse list.
- `verify_authentic_data.py` – Standalone script to fetch authentic El Clásico results (example API usage).
- `pyproject.toml` – Minimal Python deps.
- `Dockerfile`, `render.yaml`, `apprunner.yaml` – Optional deployment artifacts.
- `.env`, `.env.example` – Environment configuration.
- `WAY_FORWARD.md`, `README.md` – Design and quick‑start docs.

Archived and ML assets:
- `archive/` – Legacy code (DB, heavy ML frameworks, etc.). Not used in the current runtime.
- `ml/` – Utility modules for optional heuristic engines.

## Runtime architecture

- Telegram layer: `python-telegram-bot` v20.
  - Application is built via `Application.builder().post_init(_post_init).build()`.
  - `post_init` performs one‑time initialization of the LLM predictor.
- LLM predictor: `llm_predictor.GeminiPredictor`
  - Configures `google-generativeai` with `GEMINI_API_KEY`.
  - Uses a concise system instruction (if supported) and a minimal prompt per request.
  - Returns parsed probabilities and the raw payload for observability.
- Data layer: `simple_football_api.SimpleFootballAPI`
  - Fetches upcoming fixtures for major leagues via football‑data.org.
- Engines:
  - `EnhancedPredictionEngine` and `AdvancedPredictionEngine` are used for `/analysis`, `/advanced`, `/live`, `/accuracy` routes. They are heuristic/simulated; no DB.
- Health endpoint:
  - A lightweight HTTP server thread (`start_health_server_thread`) responds on `PORT` (default 10000).

### Request lifecycle: `/predict`
1) User sends `/predict TeamA vs TeamB`.
2) `fixed_bot.SimpleSportsBot._generate_prediction_for_match_async` calls `GeminiPredictor.predict(home, away)`.
3) `GeminiPredictor` makes exactly one concise call to Gemini, extracts JSON, coerces numeric fields, normalizes probabilities, and returns:
   - `{ home_win, draw, away_win, framework: 'gemini-llm', source: 'gemini-2.5', raw: <raw_text> }`
4) The bot chooses the predicted outcome by max(home_win, draw, away_win) and replies to the chat.
5) If any failure occurs, the user receives a clear error message; no heuristic fallback is used.

## Command surface and UX

- `/start` – Welcome + inline buttons.
- `/help` – Command descriptions.
- `/leagues` – Supported leagues.
- `/upcoming` – Live upcoming fixtures (football‑data.org). Note: UI lines currently say “TheSportsDB” in two places; should be updated to “football‑data.org”.
- `/predict` – Gemini‑only prediction path. Requires input format: `TeamA vs TeamB`.
- `/analysis` – Enhanced analysis (form, head‑to‑head, simulated injuries).
- `/advanced` – Heuristic “professional” ensemble explanation.
- `/live` – Live match updates (football‑data.org).
- `/accuracy` – Accuracy stats computed from in‑memory history in `EnhancedPredictionEngine`.
- `/odds` – Menu with safe, heuristic‑style odds display (not used by `/predict`).
- `/stats` – Static example stats (can be wired to local tracking later).

## LLM predictor design (Gemini)

File: `llm_predictor.py`

- Initialization
  - Reads `GEMINI_API_KEY`, `GEMINI_MODEL_ID` (default `gemini-2.5-pro`).
  - Creates `GenerativeModel`. When supported, passes a `system_instruction` to enforce concise, JSON‑only replies.

- Prompting
  - Single concise prompt, e.g.:
    - "Matchup: <Home> vs <Away>. Return only valid JSON with numeric fields: home_win, draw, away_win (0–100). Be concise."
  - No token ceilings. No retries. If the SDK fails, we surface an error to the caller.

- Extraction and parsing
  - Scans `resp.text` and all candidates/parts.
  - Supports `inline_data` payloads (base64 decoded) if present.
  - Parses JSON; if that fails, sanitizes (trim braces, remove trailing commas) and tries again.
  - If sanitized JSON still fails, attempts regex extraction of numeric fields. If that also fails, returns `{error}`.
  - Numeric coercion accepts strings with `%`, clamps [0..100], and normalizes to sum ~100.

- Return shape
  - Success: `{ home_win, draw, away_win, framework: 'gemini-llm', source: 'gemini-2.5', raw: <raw string> }`
  - Failure: `{ error: '<message>' }`

- Debug/observability
  - `DEBUG_LLM=1`: logs a sanitized “raw payload (snippet)” when data is extracted and includes candidate diagnostics when nothing extractable is found.

## Football data collection (football‑data.org)

File: `simple_football_api.py`

- Collects upcoming fixtures for 5 major European leagues using `FOOTBALL_API_KEY`.
- Per‑request session with short timeouts; handles 429 rate‑limits with backoff.
- Returns sorted fixtures, each item includes: `home_team`, `away_team`, `league`, `match_time`, `date`, `source`.

Alternate (not wired by default): `football_data_collector.py` collects a broader set of leagues and has a similar structure.

## Heuristic engines (analysis/advanced)

- `enhanced_predictions.py` (EnhancedPredictionEngine)
  - Live updates for `/live`, in‑memory history for `/accuracy`, team form, head‑to‑head, and injury simulations for `/analysis`.
- `advanced_prediction_engine.py` (AdvancedPredictionEngine)
  - Ensemble of explainable heuristic factors (team strength, form, head‑to‑head, tactics, venue, players). Used by `/advanced`.

Note: These engines are not used by `/predict` anymore. `/predict` is strictly Gemini‑only.

## Configuration and environment variables

- Required
  - `TELEGRAM_BOT_TOKEN`: Telegram bot token.

- LLM
  - `GEMINI_API_KEY`: Google Generative AI key.
  - `GEMINI_MODEL_ID` (optional, default `gemini-2.5-pro`).
  - `USE_LLM=1`: enable LLM for `/predict` (current code assumes LLM path in `/predict`; keep this enabled).
  - `DEBUG_LLM=1`: opt‑in to richer logs (raw snippets and diagnostics).

- Football‑data
  - `FOOTBALL_API_KEY`: for football‑data.org.

- Health server
  - `PORT` (default 10000): HTTP health endpoint.

Edit `.env` and restart the process to apply changes.

## Logging and observability

- Logging config is forced in `fixed_bot.py` (so other libs can’t override it).
- Key events:
  - Startup: “LLM initialized in post_init”.
  - LLM success (DEBUG_LLM=1): “Gemini raw payload (snippet): …”.
  - LLM parse failure: concise warning; if DEBUG_LLM=1, includes candidate/safety diagnostics.
  - LLM call error: full stack trace via `logger.exception`.

## Deployment and operations

- Local (Linux/macOS):
  - `pip install -U uv && uv sync` or use `setup_run.ps1` on Windows.
  - Run: `python fixed_bot.py`.

- Windows: `setup_run.ps1` creates a virtualenv, installs deps, loads `.env`, and starts the bot.

- Docker:
  - `docker build -t sports-bot .`
  - `docker run -p 8080:8080 -e TELEGRAM_BOT_TOKEN=... -e GEMINI_API_KEY=... -e GEMINI_MODEL_ID=gemini-2.5-pro sports-bot`

- Render:
  - `render.yaml` uses `python fixed_bot.py`. Add secrets in the dashboard.

## Health checks

- A lightweight HTTP server runs in a background thread.
- GET `/` or `/health` → “Sports Prediction Bot is running!”.

## Security considerations

- Do not commit real tokens to `.env` or `.env.example`.
- Treat `GEMINI_API_KEY` and `TELEGRAM_BOT_TOKEN` as secrets.
- Consider secret management on deployment hosts (Render/CI/CD/env vaults).

## Extensibility guidelines

- Adding a bot command:
  1) Create an async handler in `fixed_bot.py` (e.g., `async def my_command(...)`).
  2) Register with `self.application.add_handler(CommandHandler("my", self.my_command))` in `setup_handlers`.

- Creating an LLM‑powered command:
  1) Reuse `GeminiPredictor` or create a new predictor class with a strict JSON schema.
  2) Add handler that calls predictor, formats the reply and sends to the user.
  3) Use `DEBUG_LLM` for safe diagnostics.

- Expanding fixture sources:
  - Extend `SimpleFootballAPI` leagues, or centralize in a single collector.

- Reintroducing a DB (optional):
  - Add a small async store for personal stats, behind a feature flag.

## Known limitations and roadmap

- `/upcoming` UI string mentions “TheSportsDB API” in two places; should be changed to “football‑data.org ✅”.
- `/stats` is static text; wiring it to the engine’s in‑memory history or a local JSON file would improve realism.
- No caching for LLM predictions: optionally add a 5‑minute in‑memory cache per match to reduce calls and latency.
- No retry on transient Gemini 500s: optional short retry with jitter could be reintroduced if desired (currently avoided by design to eliminate waste).

## Troubleshooting checklist

- Bot starts but LLM is not used:
  - Ensure `.env` has `USE_LLM=1` and valid `GEMINI_API_KEY`.
  - Look for “LLM initialized in post_init”.

- LLM call errors (500s, timeouts):
  - Check logs for `logger.exception` stack traces.
  - With `DEBUG_LLM=1`, inspect diagnostics and raw snippet presence.

- User gets “⚠️ Error generating LLM prediction…”:
  - Indicates LLM returned error or no extractable data; try again, or inspect logs for details.

- `/upcoming` shows 0 matches:
  - Verify `FOOTBALL_API_KEY` and network. football‑data.org rate limits may apply.

---

For questions or contributions, start by reading this file, then open `fixed_bot.py` and `llm_predictor.py` to see the exact runtime paths and conventions used across the bot.
