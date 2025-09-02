# /deepanalyze Architecture and Execution Guide

This document explains how the `/deepanalyze` command works end to end, including architecture, execution flow, components, key algorithms, configuration, logging, and troubleshooting.

## Overview

- Goal: Perform high-quality, multi-source research and analysis for a given match/event, and present a trustworthy, concise report with sources.
- Approach: LLM-assisted, multi-stage pipeline with strict sourcing/citation controls and robustness features to reduce timeouts and cancellations.

## Top-level architecture

- Entry point: `fixed_bot.py` (python-telegram-bot)
  - `deep_analyze_command` orchestrates the end-to-end sequence with user feedback and progress updates.
- Query generation: `query_generator.py`
  - Uses Gemini Flash 2.5 to generate diverse, intent-rich queries; adds site: reputable domain variants; interleaves coverage queries for both teams.
- Data acquisition: `data_acquisition.py`
  - Executes searches (DuckDuckGo via ddgs), adds site-specific queries, dedupes and caps, applies Flash reranking with domain/recency boost; enforces whitelist/domain caps; fetches HTML; extracts text; filters and selects final articles.
- Content processing: `content_processor.py`
  - Uses Gemini Flash 2.5 to check relevance and extract summaries; accepts both JSON and plain text; keeps relevant items even if JSON is not returned.
- Final synthesis: `final_analyzer.py`
  - Uses Gemini Pro 2.5 to synthesize a final analysis based on processed data; simpler, Telegram-safe formatting with explicit Sources.

## End-to-end flow for `/deepanalyze`

1) User invokes `/deepanalyze` with an event query (e.g., "Netherlands vs Poland").
2) Command validates input and posts a progress message.
3) Step 1 – Query generation (Flash; no fallback)
   - Flash prompt instructs generation of 10–15 targeted queries covering:
     - Head-to-head and historical trends
     - Team-specific news for both teams (form, selection, tactics)
     - League/competition context (standings, fixture congestion, disciplinary rules, transfer window impact)
     - Injuries/suspensions (for each team)
     - Press conferences/manager quotes
     - Tactical previews/expected setup
     - Probable lineups
     - Official communications (club/league)
     - Bookmaker/odds availability and price movements
     - 3–5 site: queries for reputable sources (ESPN, Sky Sports, BBC, Guardian, WhoScored, Transfermarkt, The Athletic, official club/league).
   - Coverage queries are generated to ensure balanced intent and both-team pairing (e.g., "Chelsea injury update" AND "Liverpool injury update").
   - The mixer interleaves Flash queries with coverage queries and prioritizes site: queries, dedupes, and caps to `MAX_QUERIES` (recommended 10).

4) Step 2 – Data acquisition (search + rerank + fetch + select)
   - Search planning:
     - For each query, add site-specific variants for top reputable domains (`TOP_DOMAINS_PER_QUERY`).
     - Deduplicate and cap the final set of query strings (`MAX_SEARCH_QUERIES_TOTAL`).
     - Execute searches concurrently with `SEARCH_MAX_CONCURRENT`. Site: queries typically request fewer results (`PER_DOMAIN_QUERIES`).
     - Retry policy:
       - No per-call retries (`do_retries=False`) to prevent spamming for isolated failures.
       - Perform a single global retry only if more than half of all searches fail.
     - Logging:
       - `Search planning | base=X site_augmented=Y max_total=… top_domains_per_query=…`
       - `Search completed | queries_executed=… succ=… fail=… serp_items=…`
   - Flash reranking (no fallback):
     - Flash returns an order; we compute a final score per item:
       `final_score = rerank_score + DOMAIN_WEIGHT (for whitelisted domains) + RECENCY_WEIGHT (for recency hints in title/snippet)`
   - Strict selection:
     - Enforce at least `MIN_WHITELIST_COUNT` items from `REPUTABLE_DOMAINS`.
     - Enforce per-domain cap `MAX_PER_DOMAIN`.
     - Fill remaining slots strictly by score up to `MAX_FETCHES`.
     - Logging:
       - `URL selection | whitelisted=… others=… selected=… min_whitelist=… max_fetches=…`
   - Fetch + parse + filter:
     - Polite jitter before HTTP requests, adaptive timeouts with exponential backoff, explicit handling for 429, 403, 404.
     - Extract text; accept only if extracted body ≥ 100 chars.
     - Filter obvious junk.
     - Ensure up to `TARGET_ARTICLES` (e.g., 10) are returned. If filtered is fewer than `TARGET_ARTICLES`, pad from remaining valid-but-filtered items.
     - Logging:
       - `Articles selected | valid=… filtered=… selected=… target=…`
     - End of Step 2:
       - `Acquired X articles for 'Event'` means X items are passed to content processing. It does NOT mean all are semantically relevant—only that they were fetchable and minimally parsed.

5) Step 3 – Content processing (Flash) with relaxed strictness
   - Each article is processed with a relevance gate and summary extraction.
   - Prompt requires:
     - First line: `RELEVANCE: YES/NO`
     - If YES: include a 2–3 sentence summary (plain text allowed; JSON optional).
   - Parser accepts:
     - JSON dict responses: if relevance == `NO`, drop; else attach url, clean/validate fields and keep.
     - Plain text responses: if first line `RELEVANCE: NO`, drop; otherwise keep a minimal structured result with summary.
   - This loosens strictness so more relevant items are kept even without perfect JSON.

6) Step 4 – Final synthesis (Pro)
   - Compile a briefing document from processed items (summaries, players, injuries/suspensions, statistics, sources).
   - Pro prompt (Telegram-safe) asks for:
     - Summary, key factors, prediction, confidence score, brief reasoning, Sources.
   - Response is formatted safely for Telegram:
     - Safer Markdown; explicit Sources list; fallback to plain text when necessary.

## Key modules and responsibilities

- `fixed_bot.py`
  - `deep_analyze_command` orchestrates the pipeline; edits progress messages; logs runtime flags and file origins for verification.
  - Global error handler prevents noisy "No error handlers registered" logs.
  - Safe message editing avoids Telegram BadRequest (parse) errors.
- `query_generator.py`
  - `generate_search_queries` returns a balanced set of queries using:
    - Flash primary queries (site: prioritized)
    - Coverage queries with pairwise team balance (injury, pressers, tactics, league context, odds, official)
  - `MAX_QUERIES` caps the final queries (recommend 10).
- `data_acquisition.py`
  - Builds search plan (base + site: reputable variants), dedupes and caps.
  - Executes searches concurrently; **no per-call retries**; single global retry only if majority fail.
  - Flash reranker (no fallback) attaches `rerank_score`; final score adds domain/recency boosts.
  - Selection enforces whitelist presence and per-domain diversity; fetches and parses with robust policies; ensures up to `TARGET_ARTICLES` are returned.
- `content_processor.py`
  - Calls Flash to check relevance and extract summary.
  - Accepts both JSON and plain text; only explicit `RELEVANCE: NO` is dropped.
  - Cleans and standardizes minimal structure.
- `final_analyzer.py`
  - Pro synthesizes the final report with Sources; uses safer Markdown formatting.

## Quality and freshness strategies

- **Reputable domains (multi-sport)**: configurable via `REPUTABLE_DOMAINS`.
  - Default list includes ESPN, SkySports, BBC, Guardian, Reuters/AP, WhoScored, Transfermarkt, official league/club sites, NBA/NFL/MLB/NHL, ATP/WTA/Tennis.com, PGA/Golf outlets, NCAA, The Athletic, and reference sites.
- **Scoring**:
  - `DOMAIN_WEIGHT` (default 15) boosts whitelisted domains.
  - `RECENCY_WEIGHT` (default 10) boosts content with recency hints ("today", "this week", "hours ago", recent year/month tokens).
  - Future enhancement: parse HTML meta published_time to enforce hard recency.
- **Strict selection**:
  - `MIN_WHITELIST_COUNT` ensures top outlets appear in the final set.
  - `MAX_PER_DOMAIN` ensures diversity.
- **Balanced queries**:
  - Enforced pairing: any team-specific query for Team A has a Team B counterpart (injury, pressers, tactics, lineup, etc.).

## Robustness and resilience

- Reduced timeouts and cancellations:
  - Politeness jitter, adaptive timeouts, explicit handling for 429/403/404.
  - Protects rate limits and avoids cascading failures.
- Controlled retries:
  - No per-call retry for searches.
  - Single global retry for failed searches only if **most** of them fail.
- Safe Telegram messaging:
  - Global error handler.
  - Safe edit with Markdown sanitize + plain text fallback.

## Configuration (env variables)

- Query generation:
  - `MAX_QUERIES` (e.g., 10)
- SERP planning/execution:
  - `TOP_DOMAINS_PER_QUERY` (default 6), `PER_DOMAIN_QUERIES` (default 1)
  - `MAX_SEARCH_QUERIES_TOTAL` (default 40)
  - `SEARCH_MAX_CONCURRENT` (default 6)
  - `REPUTABLE_DOMAINS` (multi-sport list)
  - `MIN_WHITELIST_COUNT` (default 5)
  - `MAX_PER_DOMAIN` (default 2)
  - `DOMAIN_WEIGHT` (default 15), `RECENCY_WEIGHT` (default 10)
- Fetching:
  - `MAX_FETCHES` (default 12)
  - `DATA_ACQUISITION_MAX_CONCURRENT_REQUESTS` (default 5)
  - `DATA_ACQUISITION_REQUEST_TIMEOUT` (default 30.0–45.0)
  - `DATA_ACQUISITION_ADAPTIVE_TIMEOUT_MULTIPLIER` (default 1.5)
  - `DATA_ACQUISITION_MAX_RETRIES` (default 3; used for HTTP fetches, not individual ddg searches)
  - `DATA_ACQUISITION_REQUEST_DELAY_MIN/MAX` (default 0.5/2.0)
- Content processing:
  - `TARGET_ARTICLES` (default 10)
  - `CONTENT_MAX_CONCURRENT_API_CALLS` (default 5)
  - `CONTENT_INITIAL_RETRY_DELAY` (default 1.0)
  - `CONTENT_MAX_RETRIES` (default 3)
- Synthesis:
  - `GEMINI_API_KEY`, `GEMINI_FLASH_MODEL_ID` (default `gemini-2.5-flash`), `GEMINI_MODEL_ID` (default `gemini-2.5-pro`)
- Bot:
  - `TELEGRAM_READ_TIMEOUT`, `TELEGRAM_CONNECT_TIMEOUT`
- Feature flags:
  - Flash query generation and reranking are required for `/deepanalyze`; there are no silent fallbacks.

## Logging and runtime verification

- `fixed_bot.py`:
  - Prints DeepAnalyze flags (Flash enabled, model IDs) and paths to `generate_search_queries` and `acquire_data` to ensure correct module usage.
- `data_acquisition.py`:
  - Logs search plan size, success/failure counts, reranker usage, selection metrics, and final article selection.
- Optional (recommended during tuning):
  - Add debug logs to print top selected queries and top hostnames with scores to verify quality quickly.

## Troubleshooting FAQ

- "Acquired N articles" but only a few processed:
  - That log marks Step 2 completion. Step 3 still filters for relevance. With relaxed strictness, more items should be kept as minimal structured summaries unless explicitly `RELEVANCE: NO`.
- Not enough ESPN/Sky/NFL content:
  - Increase `PER_DOMAIN_QUERIES`, `TOP_DOMAINS_PER_QUERY`, raise `MIN_WHITELIST_COUNT` and `DOMAIN_WEIGHT`; consider adding DDG region/time bias and citation-only handling for gated sources (403).
- Looping or repeated retries:
  - Per-call retries are disabled; only one global retry happens if the majority of searches fail.
- Telegram parse errors:
  - Final message uses safe Markdown fallback; apply the same to intermediate progress messages if needed.

## Security and compliance

- Respect robots.txt and legal constraints for scraping.
- Limit concurrency and retries to avoid overloading publishers.
- Prefer official and reputable outlets; avoid content farms.

## Extensibility ideas

- Per-sport domain configs and score weights.
- HTML meta `published_time` parsing for hard recency enforcement.
- DDG `region`/`time` filters.
- Citation-only inclusion for whitelisted but gated pages.
- Chunk-based extraction for very long articles.
