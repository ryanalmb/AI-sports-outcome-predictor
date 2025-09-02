# Flash Live Implementation Plan for /degenanalyze (Solana Degen Utility)

This document outlines a phased, production-grade plan to add a new command, `/degenanalyze`, powered by Gemini Flash “Live” capabilities. It borrows proven patterns from `/deepanalyze` while keeping `/deepanalyze` untouched. The new command targets Solana degen use cases, provides live research-driven predictions with degen-themed framing, and encourages users to reinvest a portion of wins back into the project—intelligently and responsibly.

---

## Goals & Non-goals

- Goals
  - Create a live, tool-using research and analysis command tailored to Solana degens.
  - Orchestrate live web/news/on-chain queries, extract signals, and stream meaningful updates in near real time.
  - Produce degen-themed summaries (“Degen Thesis”, “Catalysts”, “Risks/Rug Checks”, “Entry/Exit Ideas”) with explicit Sources.
  - Encourage reinvesting a portion of winnings back into the project treasury/LP (responsible framing, opt-in).
  - Respect rate-limits, robots.txt, and legal constraints; avoid overloading domains/APIs.

- Non-goals
  - Do not modify `/deepanalyze` behavior.
  - Do not add heavy database dependencies or complex ML frameworks.

---

## High-level UX for `/degenanalyze`

- Invocation examples
  - `/degenanalyze` — guided prompt for token/ticker/topic.
  - `/degenanalyze $TICKER`
  - `/degenanalyze <token name or topic>` (e.g., “bonk catalyst”, “new launch meme meta”).

- Output (streamed and final)
  - Streamed progress (e.g., “Found new CT thread”, “DexScreener: liquidity +X% last 24h”).
  - Final degen-themed report:
    - Degen Thesis (1–2 paragraphs)
    - Catalysts (3–5 bullets)
    - On-chain Signals (holders, liquidity, whales, volume)
    - CT Sentiment (high-level)
    - Risks / Rug Checks (with sources)
    - Entry/Exit Ideas (not financial advice)
    - Reinvest Hook: tasteful nudge to reinvest a portion of wins (opt-in CTA), with a link to project contribution/LP.
    - Sources list (URLs/domains)

- Safety
  - Add a clear “not financial advice” disclaimer.
  - Respect content policies and publisher ToS.

---

## Architecture Overview

- New Live path (separate from `/deepanalyze`):
  - `fixed_bot.py`: add `/degenanalyze` command handler (no change to `/deepanalyze`).
  - `live/` module (new):
    - `live_session.py`: LiveSession orchestrator for Flash Live (state, streaming, guardrails).
    - `tools/`: tool adapters for web_search, fetch_url, html_to_text, parse_meta, and crypto-specific APIs (DexScreener, Solscan/Helius, RugCheck, CoinGecko). Each tool enforces rate limits and polite usage.
  - `prompts/`: curated prompts and policies for Live.
  - `renderers/`: Telegram-safe message builders for streamed updates and final report.

- Coexists with current pipeline:
  - `/deepanalyze` uses existing Flash (non-live) + Pro path.
  - `/degenanalyze` uses Flash Live end-to-end and Pro only if/when needed for final write-up.

---

## Tooling Design (Flash Live)

- Generic tools (available to Live):
  - `web_search(query: str, region?: str, time?: str) -> List[{title, url, snippet}]`
  - `fetch_url(url: str) -> {status, headers, html}`
  - `html_to_text(html: str) -> {text, meta}`
  - `parse_published_time(html: str | text) -> ISO8601 | null`

- Crypto-specific tools:
  - `dexscreener_pair(query: str or address: str) -> price, volume, liquidity, 24h delta`
  - `onchain_holders(address: str) -> {holders_count, top_10_share, whales, new_holders_24h}` (via Helius/Solscan)
  - `rugcheck(address: str) -> {risk_level, flags, notes, url}` (RugCheck API)
  - `coingecko_trending() -> tokens list` (optional)
  - `social_search(query: str) -> recent posts (from sources like Nitter/APIs)`

- Guardrails & scoring:
  - Domain whitelist/boost for reputable crypto outlets (CoinDesk, The Block), reference sites (Messari blog), official sources (solana.com, project’s official site), and mainstream reputable outlets (BBC/Reuters/AP when relevant).
  - Per-domain caps, rate limits, polite backoffs.
  - Recency bias: favor items with `published_time` within window.

---

## Phased Implementation Plan

### Phase 0 – Foundations & Flags (0.5–1 day)
- Add env flags (no code paths changed yet):
  - `USE_FLASH_LIVE=0` (off by default)
  - `DEGEN_MAX_STREAM_DURATION=60s` (streaming budget)
  - `DDG_REGION=us-en`, `DDG_TIME=w` (default region/time window for searches)
  - API keys: DexScreener (none needed), Helius/Solscan, RugCheck, CoinGecko
- Add minimal module stubs in `live/` for future tools and session.

### Phase 1 – Command & Session Skeleton (1 day)
- Add `/degenanalyze` handler in `fixed_bot.py`:
  - Validate input; open a LiveSession; send initial message.
  - Stream small updates every few seconds (coalesced) with safe-edit fallbacks.
- Add `live/live_session.py`:
  - Initialize Flash Live client.
  - Manage tool registry and dispatch.
  - Keep session state: found_sources, attempted_domains, last_update_ts, errors.
  - Basic policies for domain caps and timeouts.

### Phase 2 – Generic Tools & Streaming (1–2 days)
- Implement generic tools: `web_search`, `fetch_url`, `html_to_text`, `parse_published_time`.
- Add domain/recency scoring policy for Live to prioritize sources.
- Streaming messages:
  - “Found potential source: …”
  - “Rejected stale/low-signal: …”
  - “Extracted quote/stat: …”

### Phase 3 – Crypto Tooling (2–3 days)
- Implement crypto-specific tools:
  - `dexscreener_pair()`: price/volume/liquidity and time series deltas.
  - `onchain_holders()`: holders distribution (Helius/Solscan), growth.
  - `rugcheck()`: risk flags and audit notes.
  - Optional: `coingecko_trending()`.
- Live orchestration policy for crypto:
  - Query planning: combine token ticker/name + memecoin-specific heuristics.
  - Enforce minimum reputable crypto outlets in the final set (The Block, CoinDesk, Messari, etc.).
  - Highlight whales/liquidity/volume anomalies.

### Phase 4 – Degen-themed Synthesis & Hooks (1 day)
- Create degen-styled renderers:
  - “Degen Thesis” (concise, energetic, meme-aware).
  - “Catalysts” & “Risks / Rug Checks”.
  - “On-chain Signals” & “CT Sentiment”.
  - “Entry/Exit Ideas” (always with NFA disclaimer).
  - “Reinvest Hook”: tasteful CTA to reinvest a portion of wins back into the project with a link.
- Optionally call Pro to polish the final summary while keeping the degen tone.

### Phase 5 – Rate Limits, Caching & Observability (1 day)
- Add per-domain/tool rate limits and exponential backoff.
- Cache recent query results (short TTL) to avoid re-hitting sources in the same session.
- Metrics & logs: tool call counts, success/fail, timeouts, final sources mix, latency.

### Phase 6 – QA, Flags, and Rollout (0.5–1 day)
- Feature flag: `USE_FLASH_LIVE=1` to enable `/degenanalyze`.
- Smoke tests with limited tool access; tune domain caps and recency.
- Stage rollout; collect user feedback.

---

## Prompting & Policies (Live)

- System prompt (Live):
  - “You are a degen-aligned research agent for Solana memecoins. Find reputable, fresh sources; summarize key signals; cite links; obey rate limits; avoid spam.”
- Planning policy:
  - Mix team/project-specific, market context, and odds/price movement angles.
  - Cover catalysts, on-chain signals, and risks/rug checks.
  - Prefer fresh content (time window configured by `DDG_TIME`).
- Tone policy:
  - Meme-aware, fun, but informative and non-coercive.
  - Encourage reinvestment subtly (opt-in), never overpromise gains; always add NFA.

---

## Environment Variables

```
# Feature
USE_FLASH_LIVE=0
DEGEN_MAX_STREAM_DURATION=60

# Search bias
DDG_REGION=us-en
DDG_TIME=w

# API keys (optional, depending on tools you enable)
HELIUS_API_KEY=
SOLSCAN_API_KEY=
RUGCHECK_API_KEY=
COINGECKO_API_KEY=

# Rate/limits (examples)
LIVE_MAX_PER_DOMAIN=3
LIVE_MIN_REPUTABLE=5
LIVE_DOMAIN_WEIGHT=20
LIVE_RECENCY_WEIGHT=15
```

---

## Logging & Metrics

- LiveSession:
  - `live.plan` (queries planned), `live.search.succ/fail`, `live.fetch.succ/fail`, `live.extract.count`.
  - Source mix: `whitelist_count`, per-domain counts, top hostnames.
  - Streaming cadence and backoffs.

- Telegram:
  - Safe-edit wrappers with sanitizer + plain text fallback.
  - Stream frequency caps.

---

## Risks & Mitigations

- API quotas/rate limits → strict budgets, backoff, caching; user-visible progress when slowed.
- Gated content (paywalls/403) → keep citation-only entries; prefer accessible confirmations.
- Low-signal/meme noise → enforce reputable crypto outlets minimum and on-chain verification.
- Legal/ToS → avoid scraping disallowed paths; honor robots.txt; prefer APIs.

---

## Acceptance Criteria

- `/degenanalyze` streams progress within 2–5 seconds, completes within configured budget.
- Final report includes: Degen Thesis, Catalysts, On-chain Signals, CT Sentiment, Risks/Rug Checks, Entry/Exit Ideas, Reinvest Hook, Sources.
- Source mix contains minimum reputable domains and on-chain data when available.
- No change to `/deepanalyze` behavior.

---

## Next Steps

- Implement Phase 0–2 (flags, command skeleton, generic tools + streaming).
- Add Phase 3 crypto tools for Solana.
- Test with a few tokens/topics; tune domain caps, recency windows, and streaming cadence.
- Roll out behind `USE_FLASH_LIVE=1` and iterate.
