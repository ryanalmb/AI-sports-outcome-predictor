# Flash Live Implementation Plan for /degenanalyze (Sports Predictions, Degen-Toned)

This plan adds a new command, `/degenanalyze`, that delivers cutting‑edge sports predictions in degen language loved by the Solana community. It borrows the battle‑tested pipeline from `/deepanalyze`, replaces Steps 1–3 with Gemini Flash Live (for real‑time, tool‑using research and extraction), and keeps Gemini Pro for Step 4 final synthesis. `/deepanalyze` remains untouched.

Key principles:
- Same sport intelligence depth as `/deepanalyze`, but streamed and phrased for degens.
- Strong sourcing from reputable outlets; explicit citations.
- Degen‑themed, humorous framing (NFA); tasteful nudge to "full port gains back into the project" without coercion.

---

## UX and Output

- Command:
  - `/degenanalyze TeamA vs TeamB` (e.g., `/degenanalyze Chelsea vs Liverpool`).
- Live streamed updates (every few seconds or on major finds):
  - "Found fresh injury note on [Sky]"
  - "Bookie line shift detected: H2H odds moved +X%"
  - "Press conference quote: …"
- Final degen‑toned report (Telegram‑safe):
  - "Degen Playbook" (summary in meme‑aware tone, clear but punchy)
  - "YOLO Factors" (top 3–5 catalysts: injuries, form, tactics, venue)
  - "Odds & Market Vibes" (bookmaker lines and movement; H2H probabilities)
  - "Rug Checks" (risks/uncertainties: rotations, weather, travel)
  - "Entry/Exit Spin" (how to time, NFA)
  - "HFSP Hook" (humorous, opt‑in, tasteful nudge: reinvest a slice of wins to support the project treasury/LP)
  - Sources (URLs/domains)

---

## Architecture Overview

- New Live path (coexists with `/deepanalyze`):
  - `fixed_bot.py`: add `/degenanalyze` handler (no changes to `/deepanalyze`).
  - `live/` (new):
    - `live_session.py`: Flash Live session orchestrator (state, tool routing, streaming cadence).
    - `tools/`: 
      - `web_search(query, region, time) -> [title, url, snippet]`
      - `fetch_url(url) -> {status, headers, html}`
      - `html_to_text(html) -> {text, meta}`
      - `parse_published_time(html|text) -> ISO8601?`
      - (optional) official provider tool wrappers (e.g., TheSportsDB for fixtures)
    - `policies.py`: domain whitelist, per‑domain caps, recency windows.
  - `renderers/` (new): Telegram‑safe degen copywriters (stream and final).
  - `prompts/` (new): Live system + task prompts for sports context.

- Pipeline split:
  - `/deepanalyze`: Flash (non‑live) for Steps 1–3 + Pro Step 4 (existing code).
  - `/degenanalyze`: Flash Live for Steps 1–3 + Pro Step 4 (new code).

---

## Phased Implementation

### Phase 0 – Foundations (0.5 day)
- Env flags (no behavioral change yet):
  - `USE_FLASH_LIVE=0` (default off)
  - `DEGEN_STREAM_INTERVAL_MS=1200` (coalescing streamed updates)
  - `DDG_REGION=us-en`, `DDG_TIME=w` (bias recency & locale)
- Skeleton dirs: `live/`, `live/tools/`, `live/renderers/`, `live/prompts/`.

### Phase 1 – Command & LiveSession (1 day)
- `/degenanalyze` handler:
  - Validate input; open LiveSession; post initial degen‑toned message.
  - Stream incremental updates with safe‑edit fallback (sanitize → plain text).
- `live_session.py`:
  - Initialize Flash Live model; register tools; hold session state (found_sources, attempted_domains, current best lines).
  - Enforce budgets: stream time cap, per‑domain caps, backoffs.

### Phase 2 – Generic Tools & Policies (1–2 days)
- Tools: `web_search`, `fetch_url`, `html_to_text`, `parse_published_time`.
- Policies:
  - Reputable domains whitelist for sports (BBC, Guardian, Reuters/AP, ESPN, Sky, WhoScored, Transfermarkt, league/club sites, etc.).
  - Scoring: domain weight, recency weight; enforce minimum whitelist hits & per‑domain caps.
  - Region/time filters (DDG) for freshness.
- Live prompt instructs:
  - Propose diverse query intents (A/B team, league context, injuries, pressers, tactics, odds, official).
  - Iteratively refine until the mix is fresh and reputable.

### Phase 3 – Sports Extraction & Market Signals (1–2 days)
- Live orchestrates page extraction:
  - Pull summaries; highlight injuries, pressers quotes, tactics, venue/travel/weather when relevant.
  - Detect odds lines/H2H probabilities and movement from reputable betting coverage; scraping of bookmaker sites.
- Maintain structured state for Pro (Step 4):
  - factors[], injuries[], quotes[], odds_snapshot, sources[]

### Phase 4 – Degen Renderers & Pro Synthesis (1 day)
- Final message builder (Telegram‑safe):
  - Degen Playbook, YOLO Factors, Odds & Market Vibes, Rug Checks, Entry/Exit Spin (always NFA), HFSP Hook.
- Call Gemini Pro for polishing the final summary, preserving tone and Sources.

### Phase 5 – Observability & Rate Limits (0.5–1 day)
- Metrics/logs: tool calls, succ/fail, timeouts, selected domains, publish times, stream cadence.
- Per‑domain quotas, exponential backoff, session cache for sources already fetched.

### Phase 6 – QA & Rollout (0.5 day)
- Gated via `USE_FLASH_LIVE=1`.
- Smoke test across multiple leagues & timezones; tune weights and degen tone.
- Ship.

---

## Prompting & Tone

- Live System prompt (sports degen):
  - "You are a degen‑aligned sports research agent. Find fresh, reputable sources; extract key signals (injuries, pressers, tactics, odds); cite links; adhere to rate limits. Keep it punchy and fun; never coercive; always NFA."
- Final tone:
  - Meme‑aware, concise, witty; e.g.,
    - "Degen Thesis: Midfield cooks; backline coping. Expect pressure early; watch the press."
    - "HFSP Hook: If you print, consider yeeting a slice back into the treasury—keeps the degen machine humming (NFA)."

---

## Environment Variables

```
# Feature
USE_FLASH_LIVE=0
DEGEN_STREAM_INTERVAL_MS=1200

# Search bias
DDG_REGION=us-en
DDG_TIME=w

# Live scoring/tunables (examples)
LIVE_MIN_REPUTABLE=5
LIVE_MAX_PER_DOMAIN=3
LIVE_DOMAIN_WEIGHT=20
LIVE_RECENCY_WEIGHT=15
```

---

## Logging & Metrics

- Plan: queries planned; site: expansions; per‑domain caps applied.
- Search: succ/fail counts; region/time used.
- Selection: whitelist vs others; top hostnames; publish times range.
- Streaming: updates count; throttling events.

---

## Risks & Mitigations

- API quotas/rate limits → strict budgets, backoffs, session caching.
- Gated content → keep citation‑only entries; prefer accessible confirmations.
- Over‑memeing → keep signal density high; degen tone but professional accuracy; always NFA.

---

## Acceptance Criteria

- `/degenanalyze` streams useful updates quickly and completes within the configured budget.
- Final message includes degen‑toned sections and explicit Sources.
- High share of reputable domains; fresh publish times; odds context when available.
- `/deepanalyze` remains unchanged.

---

## Next Steps

- Implement Phases 0–2 (flags, command skeleton, generic tools + policies + streaming).
- Add Phase 3 extraction improvements and odds context.
- Add Phase 4 renderers with degen tone and Pro polishing.
- Roll out behind `USE_FLASH_LIVE=1`.
