import os
import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Optional: Flash-powered query generation
try:
    import google.generativeai as genai
except Exception:
    genai = None


def _flag_on(name: str) -> bool:
    return str(os.getenv(name, '0')).strip().lower() in ('1', 'true', 'yes', 'on')

def generate_search_queries(event_query: str, sport_type: Optional[str] = None) -> List[str]:
    """
    Generate targeted search queries for web research based on a sports event query.
    Feature-flagged: USE_FLASH_QUERY_GENERATION=1 to enable Gemini Flash 2.5 assisted queries.
    
    Args:
        event_query (str): The event query (e.g., "Manchester United vs Liverpool")
        sport_type (str, optional): The sport type (e.g., "soccer", "basketball", "tennis")
        
    Returns:
        List[str]: A list of targeted search query strings
    """
    # Parse teams/players from the query
    team_a, team_b = _parse_teams(event_query)

    # Phase 1 default: Flash-assisted generation is required; no template fallback.
    if genai is None or not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Flash query generation required but GEMINI_API_KEY or client is unavailable")
    try:
        flash_queries = _generate_with_flash(team_a, team_b, sport_type or _detect_sport_type(event_query), event_query)
        if not flash_queries:
            raise RuntimeError("Flash query generation returned no candidates")
        logger.info(f"Flash query generation USED | candidates={len(flash_queries)}")
        # Build coverage queries to ensure diverse intents (team-specific, league, injuries, pressers, tactics, lineups, odds, official)
        coverage = _build_coverage_queries(team_a, team_b, sport_type or _detect_sport_type(event_query))
        mixed = _mix_queries(flash_queries, coverage, int(os.getenv("MAX_QUERIES", "7")), team_a, team_b)
        return mixed
    except Exception as e:
        logger.error(f"Flash query generation failed (no fallback): {e}")
        raise

def _parse_teams(event_query: str) -> tuple:
    """
    Parse teams/players from the event query.
    
    Args:
        event_query (str): The event query
        
    Returns:
        tuple: A tuple containing (team_a, team_b) or (event_query, "opponent") if no delimiter found
    """
    # Split on "vs" or "versus" (case insensitive)
    parts = re.split(r'\s+(?:vs|versus)\s+', event_query, flags=re.IGNORECASE)
    
    if len(parts) >= 2:
        # Take the first two parts as teams
        team_a = parts[0].strip()
        team_b = parts[1].split()[0].strip() if parts[1].split() else parts[1].strip()
        return team_a, team_b
    else:
        # Fallback mechanism if no "vs" or "versus" found
        # Try to split on common delimiters
        delimiters = ['-', '–', '&', 'and', '@']
        for delimiter in delimiters:
            parts = re.split(r'\s+' + re.escape(delimiter) + r'\s+', event_query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()
        
        # If still no delimiter found, return the whole query as team_a and a generic term
        return event_query.strip(), "opponent"

def _detect_sport_type(event_query: str) -> str:
    """
    Attempt to detect sport type from the event query.
    """
    sport_keywords = {
        'soccer': ['fc', 'united', 'city', 'afc', 'league', 'premier', 'championship', 'cup', 'real', 'barcelona', 'madrid'],
        'basketball': ['nba', 'basketball', 'lakers', 'warriors', 'celtics'],
        'tennis': ['atp', 'wta', 'tennis', 'open'],
        'american football': ['nfl', 'football'],
        'baseball': ['mlb', 'baseball'],
        'hockey': ['nhl', 'hockey', 'ice'],
        'golf': ['pga', 'golf', 'masters'],
    }
    query_lower = event_query.lower()
    for sport, keywords in sport_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                return sport
    return "unknown"

def _generate_sport_specific_queries(team_a: str, team_b: str, sport_type: str) -> List[str]:
    """Generate sport-specific queries based on the sport type."""
    sport_queries = []

    def _site_hint(q: str, sites: List[str]) -> List[str]:
        # mix site: filters into some variants
        hinted = [q]
        for s in sites[:2]:  # at most 2 site hints to keep diversity
            hinted.append(f"{q} site:{s}")
        return hinted

    soccer_sites = [
        "bbc.com", "espn.com", "skysports.com", "theguardian.com",
        "whoscored.com", "transfermarkt.com", "uefa.com", "premierleague.com"
    ]

    if sport_type == "soccer":
        base = [
            f"{team_a} vs {team_b} match preview",
            f"{team_a} starting XI vs {team_b}",
            f"{team_b} manager tactical analysis",
            f"{team_a} vs {team_b} injury update",
            f"{team_a} vs {team_b} press conference"
        ]
        for q in base:
            sport_queries.extend(_site_hint(q, soccer_sites))
    elif sport_type == "basketball":
        base = [
            f"{team_a} vs {team_b} season stats comparison",
            f"{team_a} star player performance vs {team_b}",
            f"{team_b} injury report and depth chart"
        ]
        for q in base:
            sport_queries.append(q)
    elif sport_type == "tennis":
        base = [
            f"{team_a} vs {team_b} head to head statistics",
            f"{team_a} recent tournament form",
            f"{team_b} surface preference analysis"
        ]
        for q in base:
            sport_queries.append(q)
    elif sport_type == "american football":
        base = [
            f"{team_a} vs {team_b} season record",
            f"{team_a} offensive strategy vs {team_b} defense",
            f"{team_b} quarterback performance analysis"
        ]
        for q in base:
            sport_queries.append(q)
    elif sport_type == "baseball":
        base = [
            f"{team_a} vs {team_b} pitching matchup",
            f"{team_a} batting order analysis",
            f"{team_b} bullpen situation report"
        ]
        for q in base:
            sport_queries.append(q)
    elif sport_type == "hockey":
        base = [
            f"{team_a} vs {team_b} power play statistics",
            f"{team_a} goaltender matchup analysis",
            f"{team_b} injury report and lineup news"
        ]
        for q in base:
            sport_queries.append(q)
    elif sport_type == "golf":
        base = [
            f"{team_a} vs {team_b} course history",
            f"{team_a} recent tournament results",
            f"{team_b} putting statistics this season"
        ]
        for q in base:
            sport_queries.append(q)
    else:
        base = [
            f"{team_a} vs {team_b} latest news",
            f"{team_a} key players to watch vs {team_b}",
            f"{team_b} tactical analysis and strategy",
        ]
        for q in base:
            sport_queries.append(q)

    return sport_queries


def _build_coverage_queries(team_a: str, team_b: str, sport_type: str) -> List[str]:
    """Build coverage queries to guarantee a balanced mix (team A, team B, league, odds, official)."""
    league_terms = ["league standings", "fixtures congestion", "disciplinary rules", "suspension rules", "transfer window impact"]
    team_terms = ["form", "selection", "injury update", "suspension", "tactical setup", "press conference", "probable lineup"]
    odds_terms = ["odds", "betting lines", "price movement", "bookmaker"]

    queries: List[str] = []
    # Team-specific coverage
    for t in team_terms:
        queries.append(f"{team_a} {t}")
        queries.append(f"{team_b} {t}")
    # League/competition context
    for lt in league_terms:
        queries.append(f"{team_a} {lt}")
        queries.append(f"{team_b} {lt}")
    # Odds/bookmaker context
    for ot in odds_terms:
        queries.append(f"{team_a} vs {team_b} {ot}")
        queries.append(f"{team_a} {ot}")
        queries.append(f"{team_b} {ot}")
    # Official communications
    queries.append(f"{team_a} official site announcements")
    queries.append(f"{team_b} official site announcements")

    # Normalize and dedupe
    queries = [q.strip() for q in queries if q.strip()]
    queries = _dedupe_and_cap(queries, max_queries=50)
    return queries


def _mix_queries(primary: List[str], coverage: List[str], cap: int, team_a: str = None, team_b: str = None) -> List[str]:
    """Interleave Flash queries with coverage queries to ensure intent diversity."""
    # Prioritize site: queries first from primary
    prim_site = [q for q in primary if 'site:' in q]
    prim_other = [q for q in primary if 'site:' not in q]

    mix: List[str] = []
    # Take some primary site queries
    take_site = min(len(prim_site), max(2, cap // 3))
    mix.extend(prim_site[:take_site])

    # Enforce paired team coverage: for every team_a query we try to include a team_b counterpart
    def is_team_q(q: str, team: str) -> bool:
        return team and team.lower() in q.lower()

    team_a_q = [q for q in coverage if is_team_q(q, team_a)]
    team_b_q = [q for q in coverage if is_team_q(q, team_b)]

    # Interleave A/B team queries
    i = 0
    while len(mix) < cap and (i < len(team_a_q) or i < len(team_b_q)):
        if i < len(team_a_q):
            mix.append(team_a_q[i])
        if i < len(team_b_q) and len(mix) < cap:
            mix.append(team_b_q[i])
        i += 1

    # Add remaining coverage (league/odds/official), then primary others
    remain_cov = [q for q in coverage if q not in team_a_q and q not in team_b_q]
    need = cap - len(mix)
    if need > 0:
        mix.extend(remain_cov[:need])
    need = cap - len(mix)
    if need > 0:
        mix.extend(prim_other[:need])

    return _dedupe_and_cap(mix, max_queries=cap)

def _dedupe_and_cap(queries: List[str], max_queries: int = None) -> List[str]:
    """Dedupe while preserving order and cap length (default 7)."""
    seen = set()
    unique = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    if max_queries is None:
        max_queries = int(os.getenv("MAX_QUERIES", "7"))
    return unique[:max_queries]


def _generate_with_flash(team_a: str, team_b: str, sport_type: str, event_query: str) -> List[str]:
    """Generate query candidates via Gemini Flash 2.5 then return a curated list."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return []
    genai.configure(api_key=api_key)
    model_id = os.getenv("GEMINI_FLASH_MODEL_ID") or "gemini-2.5-flash"
    model = genai.GenerativeModel(model_id)

    prompt = f"""
You are a search query strategist for sports intelligence.
Generate targeted queries for the event: {event_query}.
Cover a balanced mix of intents:
- Head-to-head context and historical trends
- Team-specific news for each team (form, selection, tactics)
- League/competition context and how it affects this event (schedule congestion, standings, disciplinary rules)
- Injuries & suspensions (each team)
- Press conferences / manager quotes
- Tactical previews / expected setup
- Probable lineups
- Official communications (club/league)
- Bookmaker/odds availability (if any) and lines movement

Include 3–5 queries with site: filters for reputable sources (espn.com, skysports.com, bbc.com, theguardian.com, whoscored.com, transfermarkt.com, theathletic.com, official club/league).
Favor current/recency words ("this week", today). Return one query per line, no numbering.
Teams: {team_a} vs {team_b}
Sport: {sport_type}
"""
    resp = model.generate_content(prompt)
    text = getattr(resp, "text", "") or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    # Prioritize site: queries first to bias reputable sources
    site_lines = [ln for ln in lines if 'site:' in ln]
    non_site_lines = [ln for ln in lines if 'site:' not in ln]
    return site_lines + non_site_lines
