"""
TheSportsDB v1 provider for leagues and upcoming fixtures (free key friendly).
- Uses simple numeric key in URL path (e.g., /123/)
- Focused on Soccer only; fetches top leagues and pulls next events per league
- Adds basic in-memory caching and single aiohttp session
"""
from __future__ import annotations

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import aiohttp

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("THESPORTSDB_BASE_URL", "https://www.thesportsdb.com/api/v1/json")
DEFAULT_API_KEY = os.getenv("THESPORTSDB_API_KEY", "123")

# Common league name mappings to prefer when listing
PREFERRED_SOCCER_LEAGUES = {
    "English Premier League",
    "Spanish La Liga",
    "Italian Serie A",
    "German Bundesliga",
    "French Ligue 1",
}

@dataclass
class League:
    id: str
    name: str
    country: Optional[str] = None
    sport: Optional[str] = None

@dataclass
class Match:
    id: Optional[str]
    league_id: Optional[str]
    league_name: Optional[str]
    home_team: str
    away_team: str
    date: str  # YYYY-MM-DD
    match_time: str  # HH:MM (24h) or 'TBD'
    iso_timestamp: Optional[str] = None
    source: str = "TheSportsDB"

class TheSportsDbProvider:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or DEFAULT_API_KEY
        self.base = f"{BASE_URL}/{self.api_key}"
        self.session: Optional[aiohttp.ClientSession] = None
        # caches
        self._leagues_cache: Tuple[float, List[League]] = (0.0, [])
        self._upcoming_cache: Tuple[float, List[Match]] = (0.0, [])
        # TTLs - Extended for better performance with 30 req/min limit
        self._leagues_ttl = 24 * 3600  # 24 hours
        self._upcoming_ttl = 600  # 10 minutes as requested

    async def initialize(self) -> None:
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=12)
            self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("TheSportsDB provider initialized")

    async def close(self) -> None:
        try:
            if self.session and not self.session.closed:
                await self.session.close()
        except Exception:
            pass

    async def list_leagues(self) -> List[League]:
        """List core soccer leagues (filtered, cached)."""
        now = asyncio.get_running_loop().time()
        ts, cached = self._leagues_cache
        if cached and (now - ts) < self._leagues_ttl:
            return cached
        url = f"{self.base}/all_leagues.php"
        leagues: List[League] = []
        try:
            data = await self._get_json(url)
            items = (data or {}).get("leagues") or []
            for it in items:
                sport = (it.get("strSport") or "").strip()
                if sport.lower() != "soccer":
                    continue
                name = (it.get("strLeague") or "").strip()
                id_league = str(it.get("idLeague") or "").strip()
                country = (it.get("strCountry") or it.get("strLeagueAlternate") or "").strip() or None
                if name:
                    leagues.append(League(id=id_league, name=name, country=country, sport=sport))
            # Prefer top leagues at the top
            leagues.sort(key=lambda x: (0 if x.name in PREFERRED_SOCCER_LEAGUES else 1, x.name))
            self._leagues_cache = (now, leagues)
            return leagues
        except Exception as e:
            logger.error(f"TheSportsDB list_leagues error: {e}")
            return []

    async def get_upcoming_matches(self, max_total: int = 10) -> List[Match]:
        """Fetch upcoming matches with 10-minute cache optimization.
        Uses eventsnextleague.php for preferred soccer leagues with intelligent rate limiting.
        Results cached for 10 minutes to optimize the 30 requests/minute free tier limit.
        """
        now = asyncio.get_running_loop().time()
        ts, cached = self._upcoming_cache
        if cached and (now - ts) < self._upcoming_ttl:
            logger.info(f"Using 10-minute cache, {len(cached)} matches available")
            return cached
        
        leagues = await self.list_leagues()
        # Expand to include more sports as requested: NFL, MLS, NHL, NBA, etc.
        enhanced_preferred = PREFERRED_SOCCER_LEAGUES.union({
            "National Football League", "NFL", 
            "Major League Soccer", "MLS",
            "National Hockey League", "NHL",
            "National Basketball Association", "NBA",
            "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"
        })
        
        # Select up to 8 leagues for comprehensive coverage
        selected = [lg for lg in leagues if any(pref in lg.name for pref in enhanced_preferred)][:8]
        # If not enough preferred leagues found, add popular ones
        if len(selected) < 6:
            additional = [lg for lg in leagues if lg not in selected][:6-len(selected)]
            selected.extend(additional)
        
        matches: List[Match] = []
        request_count = 0
        
        logger.info(f"Fetching matches from {len(selected)} leagues with 10-min cache")
        
        for lg in selected:
            if not lg.id:
                continue
            
            # Conservative rate limiting for 30 req/min (allow some buffer)
            if request_count >= 10:  # Use max 1/3 of rate limit
                logger.info(f"Rate limit protection: stopping at {request_count} requests")
                break
                
            url = f"{self.base}/eventsnextleague.php?id={lg.id}"
            try:
                data = await self._get_json(url)
                request_count += 1
                
                events = (data or {}).get("events") or []
                for ev in events[:6]:  # Limit events per league for performance
                    mt = self._normalize_event(ev)
                    if mt:
                        matches.append(mt)
                        
            except Exception as e:
                logger.warning(f"eventsnextleague failed for {lg.id} {lg.name}: {e}")
            
            # Respectful rate limiting: 2 requests per second max
            await asyncio.sleep(0.5)
        
        # Sort by ISO timestamp/date
        def sort_key(m: Match):
            if m.iso_timestamp:
                try:
                    return datetime.fromisoformat(m.iso_timestamp.replace("Z", "+00:00"))
                except Exception:
                    pass
            return datetime.fromisoformat(f"{m.date}T{(m.match_time or '00:00')}:00+00:00") if m.date else datetime.now(timezone.utc)
        
        matches.sort(key=sort_key)
        # Cap to requested amount
        matches = matches[:max_total]
        
        # Cache for 10 minutes as requested
        self._upcoming_cache = (now, matches)
        logger.info(f"Cached {len(matches)} matches using {request_count} API calls")
        return matches
    
    async def get_upcoming_matches_for_league(self, league_id: str, max_matches: int = 12) -> List[Match]:
        """Get upcoming matches for a specific league with caching"""
        cache_key = f"league_{league_id}"
        now = asyncio.get_running_loop().time()
        
        # Check if we have league-specific cache
        if hasattr(self, '_league_cache'):
            if cache_key in self._league_cache:
                ts, cached_matches = self._league_cache[cache_key]
                if (now - ts) < 300:  # 5-minute cache for individual leagues
                    return cached_matches[:max_matches]
        else:
            self._league_cache = {}
        
        # Fetch fresh data
        url = f"{self.base}/eventsnextleague.php?id={league_id}"
        matches: List[Match] = []
        
        try:
            data = await self._get_json(url)
            events = (data or {}).get("events") or []
            
            for ev in events[:max_matches]:
                mt = self._normalize_event(ev)
                if mt:
                    matches.append(mt)
            
            # Cache the results
            self._league_cache[cache_key] = (now, matches)
            logger.info(f"Fetched {len(matches)} matches for league {league_id}")
            
        except Exception as e:
            logger.error(f"Error fetching matches for league {league_id}: {e}")
            
        return matches

    def _normalize_event(self, ev: Dict[str, Any]) -> Optional[Match]:
        try:
            home = (ev.get("strHomeTeam") or "").strip() or "TBD"
            away = (ev.get("strAwayTeam") or "").strip() or "TBD"
            id_event = str(ev.get("idEvent") or "").strip() or None
            id_league = str(ev.get("idLeague") or "").strip() or None
            league = (ev.get("strLeague") or "").strip() or None
            # Dates
            date_ev = (ev.get("dateEvent") or ev.get("dateEventLocal") or "").strip()
            time_ev = (ev.get("strTime") or ev.get("strTimeLocal") or "").strip()
            ts = (ev.get("dateEvent") or "") + ("T" + (ev.get("strTimestamp") or ev.get("strTime") or "00:00:00") if (ev.get("strTimestamp") or ev.get("strTime")) else "")
            # Some payloads provide strTimestamp as full ISO; prefer that if present
            iso = ev.get("strTimestamp") or None
            # Derive basic fields
            date = date_ev or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            match_time = "TBD"
            if time_ev:
                # Keep HH:MM
                match_time = time_ev[:5]
            elif iso:
                try:
                    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                    match_time = dt.strftime("%H:%M")
                except Exception:
                    match_time = "TBD"
            return Match(
                id=id_event,
                league_id=id_league,
                league_name=league,
                home_team=home,
                away_team=away,
                date=date,
                match_time=match_time or "TBD",
                iso_timestamp=iso,
                source="TheSportsDB",
            )
        except Exception as e:
            logger.debug(f"normalize_event error: {e}")
            return None

    async def _get_json(self, url: str) -> Optional[Dict[str, Any]]:
        if not self.session or self.session.closed:
            await self.initialize()
        assert self.session is not None
        async with self.session.get(url, headers={"Accept": "application/json"}) as resp:
            if resp.status == 429:
                logger.warning("TheSportsDB rate limit encountered (429)")
                return None
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
            return await resp.json()
