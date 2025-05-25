"""
Clean, streamlined sports data collector using football-data.org API
"""

import asyncio
import logging
import os
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class FootballDataCollector:
    """Streamlined sports data collector using football-data.org API"""
    
    def __init__(self):
        self.api_key = os.getenv('FOOTBALL_API_KEY')
        self.base_url = "https://api.football-data.org/v4"
        self.session = None
        
        # Major league IDs for football-data.org
        self.leagues = {
            'Premier League': 'PL',
            'La Liga': 'PD', 
            'Serie A': 'SA',
            'Bundesliga': 'BL1',
            'Ligue 1': 'FL1',
            'Champions League': 'CL',
            'Championship': 'ELC',
            'Eredivisie': 'DED',
            'Primeira Liga': 'PPL'
        }
    
    async def initialize(self):
        """Initialize the collector"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        logger.info("Football Data collector initialized")
    
    async def close(self):
        """Close the collector"""
        if self.session:
            await self.session.close()
    
    async def get_real_upcoming_matches(self) -> List[Dict]:
        """Get upcoming matches from football-data.org for next 7 days"""
        matches = []
        
        try:
            if not self.api_key:
                logger.error("FOOTBALL_API_KEY not found")
                return []
            
            today = datetime.now()
            end_date = today + timedelta(days=7)
            
            headers = {
                'X-Auth-Token': self.api_key,
                'Content-Type': 'application/json'
            }
            
            # Create a fresh session for this request
            async with aiohttp.ClientSession() as session:
                for league_name, league_code in self.leagues.items():
                    try:
                        logger.info(f"Fetching fixtures for {league_name}...")
                        
                        # Get matches for this league
                        url = f"{self.base_url}/competitions/{league_code}/matches"
                        params = {
                            'dateFrom': today.strftime('%Y-%m-%d'),
                            'dateTo': end_date.strftime('%Y-%m-%d'),
                            'status': 'SCHEDULED'
                        }
                        
                        async with session.get(url, headers=headers, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                league_matches = data.get('matches', [])
                                
                                for match in league_matches:
                                    parsed_match = self._parse_match(match, league_name)
                                    if parsed_match:
                                        matches.append(parsed_match)
                                
                                logger.info(f"Got {len(league_matches)} matches from {league_name}")
                            
                            elif response.status == 429:
                                logger.warning(f"Rate limit hit for {league_name}")
                                await asyncio.sleep(60)  # Wait 1 minute for rate limit
                            
                            else:
                                logger.warning(f"API error {response.status} for {league_name}")
                    
                    except Exception as e:
                        logger.error(f"Error getting {league_name} fixtures: {e}")
                        continue
                    
                    # Small delay between requests to respect rate limits
                    await asyncio.sleep(0.1)
            
            # Sort matches by date
            matches.sort(key=lambda x: x.get('date', ''))
            
            logger.info(f"Total matches retrieved: {len(matches)}")
            return matches
            
        except Exception as e:
            logger.error(f"Error in get_real_upcoming_matches: {e}")
            return []
    
    def _parse_match(self, match: Dict, league_name: str) -> Optional[Dict]:
        """Parse match data from football-data.org API"""
        try:
            home_team = match.get('homeTeam', {}).get('name', 'TBD')
            away_team = match.get('awayTeam', {}).get('name', 'TBD')
            
            if not home_team or not away_team:
                return None
            
            # Parse match date and time
            utc_date = match.get('utcDate', '')
            match_date = ''
            match_time = 'TBD'
            
            if utc_date:
                try:
                    parsed_datetime = datetime.fromisoformat(utc_date.replace('Z', '+00:00'))
                    match_date = parsed_datetime.strftime('%Y-%m-%d')
                    match_time = parsed_datetime.strftime('%H:%M')
                except:
                    match_date = datetime.now().strftime('%Y-%m-%d')
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'league': league_name,
                'match_time': match_time,
                'date': match_date,
                'source': 'Football-Data.org'
            }
            
        except Exception as e:
            logger.debug(f"Error parsing match: {e}")
            return None