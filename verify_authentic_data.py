"""
Verify authentic data by checking real Barcelona vs Real Madrid match results
"""

import asyncio
import aiohttp
import os
from datetime import datetime

async def get_real_clasico_results():
    """Get actual Barcelona vs Real Madrid results from football-data.org"""
    
    api_key = os.getenv('FOOTBALL_API_KEY')
    if not api_key:
        print("❌ No Football API key available")
        return
        
    headers = {'X-Auth-Token': api_key}
    
    async with aiohttp.ClientSession() as session:
        try:
            # Get La Liga matches for recent seasons
            seasons = ['2024', '2023', '2022', '2021']
            clasico_matches = []
            
            for season in seasons:
                url = f"https://api.football-data.org/v4/competitions/PD/matches?season={season}"
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        matches = data.get('matches', [])
                        
                        for match in matches:
                            if match['status'] == 'FINISHED':
                                home_team = match['homeTeam']['name']
                                away_team = match['awayTeam']['name']
                                
                                # Check for Barcelona vs Real Madrid (both ways)
                                if (('Barcelona' in home_team and 'Real Madrid' in away_team) or 
                                    ('Real Madrid' in home_team and 'Barcelona' in away_team)):
                                    
                                    result = {
                                        'date': match['utcDate'][:10],
                                        'home_team': home_team,
                                        'away_team': away_team,
                                        'home_score': match['score']['fullTime']['home'],
                                        'away_score': match['score']['fullTime']['away'],
                                        'season': season,
                                        'competition': match['competition']['name']
                                    }
                                    clasico_matches.append(result)
                                    
                        await asyncio.sleep(6)  # Rate limiting
                    else:
                        print(f"Error {response.status} for season {season}")
                        
            # Sort by date (most recent first)
            clasico_matches.sort(key=lambda x: x['date'], reverse=True)
            
            print("🏆 AUTHENTIC EL CLASICO RESULTS (Last 4 matches):")
            print("=" * 60)
            
            for i, match in enumerate(clasico_matches[:4]):
                print(f"\n{i+1}. {match['date']} - {match['competition']}")
                print(f"   {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']}")
                
                # Determine winner
                if match['home_score'] > match['away_score']:
                    winner = match['home_team']
                elif match['away_score'] > match['home_score']:
                    winner = match['away_team']
                else:
                    winner = "Draw"
                print(f"   Winner: {winner}")
                
            print(f"\n✅ Found {len(clasico_matches)} authentic El Clasico matches in database")
            
        except Exception as e:
            print(f"❌ Error fetching authentic data: {e}")

if __name__ == "__main__":
    asyncio.run(get_real_clasico_results())