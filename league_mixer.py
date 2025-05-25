"""
Simple league mixer to ensure diverse odds display
"""

def mix_leagues_for_display(matches_by_league, max_total=8, max_per_league=2):
    """Mix matches from different leagues for diverse display"""
    mixed_matches = []
    
    # Round-robin selection from available leagues
    while len(mixed_matches) < max_total and any(matches_by_league.values()):
        for league_name in list(matches_by_league.keys()):
            if not matches_by_league[league_name]:
                continue
                
            # Count how many from this league we already have
            current_count = sum(1 for m in mixed_matches if m['league'] == league_name)
            
            if current_count < max_per_league:
                mixed_matches.append(matches_by_league[league_name].pop(0))
                
            if len(mixed_matches) >= max_total:
                break
    
    return mixed_matches