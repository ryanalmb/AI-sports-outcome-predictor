"""
Task-specific prompts for Flash Live Degen feature.
This module defines prompts for different types of sports research tasks.
"""

# Query intent prompts for different types of sports information
QUERY_INTENT_PROMPTS = {
    "injuries": "Find recent injury reports and player availability updates for {team_a} and {team_b}",
    "press_conference": "Find recent press conference quotes and team news for {team_a} vs {team_b}",
    "tactics": "Find tactical insights, formation changes, and strategic analysis for {team_a} vs {team_b}",
    "odds": "Find current odds, betting lines, and market movements for {team_a} vs {team_b}",
    "lineup": "Find probable lineups and team selection news for {team_a} vs {team_b}",
    "form": "Find recent form analysis and performance statistics for {team_a} and {team_b}",
    "head_to_head": "Find head-to-head statistics and historical performance for {team_a} vs {team_b}",
    "venue": "Find venue-specific information, weather conditions, and home advantage factors for {team_a} vs {team_b}"
}

# Search query refinement prompts
SEARCH_REFINEMENT_PROMPTS = {
    "initial": "Generate diverse search queries for the {sport} match between {team_a} and {team_b}",
    "follow_up": "Refine search queries based on initial results to find more specific information",
    "coverage": "Ensure mix of fresh and reputable sources by adjusting search terms"
}

# Sports-specific prompt templates
SPORTS_SPECIFIC_PROMPTS = {
    "soccer": {
        "focus_areas": ["injuries", "tactics", "lineup", "form", "head_to_head", "venue"],
        "key_terms": ["starting XI", "injury update", "tactical analysis", "formation", "press conference"]
    },
    "basketball": {
        "focus_areas": ["injuries", "form", "head_to_head", "odds"],
        "key_terms": ["injury report", "player stats", "head-to-head", "betting lines", "depth chart"]
    },
    "tennis": {
        "focus_areas": ["form", "head_to_head", "venue"],
        "key_terms": ["recent form", "head-to-head", "surface preference", "injury update"]
    },
    "american_football": {
        "focus_areas": ["injuries", "tactics", "form", "odds"],
        "key_terms": ["injury report", "offensive strategy", "defensive analysis", "betting lines"]
    },
    "baseball": {
        "focus_areas": ["injuries", "form", "odds"],
        "key_terms": ["pitching matchup", "batting order", "injury report", "betting lines"]
    },
    "hockey": {
        "focus_areas": ["injuries", "form", "odds"],
        "key_terms": ["injury report", "lineup news", "power play", "betting lines"]
    }
}

# Prompt for generating search queries
SEARCH_QUERY_GENERATION_PROMPT = """Generate targeted search queries for researching the {sport} match between {team_a} and {team_b}.

Instructions:
1. Create diverse queries covering different aspects:
   - Team news and injuries
   - Tactical analysis and formations
   - Recent form and performance
   - Head-to-head statistics
   - Odds and betting movements
   - Press conference quotes
   - Venue and weather conditions
2. Ensure a mix of fresh and reputable sources
3. Use sport-specific terminology
4. Include site-specific searches for reputable domains when appropriate

Return a list of 10-15 search queries as a JSON array.
"""

# Prompt for evaluating search results
SEARCH_RESULT_EVALUATION_PROMPT = """Evaluate the relevance and quality of search results for sports research.

Criteria:
1. Relevance to the specific match ({team_a} vs {team_b})
2. Information richness (contains concrete facts vs. speculation)
3. Source credibility (official sites, reputable sports journalism)
4. Recency (prefer recent sources)
5. Diversity (different aspects of the match)

Rate each result on a scale of 1-100 and select the top 5 most relevant sources.
"""

# Prompt for content analysis
CONTENT_ANALYSIS_PROMPT = """Analyze the content from the provided sports article.

Extract and organize the following information:
1. Match details (teams, competition, date)
2. Key injuries and suspensions
3. Tactical insights and formation changes
4. Player performance analysis
5. Manager/Coach quotes from press conferences
6. Odds movements and betting insights
7. Venue conditions and external factors
8. Confidence score (0-100) in the reliability of information

Format your response as structured JSON with appropriate keys for each category.
"""

def get_query_intent_prompt(intent: str, team_a: str, team_b: str) -> str:
    """
    Get a query intent prompt for a specific research intent.
    
    Args:
        intent: The research intent (e.g., 'injuries', 'press_conference')
        team_a: The first team name
        team_b: The second team name
        
    Returns:
        The formatted query intent prompt
    """
    template = QUERY_INTENT_PROMPTS.get(intent, "Find information about {team_a} vs {team_b}")
    return template.format(team_a=team_a, team_b=team_b)

def get_search_refinement_prompt(refinement_type: str, sport: str, team_a: str, team_b: str) -> str:
    """
    Get a search refinement prompt.
    
    Args:
        refinement_type: The type of refinement (e.g., 'initial', 'follow_up')
        sport: The sport type
        team_a: The first team name
        team_b: The second team name
        
    Returns:
        The formatted search refinement prompt
    """
    template = SEARCH_REFINEMENT_PROMPTS.get(refinement_type, "Refine search for {sport} match")
    return template.format(sport=sport, team_a=team_a, team_b=team_b)

def get_sports_specific_prompt(sport: str) -> dict:
    """
    Get sports-specific prompts and focus areas.
    
    Args:
        sport: The sport type
        
    Returns:
        Dictionary with sports-specific prompts and focus areas
    """
    return SPORTS_SPECIFIC_PROMPTS.get(sport, SPORTS_SPECIFIC_PROMPTS["soccer"])

def get_search_query_generation_prompt(sport: str, team_a: str, team_b: str) -> str:
    """
    Get the search query generation prompt.
    
    Args:
        sport: The sport type
        team_a: The first team name
        team_b: The second team name
        
    Returns:
        The formatted search query generation prompt
    """
    return SEARCH_QUERY_GENERATION_PROMPT.format(sport=sport, team_a=team_a, team_b=team_b)

def get_search_result_evaluation_prompt(team_a: str, team_b: str) -> str:
    """
    Get the search result evaluation prompt.
    
    Args:
        team_a: The first team name
        team_b: The second team name
        
    Returns:
        The formatted search result evaluation prompt
    """
    return SEARCH_RESULT_EVALUATION_PROMPT.format(team_a=team_a, team_b=team_b)

def get_content_analysis_prompt() -> str:
    """
    Get the content analysis prompt.
    
    Returns:
        The content analysis prompt string
    """
    return CONTENT_ANALYSIS_PROMPT