"""
Content processing module for the /deepanalyze command.
This module processes articles using the Gemini 2.5 Flash model to extract relevant information.
"""

import os
import asyncio
import json
import logging
import time
import re
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for concurrency control
MAX_CONCURRENT_API_CALLS = int(os.getenv('CONTENT_MAX_CONCURRENT_API_CALLS', '5'))  # Limit concurrent API calls
INITIAL_RETRY_DELAY = float(os.getenv('CONTENT_INITIAL_RETRY_DELAY', '1.0'))  # Initial delay for retries in seconds
MAX_RETRIES = int(os.getenv('CONTENT_MAX_RETRIES', '3'))  # Maximum number of retries for rate limiting

# Prompt template for Gemini 2.5 Flash
PROMPT_TEMPLATE = """
You are a data extraction AI.
From the article text below, do a relevance check and then provide a concise summary if relevant.

Required:
- First line must be: RELEVANCE: YES or RELEVANCE: NO
- If YES, include a 8-10 sentence SUMMARY. Optionally include bullets for players, injuries/suspensions, and key stats.
- JSON is NOT required; plain text is acceptable.

--- ARTICLE TEXT ---
{article_text}
"""

class ContentProcessor:
    def __init__(self):
        """
        Initialize the ContentProcessor with the Gemini 2.5 Flash model.
        """
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the Gemini model with the API key from environment variables.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.5 Flash model
        flash_model_id = os.getenv("GEMINI_FLASH_MODEL_ID") or "gemini-2.5-flash"
        try:
            self.model = genai.GenerativeModel(
                flash_model_id,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Flash model: {e}")
            raise

    async def _call_gemini_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Call the Gemini API with retry mechanisms for rate limiting.
        
        Args:
            prompt: The prompt to send to the Gemini API
            
        Returns:
            Parsed JSON response or None if failed
        """
        retries = 0
        delay = INITIAL_RETRY_DELAY
        
        while retries <= MAX_RETRIES:
            try:
                # Call the Gemini API
                response = await self.model.generate_content_async(prompt)
                
                # Extract text from response
                if hasattr(response, 'text') and response.text:
                    response_text = response.text
                else:
                    # Try to extract from candidates
                    response_text = None
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            response_text = part.text
                                            break
                                    if response_text:
                                        break
                
                if response_text:
                    # Parse JSON response
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON response: {e}. Response text: {response_text[:200]}...")
                        # Try to extract JSON from the response
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            try:
                                return json.loads(json_match.group())
                            except json.JSONDecodeError:
                                pass
                        return None
                else:
                    logger.warning("Empty response from Gemini API")
                    return None
                    
            except Exception as e:
                # Check if it's a rate limiting error (429)
                if "429" in str(e) or "quota" in str(e).lower():
                    retries += 1
                    if retries > MAX_RETRIES:
                        logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries: {e}")
                        raise
                    else:
                        logger.warning(f"Rate limit hit, retrying in {delay} seconds: {e}")
                        await asyncio.sleep(delay)
                        delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Error calling Gemini API: {e}")
                    # For non-rate limit errors, don't retry
                    raise
    
    async def _process_single_article(self, article: Dict[str, Any], team_a: str, team_b: str) -> Optional[Dict[str, Any]]:
        """
        Process a single article using the Gemini 2.5 Flash model.
        
        Args:
            article: Dictionary containing 'url' and 'text' keys
            team_a: Name of the first team
            team_b: Name of the second team
            
        Returns:
            Parsed JSON response or None if not relevant or failed
        """
        try:
            # Create the prompt using the template
            prompt = PROMPT_TEMPLATE.format(
                team_a=team_a,
                team_b=team_b,
                article_text=article.get('text', '')
            )
            
            # Call the Gemini API
            result = await self._call_gemini_api(prompt)

            # Accept JSON or plain text; only drop if explicitly RELEVANCE: NO
            if isinstance(result, dict):
                if result.get("relevance") == "NO":
                    return None
                result["url"] = article.get("url", "")
                return self._clean_and_validate_data(result)

            if isinstance(result, str):
                txt = result.strip()
                if not txt:
                    return None
                first_line = txt.splitlines()[0].strip().lower()
                if first_line.startswith('relevance:') and 'no' in first_line:
                    return None
                return {
                    "url": article.get("url", ""),
                    "summary": re.sub(r"\s+", " ", txt)[:800],
                    "key_players_mentioned": [],
                    "injuries_or_suspensions": [],
                    "statistics": []
                }

            return None
                
        except Exception as e:
            logger.error(f"Error processing article from {article.get('url', 'unknown')}: {e}")
            return None
    
    def _clean_and_validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and validate the extracted data.
        
        Args:
            data: Raw extracted data
            
        Returns:
            Cleaned and validated data
        """
        cleaned_data = data.copy()
        
        # Remove duplicate player names
        if "key_players_mentioned" in cleaned_data and isinstance(cleaned_data["key_players_mentioned"], list):
            # Remove duplicates while preserving order
            seen = set()
            unique_players = []
            for player in cleaned_data["key_players_mentioned"]:
                if player not in seen:
                    seen.add(player)
                    unique_players.append(player)
            cleaned_data["key_players_mentioned"] = unique_players
        
        # Remove duplicate injuries/suspensions
        if "injuries_or_suspensions" in cleaned_data and isinstance(cleaned_data["injuries_or_suspensions"], list):
            # Remove duplicates while preserving order
            seen = set()
            unique_injuries = []
            for injury in cleaned_data["injuries_or_suspensions"]:
                if injury not in seen:
                    seen.add(injury)
                    unique_injuries.append(injury)
            cleaned_data["injuries_or_suspensions"] = unique_injuries
        
        # Validate statistics format
        if "statistics" in cleaned_data and isinstance(cleaned_data["statistics"], list):
            # Ensure all statistics are strings
            cleaned_data["statistics"] = [str(stat) for stat in cleaned_data["statistics"]]
        
        # Clean up summary text
        if "summary" in cleaned_data and isinstance(cleaned_data["summary"], str):
            # Remove extra whitespace and normalize
            cleaned_data["summary"] = re.sub(r'\s+', ' ', cleaned_data["summary"]).strip()
        
        return cleaned_data
    
    async def process_content_with_flash(self, articles: List[Dict[str, Any]], teams: List[str]) -> List[Dict[str, Any]]:
        """
        Process articles using the Gemini 2.5 Flash model to extract relevant information.
        
        Args:
            articles: List of articles from data acquisition
            teams: List of team names [team_a, team_b]
            
        Returns:
            List of parsed JSON objects from relevant articles
        """
        if len(teams) < 2:
            raise ValueError("Teams list must contain at least two team names")
        
        team_a, team_b = teams[0], teams[1]
        
        # Create a semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
        
        async def process_with_semaphore(article):
            async with semaphore:
                return await self._process_single_article(article, team_a, team_b)
        
        # Process articles concurrently
        tasks = [process_with_semaphore(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Error processing article: {result}")
            elif result is not None:
                # Clean and validate the data
                cleaned_result = self._clean_and_validate_data(result)
                valid_results.append(cleaned_result)
        
        return valid_results


# Convenience function for backward compatibility
async def process_content_with_flash(articles: List[Dict[str, Any]], teams: List[str]) -> List[Dict[str, Any]]:
    """
    Process articles using the Gemini 2.5 Flash model to extract relevant information.
    
    Args:
        articles: List of articles from data acquisition
        teams: List of team names [team_a, team_b]
        
    Returns:
        List of parsed JSON objects from relevant articles
    """
    processor = ContentProcessor()
    return await processor.process_content_with_flash(articles, teams)