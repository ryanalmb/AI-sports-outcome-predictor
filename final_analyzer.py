"""
Final analyzer module for the /deepanalyze command.
This module compiles processed data into a briefing document and generates a final analysis using Gemini 2.5 Pro.
"""

import os
import asyncio
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for retry mechanisms
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5.0  # seconds
MAX_RETRY_DELAY = 60.0  # seconds

# Prompt template for Gemini 2.5 Pro
PROMPT_TEMPLATE = """
You are an expert sports analyst. Your task is to provide a final prediction and analysis for the event: {event_query}.

Use only the research brief compiled below. Do not use any outside knowledge.

Your final report should be concise and include:
- Overall Summary (2-3 sentences)
- Key Factors (3-4 bullets)
- Prediction (Team A Win / Draw / Team B Win)
- Confidence Score (Low/Medium/High)
- Brief Reasoning

Cite sources inline using [domain] tokens where relevant, and add a Sources section listing the URLs.
Avoid unescaped Markdown characters (use plain text bullets '-').

--- RESEARCH BRIEF ---
{compiled_briefing_document}
"""

class FinalAnalyzer:
    def __init__(self):
        """
        Initialize the FinalAnalyzer with the Gemini 2.5 Pro model.
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
        
        # Use Gemini 2.5 Pro model
        pro_model_id = os.getenv("GEMINI_MODEL_ID") or "gemini-2.5-pro"
        try:
            self.model = genai.GenerativeModel(
                pro_model_id,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    response_mime_type="text/plain"
                )
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Pro model: {e}")
            raise

    def _escape_markdown_v2(self, text: str) -> str:
        """
        Escape special characters for Telegram MarkdownV2.
        
        Args:
            text: The text to escape
            
        Returns:
            Escaped text
        """
        # Characters to escape in MarkdownV2: _ * [ ] ( ) ~ ` > # + - = | { } . !
        special_chars = r'[_*\[\]()~`>#+\-=|{}.!]'
        return re.sub(special_chars, r'\\\g<0>', text)

    def _format_markdown_v2(self, text: str) -> str:
        """
        Format text for Telegram MarkdownV2.
        
        Args:
            text: The text to format
            
        Returns:
            Formatted text
        """
        # Replace ** with * for bold in MarkdownV2
        text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
        
        # Ensure proper escaping
        text = self._escape_markdown_v2(text)
        
        # Unescape the formatting we want to keep
        text = re.sub(r'\\\*', r'*', text)
        
        return text

    def compile_briefing_document(self, processed_data: List[Dict[str, Any]]) -> str:
        """
        Compile the briefing document from processed data.
        
        Args:
            processed_data: List of processed data from content processor
            
        Returns:
            Compiled briefing document as a string
        """
        if not processed_data:
            return "No relevant data was found for this analysis."
        
        # Organize information by category
        summaries = []
        player_mentions = set()
        injuries_suspensions = set()
        statistics = []
        sources = set()
        
        # Process each data item
        for data in processed_data:
            # Extract summaries
            if "summary" in data and data["summary"]:
                summaries.append(data["summary"])
            
            # Extract player mentions
            if "key_players_mentioned" in data and isinstance(data["key_players_mentioned"], list):
                for player in data["key_players_mentioned"]:
                    if player:  # Only add non-empty player names
                        player_mentions.add(player)
            
            # Extract injuries/suspensions
            if "injuries_or_suspensions" in data and isinstance(data["injuries_or_suspensions"], list):
                for injury in data["injuries_or_suspensions"]:
                    if injury:  # Only add non-empty injury info
                        injuries_suspensions.add(injury)
            
            # Extract statistics
            if "statistics" in data and isinstance(data["statistics"], list):
                for stat in data["statistics"]:
                    if stat:  # Only add non-empty stats
                        statistics.append(str(stat))
            
            # Extract sources
            if "url" in data and data["url"]:
                sources.add(data["url"])
        
        # Compile the briefing document
        briefing_parts = []
        
        # Add summaries section
        if summaries:
            briefing_parts.append("SUMMARIES:")
            for i, summary in enumerate(summaries, 1):
                briefing_parts.append(f"{i}. {summary}")
            briefing_parts.append("")  # Empty line for spacing
        
        # Add player mentions section
        if player_mentions:
            briefing_parts.append("KEY PLAYER MENTIONS:")
            for player in sorted(player_mentions):
                briefing_parts.append(f"- {player}")
            briefing_parts.append("")  # Empty line for spacing
        
        # Add injuries/suspensions section
        if injuries_suspensions:
            briefing_parts.append("INJURIES/SUSPENSIONS:")
            for injury in sorted(injuries_suspensions):
                briefing_parts.append(f"- {injury}")
            briefing_parts.append("")  # Empty line for spacing
        
        # Add statistics section
        if statistics:
            briefing_parts.append("STATISTICS:")
            for stat in statistics:
                briefing_parts.append(f"- {stat}")
            briefing_parts.append("")  # Empty line for spacing
        
        # Add sources section
        if sources:
            briefing_parts.append("SOURCES:")
            for source in sorted(sources):
                briefing_parts.append(f"- {source}")
            briefing_parts.append("")  # Empty line for spacing
            # Add simple domain tokens for inline citation support
            domains = [re.sub(r'^www\.', '', re.sub(r'^https?://', '', s)).split('/')[0] for s in sources]
            if domains:
                briefing_parts.append("SOURCE_DOMAINS:")
                briefing_parts.append(", ".join(sorted(set(domains))))
                briefing_parts.append("")
        
        # If no data was added, provide a message
        if not briefing_parts:
            return "No relevant data was extracted from the sources."
        
        return "\n".join(briefing_parts)

    async def _call_gemini_api_with_retry(self, prompt: str) -> Optional[str]:
        """
        Call the Gemini API with retry mechanisms for rate limiting and other errors.
        
        Args:
            prompt: The prompt to send to the Gemini API
            
        Returns:
            Response text or None if failed
        """
        retries = 0
        delay = INITIAL_RETRY_DELAY
        
        while retries <= MAX_RETRIES:
            try:
                # Call the Gemini API
                response = await self.model.generate_content_async(prompt)
                
                # Extract text from response
                if hasattr(response, 'text') and response.text:
                    return response.text
                else:
                    # Try to extract from candidates
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'content') and candidate.content:
                                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text') and part.text:
                                            return part.text
                
                # If we get here, we couldn't extract text
                logger.warning("Empty response from Gemini API")
                return None
                
            except Exception as e:
                # Check if it's a rate limiting error (429)
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                    retries += 1
                    if retries > MAX_RETRIES:
                        logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries: {e}")
                        raise
                    else:
                        # Extract retry delay from error if available
                        retry_delay = self._extract_retry_delay(e)
                        if retry_delay:
                            delay = min(retry_delay, MAX_RETRY_DELAY)
                        
                        logger.warning(f"Rate limit hit, retrying in {delay} seconds: {e}")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, MAX_RETRY_DELAY)  # Exponential backoff
                else:
                    # For other errors, retry a few times but not as many
                    retries += 1
                    if retries > MAX_RETRIES // 2:  # Only retry 2-3 times for non-rate limit errors
                        logger.error(f"Error calling Gemini API after retries: {e}")
                        raise
                    else:
                        logger.warning(f"Error calling Gemini API, retrying in {delay} seconds: {e}")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, MAX_RETRY_DELAY)  # Exponential backoff

    def _extract_retry_delay(self, error: Exception) -> Optional[float]:
        """
        Extract retry delay from error message if available.
        
        Args:
            error: The exception object
            
        Returns:
            Retry delay in seconds or None if not found
        """
        error_str = str(error)
        # Look for retry_delay in the error message
        match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_str)
        if match:
            return float(match.group(1))
        return None

    async def get_final_analysis_with_pro(self, processed_data: List[Dict[str, Any]], event_query: str) -> str:
        """Generate a final analysis using Gemini 2.5 Pro."""
        try:
            compiled_briefing = self.compile_briefing_document(processed_data)
            prompt = PROMPT_TEMPLATE.format(
                event_query=event_query,
                compiled_briefing_document=compiled_briefing
            )
            response_text = await self._call_gemini_api_with_retry(prompt)
            if not response_text:
                return "Error: Failed to generate analysis. No response from the AI model."
            # Keep formatting safe for Telegram: reduce heavy Markdown
            formatted_response = self._format_markdown_v2(response_text)
            return formatted_response
        except Exception as e:
            logger.error(f"Error generating final analysis: {e}")
            return f"Error generating final analysis: {str(e)}"


# Convenience function for backward compatibility
async def get_final_analysis_with_pro(processed_data: List[Dict[str, Any]], event_query: str) -> str:
    """
    Generate a final analysis using Gemini 2.5 Pro.
    
    Args:
        processed_data: List of processed data from content processor
        event_query: The original event query
        
    Returns:
        Formatted final report as a string
    """
    analyzer = FinalAnalyzer()
    return await analyzer.get_final_analysis_with_pro(processed_data, event_query)