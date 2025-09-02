"""
Web search tool for Flash Live Degen feature.
This module provides a tool for searching the web using DuckDuckGo.
"""

import logging
from typing import List, Dict, Any
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


def search_web(query: str, region: str = "us-en", timelimit: str = "w", max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: The search query
        region: The region to search in (default: "us-en")
        timelimit: The time limit for results (default: "w" for week)
        max_results: The maximum number of results to return (default: 5)
        
    Returns:
        A list of search results with 'url', 'title', and 'snippet' keys
    """
    items: List[Dict[str, Any]] = []
    
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query,
                region=region,
                timelimit=timelimit,
                max_results=max_results
            )
            
            # Process results to ensure consistent structure
            for r in results:
                href = r.get('href') if isinstance(r, dict) else r
                if not href:
                    continue
                items.append({
                    'url': href,
                    'title': r.get('title', '') if isinstance(r, dict) else '',
                    'snippet': r.get('body', '') if isinstance(r, dict) else ''
                })
        
        logger.info(f"Found {len(items)} results for query: {query}")
        return items
    except Exception as e:
        logger.error(f"Error searching web for '{query}': {e}")
        return []