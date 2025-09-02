"""
Published time parsing tool for Flash Live Degen feature.
This module provides a tool for extracting published time from HTML content or text.
"""

import logging
import re
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
from datetime import datetime
import dateutil.parser
import dateutil.tz

logger = logging.getLogger(__name__)


def parse_published_time(content: str) -> Optional[str]:
    """
    Extract published time from HTML content or text and return ISO8601 formatted timestamp.
    
    Args:
        content: The HTML content or text to parse
        
    Returns:
        ISO8601 formatted timestamp string if found, None otherwise
    """
    try:
        # Try to parse as HTML first
        soup = BeautifulSoup(content, 'html.parser')
        
        # Look for time in meta tags
        time_value = _extract_from_meta_tags(soup)
        if time_value:
            return time_value
        
        # Look for time in structured data (JSON-LD)
        time_value = _extract_from_structured_data(soup)
        if time_value:
            return time_value
        
        # Look for time in time tags
        time_value = _extract_from_time_tags(soup)
        if time_value:
            return time_value
        
        # Look for time in common text patterns
        time_value = _extract_from_text_patterns(content)
        if time_value:
            return time_value
            
        logger.info("No published time found in content")
        return None
        
    except Exception as e:
        logger.error(f"Error parsing published time: {e}")
        return None


def _extract_from_meta_tags(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract time from meta tags.
    
    Args:
        soup: BeautifulSoup object of the HTML content
        
    Returns:
        ISO8601 formatted timestamp string if found, None otherwise
    """
    # Check for article:published_time meta tag
    meta_tag = soup.find('meta', attrs={'property': 'article:published_time'})
    if meta_tag and meta_tag.get('content'):
        return _normalize_datetime(meta_tag['content'])
    
    # Check for article:modified_time meta tag
    meta_tag = soup.find('meta', attrs={'property': 'article:modified_time'})
    if meta_tag and meta_tag.get('content'):
        return _normalize_datetime(meta_tag['content'])
    
    # Check for og:article:published_time meta tag
    meta_tag = soup.find('meta', attrs={'property': 'og:article:published_time'})
    if meta_tag and meta_tag.get('content'):
        return _normalize_datetime(meta_tag['content'])
    
    # Check for date meta tags
    date_meta_tags = [
        'date', 'pubdate', 'published', 'article:published',
        'article:date', 'article:publication-date'
    ]
    
    for prop in date_meta_tags:
        meta_tag = soup.find('meta', attrs={'property': prop})
        if not meta_tag:
            meta_tag = soup.find('meta', attrs={'name': prop})
        if meta_tag and meta_tag.get('content'):
            parsed_time = _normalize_datetime(meta_tag['content'])
            if parsed_time:
                return parsed_time
    
    return None


def _extract_from_structured_data(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract time from structured data (JSON-LD).
    
    Args:
        soup: BeautifulSoup object of the HTML content
        
    Returns:
        ISO8601 formatted timestamp string if found, None otherwise
    """
    try:
        # Look for JSON-LD structured data
        scripts = soup.find_all('script', attrs={'type': 'application/ld+json'})
        for script in scripts:
            if script.string:
                import json
                # Try to parse JSON
                try:
                    data = json.loads(script.string)
                    # Handle both single objects and arrays
                    if isinstance(data, list):
                        items = data
                    else:
                        items = [data]
                    
                    for item in items:
                        # Look for datePublished or dateModified
                        if isinstance(item, dict):
                            date_fields = ['datePublished', 'dateModified', 'dateCreated']
                            for field in date_fields:
                                if field in item and item[field]:
                                    parsed_time = _normalize_datetime(item[field])
                                    if parsed_time:
                                        return parsed_time
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.debug(f"Error parsing structured data: {e}")
    
    return None


def _extract_from_time_tags(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract time from time tags.
    
    Args:
        soup: BeautifulSoup object of the HTML content
        
    Returns:
        ISO8601 formatted timestamp string if found, None otherwise
    """
    # Look for time tags with datetime attribute
    time_tags = soup.find_all('time')
    for time_tag in time_tags:
        if time_tag.get('datetime'):
            parsed_time = _normalize_datetime(time_tag['datetime'])
            if parsed_time:
                return parsed_time
        elif time_tag.get('pubdate'):
            # Try to parse the text content
            text = time_tag.get_text().strip()
            if text:
                parsed_time = _normalize_datetime(text)
                if parsed_time:
                    return parsed_time
    
    return None


def _extract_from_text_patterns(content: str) -> Optional[str]:
    """
    Extract time from common text patterns.
    
    Args:
        content: The text content to search
        
    Returns:
        ISO8601 formatted timestamp string if found, None otherwise
    """
    # Common date/time patterns
    patterns = [
        r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)',  # ISO8601
        r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?)',  # ISO-like without timezone
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)',  # MM/DD/YYYY HH:MM AM/PM
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)',  # MM/DD/YYYY HH:MM
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)',  # YYYY/MM/DD HH:MM
        r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)',  # Month DD, YYYY HH:MM AM/PM
        r'([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})',  # Month DD, YYYY
        r'(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)',  # DD Month YYYY HH:MM
        r'(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})',  # DD Month YYYY
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            parsed_time = _normalize_datetime(match)
            if parsed_time:
                return parsed_time
    
    return None


def _normalize_datetime(time_str: str) -> Optional[str]:
    """
    Normalize a datetime string to ISO8601 format.
    
    Args:
        time_str: The datetime string to normalize
        
    Returns:
        ISO8601 formatted timestamp string if valid, None otherwise
    """
    try:
        # Parse the datetime string
        dt = dateutil.parser.parse(time_str)
        
        # If no timezone info, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=dateutil.tz.UTC)
        
        # Return in ISO8601 format
        return dt.isoformat()
    except Exception as e:
        logger.debug(f"Error normalizing datetime '{time_str}': {e}")
        return None