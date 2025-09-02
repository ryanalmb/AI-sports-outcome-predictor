"""
HTML parsing tool for Flash Live Degen feature.
This module provides a tool for parsing HTML content and extracting text.
"""

import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def parse_html(html_content: str) -> str:
    """
    Parse HTML content and extract text.
    
    Args:
        html_content: The HTML content to parse
        
    Returns:
        The extracted text content
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        logger.error(f"Error parsing HTML content: {e}")
        return html_content  # Return original content if parsing fails