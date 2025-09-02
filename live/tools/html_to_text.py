"""
HTML to text parsing tool for Flash Live Degen feature.
This module provides a tool for parsing HTML content and extracting both clean text and metadata.
"""

import logging
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def html_to_text(html_content: str) -> Dict[str, Any]:
    """
    Parse HTML content and extract both clean text and metadata.
    
    Args:
        html_content: The HTML content to parse
        
    Returns:
        A dictionary containing:
            - text: Clean text extracted from HTML
            - metadata: Dictionary with metadata including:
                - title: Page title
                - description: Meta description
                - headings: List of heading texts (h1-h6)
                - word_count: Number of words in the text
                - links: List of links found in the content
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract metadata
        metadata = {}
        
        # Get page title
        title_tag = soup.find('title')
        metadata['title'] = title_tag.get_text().strip() if title_tag else ""
        
        # Get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        metadata['description'] = meta_desc.get('content', '').strip() if meta_desc else ""
        
        # Get headings (h1-h6)
        headings = []
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                heading_text = heading.get_text().strip()
                if heading_text:
                    headings.append(heading_text)
        metadata['headings'] = headings
        
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
        
        # Get word count
        metadata['word_count'] = len(text.split())
        
        # Get links
        links = []
        for link in soup.find_all('a', href=True):
            link_text = link.get_text().strip()
            if link_text:
                links.append({
                    'text': link_text,
                    'url': link['href']
                })
        metadata['links'] = links
        
        logger.info(f"Successfully parsed HTML content. Extracted {metadata['word_count']} words.")
        return {
            'text': text,
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Error parsing HTML content: {e}")
        # Return a fallback result with the original content
        return {
            'text': html_content,
            'metadata': {
                'title': '',
                'description': '',
                'headings': [],
                'word_count': len(html_content.split()),
                'links': []
            }
        }