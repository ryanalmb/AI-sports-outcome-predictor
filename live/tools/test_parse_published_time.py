"""
Test script for the parse_published_time tool.
This script tests the parse_published_time function with various inputs.
"""

import sys
import os

# Add the parent directory to the path so we can import the tool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from live.tools.parse_published_time import parse_published_time


def test_parse_published_time():
    """Test the parse_published_time function with various inputs."""
    
    # Test case 1: HTML with meta tags
    html_with_meta = '''
    <html>
    <head>
        <meta property="article:published_time" content="2023-10-15T14:30:00Z">
        <title>Test Article</title>
    </head>
    <body>
        <h1>Test Article</h1>
        <p>This is a test article.</p>
    </body>
    </html>
    '''
    
    result = parse_published_time(html_with_meta)
    print(f"Test 1 - HTML with meta tags: {result}")
    assert result == "2023-10-15T14:30:00+00:00", f"Expected '2023-10-15T14:30:00+00:0', got {result}"
    
    # Test case 2: HTML with time tag
    html_with_time = '''
    <html>
    <body>
        <time datetime="2023-10-15T14:30:00Z">October 15, 2023</time>
        <p>This is a test article.</p>
    </body>
    </html>
    '''
    
    result = parse_published_time(html_with_time)
    print(f"Test 2 - HTML with time tag: {result}")
    assert result == "2023-10-15T14:30:00+00:00", f"Expected '2023-10-15T14:30:00+00:00', got {result}"
    
    # Test case 3: Plain text with date
    text_with_date = "Published on October 15, 2023 at 2:30 PM"
    
    result = parse_published_time(text_with_date)
    print(f"Test 3 - Plain text with date: {result}")
    # This might not match exactly due to timezone assumptions, but should not be None
    assert result is not None, f"Expected a valid date, got None"
    
    # Test case 4: HTML with JSON-LD structured data
    html_with_jsonld = '''
    <html>
    <head>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "NewsArticle",
            "datePublished": "2023-10-15T14:30:00Z"
        }
        </script>
    </head>
    <body>
        <h1>Test Article</h1>
        <p>This is a test article.</p>
    </body>
    </html>
    '''
    
    result = parse_published_time(html_with_jsonld)
    print(f"Test 4 - HTML with JSON-LD: {result}")
    assert result == "2023-10-15T14:30:00+00:00", f"Expected '2023-10-15T14:30:00+00:00', got {result}"
    
    # Test case 5: No date information
    no_date_html = '''
    <html>
    <body>
        <h1>Test Article</h1>
        <p>This is a test article with no date information.</p>
    </body>
    </html>
    '''
    
    result = parse_published_time(no_date_html)
    print(f"Test 5 - No date information: {result}")
    assert result is None, f"Expected None, got {result}"
    
    print("All tests passed!")


if __name__ == "__main__":
    test_parse_published_time()