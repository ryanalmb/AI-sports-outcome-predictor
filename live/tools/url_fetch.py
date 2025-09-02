"""
URL fetching tool for Flash Live Degen feature.
This module provides a tool for fetching content from URLs with detailed response information.
"""

import logging
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from aiohttp import ClientError, ClientTimeout

logger = logging.getLogger(__name__)


async def fetch_url(url: str) -> Optional[Dict[str, Any]]:
    """
    Fetch content from a URL with detailed response information.
    
    Args:
        url: The URL to fetch
        
    Returns:
        A dictionary containing:
            - status: HTTP status code
            - headers: Response headers
            - content: HTML content as string
        Returns None if failed
    """
    # Use environment variables for configuration if available, with sensible defaults
    import os
    timeout_seconds = float(os.getenv('LIVE_FETCH_TIMEOUT', '30.0'))
    max_retries = int(os.getenv('LIVE_FETCH_MAX_RETRIES', '3'))
    initial_retry_delay = float(os.getenv('LIVE_FETCH_INITIAL_RETRY_DELAY', '1.0'))
    
    timeout = ClientTimeout(total=timeout_seconds)
    retries = 0
    
    while retries <= max_retries:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    content = await response.text(errors='ignore')
                    
                    logger.info(f"Successfully fetched content from {url} with status {response.status}")
                    return {
                        'status': response.status,
                        'headers': dict(response.headers),
                        'content': content
                    }
        
        except asyncio.CancelledError:
            # Treat cancellation like a timeout; do not bubble up
            retries += 1
            if retries > max_retries:
                logger.error(f"Cancelled while fetching {url} after retries")
                return None
            await asyncio.sleep(initial_retry_delay * (2 ** (retries - 1)))
            continue
            
        except asyncio.TimeoutError:
            retries += 1
            if retries > max_retries:
                logger.error(f"Timeout fetching {url} after {max_retries} retries")
                return None
            backoff = initial_retry_delay * (2 ** (retries - 1))
            logger.warning(f"Timeout fetching {url}, retrying in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            continue
            
        except ClientError as e:
            logger.warning(f"Network error fetching {url}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None
    
    return None