"""
Policies module for Flash Live Degen feature.
This module defines policies for domain whitelisting, rate limiting, and source scoring.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Reputable domains whitelist for sports coverage
REPUTABLE_DOMAINS = [
    "bbc.com", "espn.com", "skysports.com", "theguardian.com", "reuters.com", "apnews.com",
    "whoscored.com", "transfermarkt.com", "uefa.com", "fifa.com", "premierleague.com",
    "bundesliga.com", "laliga.com", "seriea.com", "as.com", "marca.com", "goal.com",
    "liverpoolfc.com", "arsenal.com", "manutd.com", "chelseafc.com", "fcbarcelona.com",
    "realmadrid.com", "bayern-muenchen.de", "acmilan.com", "juventus.com", "psg.fr",
    "nba.com", "nfl.com", "mlb.com", "nhl.com", "cbssports.com", "foxsports.com",
    "nbcsports.com", "si.com", "theathletic.com", "sports.yahoo.com", "olympics.com",
    "sportsillustrated.com", "bleacherreport.com", "sportscenter.com", "draftkings.com",
    "fanduel.com", "rotowire.com", "sportingnews.com", "talksport.com", "eurosport.com"
]

# Configuration values for caps and weights
MAX_PER_DOMAIN = 3
DOMAIN_BACKOFF_SECONDS = 10
LIVE_DOMAIN_WEIGHT = 20
LIVE_RECENCY_WEIGHT = 15
LIVE_MIN_REPUTABLE = 5

# Recency windows (in hours)
RECENCY_WINDOWS = {
    "excellent": 1,    # 1 hour
    "good": 3,         # 3 hours
    "fair": 6,         # 6 hours
    "poor": 12,        # 12 hours
    "very_poor": 24    # 24 hours
}


def is_domain_whitelisted(url: str) -> bool:
    """
    Check if a URL is from a whitelisted domain.
    
    Args:
        url: The URL to check
        
    Returns:
        True if the URL is from a whitelisted domain, False otherwise
    """
    try:
        host = urlparse(url).hostname or ''
        return any(host.endswith(domain) for domain in REPUTABLE_DOMAINS)
    except Exception as e:
        logger.error(f"Error checking domain whitelist for '{url}': {e}")
        return False


def get_domain_access_count(domain_access_counts: Dict[str, int], domain: str) -> int:
    """
    Get the number of times a domain has been accessed.
    
    Args:
        domain_access_counts: Dictionary tracking domain access counts
        domain: The domain to check
        
    Returns:
        The number of times the domain has been accessed
    """
    return domain_access_counts.get(domain, 0)


def can_access_domain(
    domain: str,
    domain_access_counts: Dict[str, int],
    domain_last_access: Dict[str, float]
) -> bool:
    """
    Check if we can access a domain based on rate limiting policies.
    
    Args:
        domain: The domain to check
        domain_access_counts: Dictionary tracking domain access counts
        domain_last_access: Dictionary tracking last access time for domains
        
    Returns:
        True if we can access the domain, False otherwise
    """
    try:
        # Check if we've exceeded the per-domain limit
        if get_domain_access_count(domain_access_counts, domain) >= MAX_PER_DOMAIN:
            logger.debug(f"Domain {domain} access limit reached")
            return False
        
        # Check if we need to back off
        last_access = domain_last_access.get(domain, 0)
        if time.time() - last_access < DOMAIN_BACKOFF_SECONDS:
            logger.debug(f"Domain {domain} backoff period not expired")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking domain access for '{domain}': {e}")
        return False


def update_domain_access_tracking(
    domain: str,
    domain_access_counts: Dict[str, int],
    domain_last_access: Dict[str, float]
) -> None:
    """
    Update domain access tracking information.
    
    Args:
        domain: The domain that was accessed
        domain_access_counts: Dictionary tracking domain access counts
        domain_last_access: Dictionary tracking last access time for domains
    """
    try:
        domain_access_counts[domain] = domain_access_counts.get(domain, 0) + 1
        domain_last_access[domain] = time.time()
        logger.debug(f"Updated access tracking for domain: {domain}")
    except Exception as e:
        logger.error(f"Error updating domain access tracking for '{domain}': {e}")


def calculate_domain_weight(url: str) -> float:
    """
    Calculate the weight of a source based on its domain reputation.
    
    Args:
        url: The URL of the source
        
    Returns:
        The domain weight (0.0 to LIVE_DOMAIN_WEIGHT)
    """
    try:
        if is_domain_whitelisted(url):
            return float(LIVE_DOMAIN_WEIGHT)
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating domain weight for '{url}': {e}")
        return 0.0


def calculate_recency_weight(published_time: Optional[str]) -> float:
    """
    Calculate the weight of a source based on its recency.
    
    Args:
        published_time: ISO8601 formatted timestamp string
        
    Returns:
        The recency weight (0.0 to LIVE_RECENCY_WEIGHT)
    """
    try:
        if not published_time:
            return 0.0
            
        # Parse the published time
        pub_datetime = datetime.fromisoformat(published_time.replace('Z', '+00:00'))
        
        # Calculate the time difference
        now = datetime.now(pub_datetime.tzinfo)
        time_diff = now - pub_datetime
        hours_diff = time_diff.total_seconds() / 3600
        
        # Assign weight based on recency windows
        if hours_diff <= RECENCY_WINDOWS["excellent"]:
            return float(LIVE_RECENCY_WEIGHT)
        elif hours_diff <= RECENCY_WINDOWS["good"]:
            return float(LIVE_RECENCY_WEIGHT) * 0.8
        elif hours_diff <= RECENCY_WINDOWS["fair"]:
            return float(LIVE_RECENCY_WEIGHT) * 0.6
        elif hours_diff <= RECENCY_WINDOWS["poor"]:
            return float(LIVE_RECENCY_WEIGHT) * 0.4
        elif hours_diff <= RECENCY_WINDOWS["very_poor"]:
            return float(LIVE_RECENCY_WEIGHT) * 0.2
        else:
            return 0.0
    except Exception as e:
        logger.error(f"Error calculating recency weight for '{published_time}': {e}")
        return 0.0


def calculate_source_score(
    url: str,
    published_time: Optional[str] = None
) -> float:
    """
    Calculate the overall score of a source based on domain reputation and recency.
    
    Args:
        url: The URL of the source
        published_time: ISO8601 formatted timestamp string (optional)
        
    Returns:
        The overall source score
    """
    try:
        domain_weight = calculate_domain_weight(url)
        recency_weight = calculate_recency_weight(published_time)
        
        # Calculate weighted score
        total_weight = LIVE_DOMAIN_WEIGHT + LIVE_RECENCY_WEIGHT
        if total_weight == 0:
            return 0.0
            
        score = (domain_weight + recency_weight) / total_weight * 100
        return min(score, 10.0)  # Cap at 100
    except Exception as e:
        logger.error(f"Error calculating source score for '{url}': {e}")
        return 0.0


def is_source_reputable(url: str) -> bool:
    """
    Check if a source is reputable based on domain whitelist.
    
    Args:
        url: The URL to check
        
    Returns:
        True if the source is from a reputable domain, False otherwise
    """
    return is_domain_whitelisted(url)


def get_reputable_domains() -> List[str]:
    """
    Get the list of reputable domains.
    
    Returns:
        List of reputable domains
    """
    return REPUTABLE_DOMAINS.copy()


def get_policies_config() -> Dict[str, Any]:
    """
    Get the current policies configuration.
    
    Returns:
        Dictionary with policies configuration
    """
    return {
        "MAX_PER_DOMAIN": MAX_PER_DOMAIN,
        "DOMAIN_BACKOFF_SECONDS": DOMAIN_BACKOFF_SECONDS,
        "LIVE_DOMAIN_WEIGHT": LIVE_DOMAIN_WEIGHT,
        "LIVE_RECENCY_WEIGHT": LIVE_RECENCY_WEIGHT,
        "LIVE_MIN_REPUTABLE": LIVE_MIN_REPUTABLE,
        "RECENCY_WINDOWS": RECENCY_WINDOWS.copy()
    }