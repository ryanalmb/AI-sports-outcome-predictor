"""
Flash Live session orchestrator for the /degenanalyze feature.
This module manages the live session state and orchestrates the real-time research process.
"""

import os
import asyncio
import logging
import time
import re
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
# Import tools
from .tools import ToolRegistry
from .tools.web_search import search_web
from .tools.url_fetch import fetch_url
from .tools.html_parse import parse_html
from .tools.html_to_text import html_to_text
from .tools.parse_published_time import parse_published_time
# Import policies
from . import policies
# Import renderers
from .renderers.stream_renderer import StreamRenderer
from .renderers.final_renderer import FinalRenderer

# Set up logging
logger = logging.getLogger(__name__)

# Environment variables for configuration
USE_FLASH_LIVE = os.getenv("USE_FLASH_LIVE", "0") == "1"
DEGEN_STREAM_INTERVAL_MS = int(os.getenv("DEGEN_STREAM_INTERVAL_MS", "1200"))
DDG_REGION = os.getenv("DDG_REGION", "us-en")
DDG_TIME = os.getenv("DDG_TIME", "w")

# Budget constants
LIVE_SESSION_TIMEOUT = int(os.getenv("LIVE_SESSION_TIMEOUT", "60"))  # seconds
MAX_PER_DOMAIN = int(os.getenv("LIVE_MAX_PER_DOMAIN", "3"))
DOMAIN_BACKOFF_SECONDS = int(os.getenv("LIVE_DOMAIN_BACKOFF_SECONDS", "10"))

# Scoring constants
LIVE_MIN_REPUTABLE = int(os.getenv("LIVE_MIN_REPUTABLE", "5"))
LIVE_DOMAIN_WEIGHT = int(os.getenv("LIVE_DOMAIN_WEIGHT", "20"))
LIVE_RECENCY_WEIGHT = int(os.getenv("LIVE_RECENCY_WEIGHT", "15"))

# Reputable domains whitelist
REPUTABLE_DOMAINS = [
    "bbc.com", "espn.com", "skysports.com", "theguardian.com", "reuters.com", "apnews.com",
    "whoscored.com", "transfermarkt.com", "uefa.com", "fifa.com", "premierleague.com",
    "bundesliga.com", "laliga.com", "seriea.com", "as.com", "marca.com", "goal.com",
    "liverpoolfc.com", "arsenal.com", "manutd.com", "chelseafc.com", "fcbarcelona.com",
    "realmadrid.com", "bayern-muenchen.de", "acmilan.com", "juventus.com", "psg.fr",
    "nba.com", "nfl.com", "mlb.com", "nhl.com", "cbssports.com", "foxsports.com",
    "nbcsports.com", "si.com", "theathletic.com", "sports.yahoo.com"
]


class LiveSession:
    """
    Flash Live session orchestrator for real-time sports research.
    """
    
    def __init__(self, event_query: str):
        """
        Initialize the LiveSession.
        
        Args:
            event_query: The sports event query (e.g., "Manchester United vs Liverpool")
        """
        self.event_query = event_query
        self.team_a, self.team_b = self._parse_teams(event_query)
        self.sport_type = self._detect_sport_type(event_query)
        
        # Session state
        self.found_sources: List[Dict[str, Any]] = []
        self.attempted_domains: Set[str] = set()
        self.current_best_lines: List[str] = []
        self.session_start_time = time.time()
        
        # Domain tracking for rate limiting
        self.domain_last_access: Dict[str, float] = {}
        self.domain_access_count: Dict[str, int] = {}
        
        # Flash model
        self.flash_model = None
        self._initialize_flash_model()
        # Tool registry
        self.tool_registry = ToolRegistry()
        self._register_tools()
        
        # Renderers
        self.stream_renderer = StreamRenderer()
        self.final_renderer = FinalRenderer()
        # Streaming state
        self.stream_queue: asyncio.Queue = asyncio.Queue()
        self.streaming_task: Optional[asyncio.Task] = None
        self.is_streaming = False
        
        logger.info(f"LiveSession initialized for '{event_query}'")
    
    def _parse_teams(self, event_query: str) -> tuple:
        """
        Parse teams/players from the event query.
        
        Args:
            event_query: The event query
            
        Returns:
            tuple: A tuple containing (team_a, team_b) or (event_query, "opponent") if no delimiter found
        """
        # Split on "vs" or "versus" (case insensitive)
        parts = re.split(r'\s+(?:vs|versus)\s+', event_query, flags=re.IGNORECASE)
        
        if len(parts) >= 2:
            # Take the first two parts as teams
            team_a = parts[0].strip()
            team_b = parts[1].split()[0].strip() if parts[1].split() else parts[1].strip()
            return team_a, team_b
        else:
            # Fallback mechanism if no "vs" or "versus" found
            # Try to split on common delimiters
            delimiters = ['-', '–', '&', 'and', '@']
            for delimiter in delimiters:
                parts = re.split(r'\s+' + re.escape(delimiter) + r'\s+', event_query, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    return parts[0].strip(), parts[1].strip()
            
            # If still no delimiter found, return the whole query as team_a and a generic term
            return event_query.strip(), "opponent"
    
    def _detect_sport_type(self, event_query: str) -> str:
        """
        Attempt to detect sport type from the event query.
        """
        sport_keywords = {
            'soccer': ['fc', 'united', 'city', 'afc', 'league', 'premier', 'championship', 'cup', 'real', 'barcelona', 'madrid'],
            'basketball': ['nba', 'basketball', 'lakers', 'warriors', 'celtics'],
            'tennis': ['atp', 'wta', 'tennis', 'open'],
            'american football': ['nfl', 'football'],
            'baseball': ['mlb', 'baseball'],
            'hockey': ['nhl', 'hockey', 'ice'],
            'golf': ['pga', 'golf', 'masters'],
        }
        query_lower = event_query.lower()
        for sport, keywords in sport_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return sport
        return "soccer"  # Default to soccer
    
    def _initialize_flash_model(self):
        """
        Initialize the Gemini Flash model with the API key from environment variables.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.5 Flash model
        flash_model_id = os.getenv("GEMINI_FLASH_MODEL_ID") or "gemini-2.5-flash"
        try:
            self.flash_model = genai.GenerativeModel(
                flash_model_id,
                generation_config=GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Flash model: {e}")
            raise
    def _register_tools(self):
        """
        Register tools with the tool registry.
        """
        # Register web search tool
        self.tool_registry.register("web_search", "Search the web using DuckDuckGo")(search_web)
        
        # Register URL fetching tool
        self.tool_registry.register("url_fetch", "Fetch content from a URL")(fetch_url)
        
        # Register HTML parsing tool
        self.tool_registry.register("html_parse", "Parse HTML content and extract text")(parse_html)
        
        # Register enhanced HTML to text tool
        self.tool_registry.register("html_to_text", "Parse HTML content and extract text with metadata")(html_to_text)
        
        # Register published time parsing tool
        self.tool_registry.register("parse_published_time", "Extract published time from HTML content or text")(parse_published_time)
        
        logger.info("Registered tools: %s", list(self.tool_registry.list_tools().keys()))
    
    async def start_session(self):
        """
        Start the live session and begin streaming updates.
        """
        if not USE_FLASH_LIVE:
            logger.warning("USE_FLASH_LIVE is disabled")
            return
        
        self.is_streaming = True
        self.streaming_task = asyncio.create_task(self._stream_research())
        logger.info("Live session started")
    
    async def stop_session(self):
        """
        Stop the live session and clean up resources.
        """
        self.is_streaming = False
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
        logger.info("Live session stopped")
    
    async def _stream_research(self):
        """
        Main research loop that streams updates.
        """
        try:
            # Generate initial search queries
            queries = self._generate_search_queries()
            logger.info(f"Generated {len(queries)} search queries")
            
            # Process queries in batches
            for query in queries:
                if not self.is_streaming:
                    break
                
                # Check if we've exceeded the session timeout
                if time.time() - self.session_start_time > LIVE_SESSION_TIMEOUT:
                    logger.info("Session timeout reached")
                    break
                
                # Process the query
                await self._process_query(query)
                
                # Respect the streaming interval
                await asyncio.sleep(DEGEN_STREAM_INTERVAL_MS / 1000.0)
            
            # Session complete
            await self._finalize_session()
            
        except Exception as e:
            logger.error(f"Error in live research stream: {e}")
            error_message = self.stream_renderer.format_update({
                "type": "error",
                "message": f"Research error: {str(e)}"
            })
            await self.stream_queue.put({
                "type": "error",
                "message": error_message
            })
        finally:
            end_message = self.stream_renderer.format_update({
                "type": "session_end",
                "message": "Research session completed"
            })
            await self.stream_queue.put({
                "type": "session_end",
                "message": end_message
            })
    
    def _generate_search_queries(self) -> List[str]:
        """
        Generate targeted search queries for the event.
        
        Returns:
            List of search query strings
        """
        # Base queries for the teams
        base_queries = [
            f"{self.team_a} vs {self.team_b} match preview",
            f"{self.team_a} starting XI vs {self.team_b}",
            f"{self.team_b} manager tactical analysis",
            f"{self.team_a} vs {self.team_b} injury update",
            f"{self.team_a} vs {self.team_b} press conference"
        ]
        
        # Sport-specific queries
        sport_queries = self._generate_sport_specific_queries()
        
        # Coverage queries to ensure balanced mix
        coverage_queries = self._build_coverage_queries()
        
        # Combine and deduplicate
        all_queries = base_queries + sport_queries + coverage_queries
        return list(dict.fromkeys(all_queries))[:20]  # Limit to 20 unique queries
    
    def _generate_sport_specific_queries(self) -> List[str]:
        """
        Generate sport-specific queries based on the sport type.
        
        Returns:
            List of sport-specific query strings
        """
        soccer_sites = [
            "bbc.com", "espn.com", "skysports.com", "theguardian.com",
            "whoscored.com", "transfermarkt.com", "uefa.com", "premierleague.com"
        ]
        
        def _site_hint(q: str, sites: List[str]) -> List[str]:
            # mix site: filters into some variants
            hinted = [q]
            for s in sites[:2]:  # at most 2 site hints to keep diversity
                hinted.append(f"{q} site:{s}")
            return hinted
        
        if self.sport_type == "soccer":
            base = [
                f"{self.team_a} vs {self.team_b} match preview",
                f"{self.team_a} starting XI vs {self.team_b}",
                f"{self.team_b} manager tactical analysis",
                f"{self.team_a} vs {self.team_b} injury update",
                f"{self.team_a} vs {self.team_b} press conference"
            ]
            queries = []
            for q in base:
                queries.extend(_site_hint(q, soccer_sites))
            return queries
        elif self.sport_type == "basketball":
            return [
                f"{self.team_a} vs {self.team_b} season stats comparison",
                f"{self.team_a} star player performance vs {self.team_b}",
                f"{self.team_b} injury report and depth chart"
            ]
        elif self.sport_type == "tennis":
            return [
                f"{self.team_a} vs {self.team_b} head to head statistics",
                f"{self.team_a} recent tournament form",
                f"{self.team_b} surface preference analysis"
            ]
        elif self.sport_type == "american football":
            return [
                f"{self.team_a} vs {self.team_b} season record",
                f"{self.team_a} offensive strategy vs {self.team_b} defense",
                f"{self.team_b} quarterback performance analysis"
            ]
        elif self.sport_type == "baseball":
            return [
                f"{self.team_a} vs {self.team_b} pitching matchup",
                f"{self.team_a} batting order analysis",
                f"{self.team_b} bullpen situation report"
            ]
        elif self.sport_type == "hockey":
            return [
                f"{self.team_a} vs {self.team_b} power play statistics",
                f"{self.team_a} goaltender matchup analysis",
                f"{self.team_b} injury report and lineup news"
            ]
        elif self.sport_type == "golf":
            return [
                f"{self.team_a} vs {self.team_b} course history",
                f"{self.team_a} recent tournament results",
                f"{self.team_b} putting statistics this season"
            ]
        else:
            return [
                f"{self.team_a} vs {self.team_b} latest news",
                f"{self.team_a} key players to watch vs {self.team_b}",
                f"{self.team_b} tactical analysis and strategy",
            ]
    
    def _build_coverage_queries(self) -> List[str]:
        """
        Build coverage queries to guarantee a balanced mix.
        
        Returns:
            List of coverage query strings
        """
        league_terms = ["league standings", "fixtures congestion", "disciplinary rules", "suspension rules", "transfer window impact"]
        team_terms = ["form", "selection", "injury update", "suspension", "tactical setup", "press conference", "probable lineup"]
        odds_terms = ["odds", "betting lines", "price movement", "bookmaker"]
        
        queries: List[str] = []
        # Team-specific coverage
        for t in team_terms:
            queries.append(f"{self.team_a} {t}")
            queries.append(f"{self.team_b} {t}")
        # League/competition context
        for lt in league_terms:
            queries.append(f"{self.team_a} {lt}")
            queries.append(f"{self.team_b} {lt}")
        # Odds/bookmaker context
        for ot in odds_terms:
            queries.append(f"{self.team_a} vs {self.team_b} {ot}")
            queries.append(f"{self.team_a} {ot}")
            queries.append(f"{self.team_b} {ot}")
        # Official communications
        queries.append(f"{self.team_a} official site announcements")
        queries.append(f"{self.team_b} official site announcements")
        
        # Normalize and dedupe
        queries = [q.strip() for q in queries if q.strip()]
        return list(dict.fromkeys(queries))[:15]  # Limit to 15 unique queries
    
    async def _process_query(self, query: str):
        """
        Process a single search query.
        
        Args:
            query: The search query to process
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Search for results (simulated for now)
            search_results = await self._search_duckduckgo(query)
            
            # Process each result
            for result in search_results:
                if not self.is_streaming:
                    break
                
                url = result.get('url')
                if not url:
                    continue
                
                # Check domain rate limiting
                domain = urlparse(url).hostname or ''
                if not self._can_access_domain(domain):
                    continue
                
                # Update domain access tracking
                self._update_domain_access(domain)
                
                # Check if we've already processed this URL
                if any(s.get('url') == url for s in self.found_sources):
                    continue
                
                # Fetch and process the content
                content = await self._fetch_url_content(url)
                if content:
                    # Parse HTML content with metadata
                    parsed_data = self._parse_html_with_metadata(content)
                    parsed_content = parsed_data.get('text', '')
                    metadata = parsed_data.get('metadata', {})
                    # Extract published time
                    published_time = self._extract_published_time(content)
                    # Process with Flash model
                    processed_data = await self._process_with_flash(parsed_content, url, published_time, metadata)
                    if processed_data:
                        # Add to found sources
                        source_data = {
                            "url": url,
                            "title": result.get('title', ''),
                            "published_time": published_time,
                            "metadata": metadata,
                            "processed_data": processed_data
                        }
                        self.found_sources.append(source_data)
                        
                        # Stream update
                        update_message = self.stream_renderer.format_update({
                            "type": "update",
                            "message": f"Found fresh data on {domain}",
                            "source": source_data
                        })
                        await self.stream_queue.put({
                            "type": "update",
                            "message": update_message,
                            "source": source_data
                        })
        
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
    
    async def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """
        Search DuckDuckGo for a query and return top results.
        
        Args:
            query: Search query string
            
        Returns:
            List of search results
        """
        try:
            results = self.tool_registry.call_tool(
                "web_search",
                query=query,
                region=DDG_REGION,
                timelimit=DDG_TIME,
                max_results=5
            )
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo for '{query}': {e}")
            return []
    
    def _can_access_domain(self, domain: str) -> bool:
        """
        Check if we can access a domain based on rate limiting rules.
        
        Args:
            domain: The domain to check
            
        Returns:
            True if we can access the domain, False otherwise
        """
        return policies.can_access_domain(domain, self.domain_access_count, self.domain_last_access)
    
    def _update_domain_access(self, domain: str):
        """
        Update domain access tracking.
        
        Args:
            domain: The domain that was accessed
        """
        policies.update_domain_access_tracking(domain, self.domain_access_count, self.domain_last_access)
        self.attempted_domains.add(domain)
    
    async def _fetch_url_content(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL.
        
        Args:
            url: The URL to fetch
            
        Returns:
            The content as a string, or None if failed
        """
        try:
            response = await self.tool_registry.call_tool("url_fetch", url=url)
            if response and 'content' in response:
                content = response['content']
                logger.info(f"Successfully fetched content from {url}")
                return content
            else:
                logger.warning(f"No content returned from {url}")
                return None
        except Exception as e:
            logger.error(f"Error fetching content from '{url}': {e}")
            return None
    
    def _parse_html_content(self, html_content: str) -> str:
        """
        Parse HTML content and extract text.
        
        Args:
            html_content: The HTML content to parse
            
        Returns:
            The extracted text content
        """
        try:
            parsed_content = self.tool_registry.call_tool("html_parse", html_content=html_content)
            return parsed_content
        except Exception as e:
            logger.error(f"Error parsing HTML content: {e}")
            return html_content  # Return original content if parsing fails
    
    def _parse_html_with_metadata(self, html_content: str) -> Dict[str, Any]:
        """
        Parse HTML content and extract text with metadata.
        
        Args:
            html_content: The HTML content to parse
            
        Returns:
            Dictionary containing text and metadata
        """
        try:
            parsed_data = self.tool_registry.call_tool("html_to_text", html_content=html_content)
            return parsed_data
        except Exception as e:
            logger.error(f"Error parsing HTML content with metadata: {e}")
            # Fallback to basic parsing
            basic_text = self._parse_html_content(html_content)
            return {
                'text': basic_text,
                'metadata': {}
            }
    
    def _extract_published_time(self, content: str) -> Optional[str]:
        """
        Extract published time from content.
        
        Args:
            content: The content to extract published time from
            
        Returns:
            ISO8601 formatted timestamp string if found, None otherwise
        """
        try:
            published_time = self.tool_registry.call_tool("parse_published_time", content=content)
            if published_time:
                logger.info(f"Extracted published time: {published_time}")
            return published_time
        except Exception as e:
            logger.error(f"Error extracting published time: {e}")
            return None
    
    async def _process_with_flash(self, content: str, url: str, published_time: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Process content with the Flash model.
        
        Args:
            content: The content to process
            url: The source URL
            published_time: The published time of the content (ISO8601 format)
            metadata: Metadata extracted from the HTML content
            
        Returns:
            Processed data as a dictionary, or None if not relevant
        """
        try:
            # Create a prompt for the Flash model
            prompt = f"""
            You are a sports research agent. Extract key information from the article below.
            
            URL: {url}
            Published Time: {published_time or "Unknown"}
            
            Metadata:
            Title: {metadata.get('title', 'N/A') if metadata else 'N/A'}
            Description: {metadata.get('description', 'N/A') if metadata else 'N/A'}
            Word Count: {metadata.get('word_count', 'N/A') if metadata else 'N/A'}
            Headings: {', '.join(metadata.get('headings', [])[:5]) if metadata else 'N/A'}
            
            Article content:
            {content[:2000]}  # Limit content length
            
            Extract and return ONLY valid JSON with this schema:
            {{
                "summary": "Brief summary of the article (1-2 sentences)",
                "key_points": ["List of key points (3-5 items)"],
                "injuries_suspensions": ["List of any mentioned injuries or suspensions"],
                "odds_movement": "Any mentioned odds or betting movement",
                "confidence": 0-100 (how confident you are in the information)
            }}
            """
            
            # Call the Flash model
            response = await self.flash_model.generate_content_async(prompt)
            
            # Extract text from response
            response_text = None
            if hasattr(response, 'text') and response.text:
                response_text = response.text
            else:
                # Try to extract from candidates
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
                # Clean up the response text to extract JSON
                import json
                import re
                
                # Try to find JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        # Parse the JSON
                        parsed_data = json.loads(json_str)
                        
                        # Validate required fields
                        required_fields = ["summary", "key_points", "injuries_suspensions", "odds_movement", "confidence"]
                        if all(field in parsed_data for field in required_fields):
                            return parsed_data
                        else:
                            logger.warning(f"Flash model response missing required fields: {parsed_data}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON from Flash model response: {e}")
                else:
                    logger.warning(f"No JSON found in Flash model response: {response_text}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing content with Flash model: {e}")
            return None
    
    async def _finalize_session(self):
        """
        Finalize the session and prepare the final analysis.
        """
        logger.info(f"Session finalized with {len(self.found_sources)} sources found")
        
        # Stream final summary
        final_summary_message = self.stream_renderer.format_update({
            "type": "final_summary",
            "message": f"Research complete! Found {len(self.found_sources)} relevant sources.",
            "sources": self.found_sources
        })
        await self.stream_queue.put({
            "type": "final_summary",
            "message": final_summary_message,
            "sources": self.found_sources
        })
    
    def is_whitelisted(self, url: str) -> bool:
        """
        Check if a URL is from a whitelisted domain.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is from a whitelisted domain, False otherwise
        """
        return policies.is_domain_whitelisted(url)
    
    def get_session_state(self) -> Dict[str, Any]:
        """
        Get the current session state.
        
        Returns:
            Dictionary with session state information
        """
        return {
            "event_query": self.event_query,
            "team_a": self.team_a,
            "team_b": self.team_b,
            "sport_type": self.sport_type,
            "found_sources_count": len(self.found_sources),
            "attempted_domains_count": len(self.attempted_domains),
            "session_duration": time.time() - self.session_start_time,
            "is_streaming": self.is_streaming
        }
    
    def generate_final_report(self) -> str:
        """
        Generate the final degen-toned report using the final renderer.
        
        Returns:
            Formatted string for Telegram
        """
        analysis_data = {
            "event_query": self.event_query,
            "sources": self.found_sources
        }
        return self.final_renderer.format_final_report(analysis_data)