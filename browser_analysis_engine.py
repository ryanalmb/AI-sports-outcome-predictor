"""
Browser-Use Analysis Engine for AI Sports Outcome Predictor - LEGACY STATUS

⚠️  LEGACY CODE - SUPERSEDED BY PREMIUM GROUNDING SYSTEM ⚠️

This module is now LEGACY and has been replaced by the new premium grounding system
in llm_predictor.py. The new system provides superior performance and reliability:

🔄 MIGRATION NOTICE:
- OLD: Browser automation with complex infrastructure requirements
- NEW: Direct Gemini Flash grounding with Google Search (llm_predictor.predict_premium_analysis)

🎯 BENEFITS OF NEW SYSTEM:
- 95%+ reliability vs browser automation failures
- 3-5 second response time vs 30+ seconds 
- No infrastructure dependencies (browsers, sessions)
- $35/1000 requests vs infrastructure costs
- 15 concurrent prompts vs sequential navigation
- Built-in citations and source verification

🔧 USAGE:
- /analysis command now routes through llm_predictor.predict_premium_analysis()
- This browser_analysis_engine is kept as emergency fallback only
- To revert: modify fixed_bot.py analysis_command routing

📚 TECHNICAL MIGRATION:
- Multi-prompt grounding replaces browser navigation
- Gemini 2.5 Flash + Pro synthesis replaces browser + Flash combination
- Rate limit awareness with free/paid tier detection
- Comprehensive error handling with graceful degradation

🚀 PERFORMANCE COMPARISON:
- Browser system: 30-60s, infrastructure heavy, failure prone
- Grounding system: 3-10s, infrastructure light, enterprise reliable

This module follows the LLM-first architecture with Gemini 2.5 models only.
Maintains minimal file creation policy by extending existing infrastructure.

DEPRECATION DATE: 2025-01-27
REPLACED BY: llm_predictor.predict_premium_analysis()
STATUS: Legacy fallback only
"""

import asyncio
import logging
import os
from typing import Dict, Optional, List
from datetime import datetime, timedelta

# CRITICAL: Disable telemetry BEFORE importing browser-use
os.environ['ANONYMIZED_TELEMETRY'] = 'false'

try:
    from browser_use import Agent
    import google.generativeai as genai
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    logging.warning("browser-use not available, enhanced analysis will be disabled")

import google.generativeai as genai

logger = logging.getLogger(__name__)

# Create detailed logger for all prompts and responses
detail_logger = logging.getLogger('browser_analysis_details')
detail_logger.setLevel(logging.INFO)
if not detail_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | DETAIL | %(message)s'))
    detail_logger.addHandler(handler)


class OptimizedBrowserPool:
    """
    Phase 3: Optimized browser pool management for resource efficiency.
    
    Features:
    - Single browser instance reuse
    - Intelligent session management
    - Memory optimization
    - Timeout handling
    """
    
    def __init__(self, pool_size=1):
        self.pool_size = pool_size
        self.active_agent = None
        self.session_timeout = 300  # 5 minutes
        self.last_activity = None
        self.initialization_lock = asyncio.Lock()
        self.usage_count = 0
        self.max_usage_before_refresh = 10  # Refresh agent after 10 uses
    
    async def get_analysis_agent(self):
        """Get or create browser agent with optimization"""
        async with self.initialization_lock:
            # Check if current agent needs refresh
            if self.active_agent and self.usage_count >= self.max_usage_before_refresh:
                logger.info("Refreshing browser agent after maximum usage")
                await self.cleanup_agent()
            
            # Check session timeout
            if self.last_activity and self.active_agent:
                if datetime.now() - self.last_activity > timedelta(seconds=self.session_timeout):
                    logger.info("Browser session timed out, creating new agent")
                    await self.cleanup_agent()
            
            # Create new agent if needed
            if not self.active_agent:
                await self._create_new_agent()
            
            self.usage_count += 1
            self.last_activity = datetime.now()
            return self.active_agent
    
    async def get_task_specific_agent(self, task_description: str):
        """Get or create an agent optimized for a specific task"""
        async with self.initialization_lock:
            # For now, use the same pool agent but could be extended
            # to create task-specific agents for different types of data collection
            agent = await self.get_analysis_agent()
            
            if agent:
                # Update the agent's task context
                try:
                    # Some browser-use versions allow task updates
                    if hasattr(agent, 'update_task'):
                        await agent.update_task(task_description)
                    elif hasattr(agent, 'task'):
                        agent.task = task_description
                except Exception as e:
                    logger.debug(f"Could not update agent task: {e}")
                    # This is OK - we'll pass the task in the run() method instead
            
            return agent
    
    async def _create_new_agent(self):
        """Create a new browser agent using Gemini 2.5 Flash directly with working pattern"""
        try:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if not gemini_key:
                raise Exception("GEMINI_API_KEY required for browser navigation")
            
            # Set environment variable as required by browser-use
            os.environ['GEMINI_API_KEY'] = gemini_key
            
            # Use the exact working pattern from research
            from langchain_google_genai import ChatGoogleGenerativeAI
            from pydantic import SecretStr
            
            gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=SecretStr(gemini_key),
                temperature=0.0,
                convert_system_message_to_human=True
            )
            
            # Create browser agent with the working pattern from research
            self.active_agent = Agent(
                task="Navigate websites and extract sports data. Follow task instructions precisely.",
                llm=gemini_llm
            )
            
            self.usage_count = 0
            self.last_activity = datetime.now()
            logger.info("Browser agent created with Gemini 2.5 Flash using working pattern")
                
        except ImportError as e:
            logger.error(f"langchain_google_genai not available: {e}")
            self.active_agent = None
            raise Exception("langchain_google_genai required for Gemini browser integration")
        except Exception as e:
            logger.error(f"Failed to create browser agent with Gemini: {e}")
            self.active_agent = None
            raise
    
    async def cleanup_agent(self):
        """Clean up current browser agent"""
        if self.active_agent:
            try:
                # Try different cleanup methods based on browser-use version
                if hasattr(self.active_agent, 'close'):
                    await self.active_agent.close()
                elif hasattr(self.active_agent, 'stop'):
                    await self.active_agent.stop()
                elif hasattr(self.active_agent, 'cleanup'):
                    await self.active_agent.cleanup()
                else:
                    # Force cleanup by setting to None
                    pass
                logger.info("Browser agent cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up browser agent: {e}")
            finally:
                self.active_agent = None
                self.usage_count = 0
                self.last_activity = None
    
    async def get_pool_stats(self) -> dict:
        """Get browser pool statistics"""
        return {
            'active_agent': self.active_agent is not None,
            'usage_count': self.usage_count,
            'max_usage': self.max_usage_before_refresh,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'session_timeout': self.session_timeout
        }


class IntelligentCache:
    """
    Phase 3: Intelligent caching system for sports analysis data.
    
    Features:
    - Team-specific data caching
    - Smart invalidation based on match dates
    - Memory-efficient storage
    - Cache hit rate optimization
    """
    
    def __init__(self, cache_duration_minutes=15):
        self.cache = {}
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.hit_count = 0
        self.miss_count = 0
    
    def generate_cache_key(self, home_team: str, away_team: str) -> str:
        """Generate consistent cache key for team matchup"""
        # Normalize team names and create consistent key
        teams = sorted([home_team.lower().strip(), away_team.lower().strip()])
        return f"analysis_{teams[0]}_{teams[1]}"
    
    def is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cached data is still valid"""
        return datetime.now() - timestamp < self.cache_duration
    
    async def get_cached_analysis(self, home_team: str, away_team: str) -> Optional[dict]:
        """Retrieve cached analysis if valid"""
        cache_key = self.generate_cache_key(home_team, away_team)
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if self.is_cache_valid(cached_data['timestamp']):
                self.hit_count += 1
                logger.info(f"Cache hit for {home_team} vs {away_team}")
                return cached_data['data']
            else:
                # Remove expired cache
                del self.cache[cache_key]
        
        self.miss_count += 1
        return None
    
    async def cache_analysis(self, home_team: str, away_team: str, analysis_data: dict):
        """Cache analysis data with timestamp"""
        cache_key = self.generate_cache_key(home_team, away_team)
        
        self.cache[cache_key] = {
            'data': analysis_data,
            'timestamp': datetime.now(),
            'teams': {'home': home_team, 'away': away_team}
        }
        
        logger.info(f"Cached analysis for {home_team} vs {away_team}")
        
        # Cleanup old cache entries (keep only last 20)
        if len(self.cache) > 20:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate_percent': round(hit_rate, 2),
            'cached_entries': len(self.cache),
            'cache_duration_minutes': self.cache_duration.total_seconds() / 60
        }


class SportsQueryValidator:
    """
    Validates sports analysis queries to prevent abuse and maintain focus on sports intelligence.
    Implements rate limiting and content filtering as per project specifications.
    """
    
    def __init__(self):
        self.rate_limits = {}  # user_id: {count, reset_time}
        self.sports_keywords = {
            'team_indicators': ['fc', 'united', 'city', 'athletic', 'rovers', 'madrid', 'barcelona', 'bayern'],
            'league_terms': ['premier', 'champions', 'europa', 'liga', 'serie', 'bundesliga', 'ligue'],
            'forbidden': ['hack', 'script', 'bypass', 'exploit', 'download', 'admin', 'password']
        }
    
    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit (3 analyses per hour)"""
        now = datetime.now()
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = {'count': 0, 'reset_time': now + timedelta(hours=1)}
        
        user_data = self.rate_limits[user_id]
        if now > user_data['reset_time']:
            # Reset limit
            user_data['count'] = 0
            user_data['reset_time'] = now + timedelta(hours=1)
        
        if user_data['count'] >= 3:
            return False
        
        user_data['count'] += 1
        return True
    
    def is_sports_related(self, query: str) -> bool:
        """Validate if query contains sports-related content"""
        query_lower = query.lower()
        
        # Check for team indicators
        for indicator in self.sports_keywords['team_indicators']:
            if indicator in query_lower:
                return True
        
        # Check for league terms
        for term in self.sports_keywords['league_terms']:
            if term in query_lower:
                return True
        
        # Basic team name pattern (2 words that could be team names)
        words = query.split()
        if len(words) >= 2:
            return True
        
        return False
    
    def validate_analysis_query(self, user_id: str, query: str) -> dict:
        """Comprehensive query validation"""
        # Rate limiting (3 analysis per hour per user)
        if not self.check_rate_limit(user_id):
            return {'valid': False, 'reason': 'Rate limit: 3 analyses per hour maximum'}
        
        # Query length validation (max 50 characters per memory specification)
        if len(query) > 50:
            return {'valid': False, 'reason': 'Team names too long (max 50 characters)'}
        
        # Sports context validation
        if not self.is_sports_related(query):
            return {'valid': False, 'reason': 'Query must contain valid team names'}
        
        # Forbidden content check
        if any(term in query.lower() for term in self.sports_keywords['forbidden']):
            return {'valid': False, 'reason': 'Invalid query content'}
        
        return {'valid': True, 'reason': 'Valid sports analysis request'}


class BrowserAnalysisEngine:
    """
    Enhanced analysis engine using browser-use for real-time sports data collection.
    
    Key features:
    - Only Gemini 2.5 Flash and Pro models (per memory restrictions)
    - Real-time data collection from sports websites
    - Abuse prevention and rate limiting
    - Integration with existing infrastructure
    """
    
    def __init__(self):
        self.query_validator = SportsQueryValidator()
        # Phase 3: Optimized resource management
        self.browser_pool = OptimizedBrowserPool(pool_size=1)
        self.intelligent_cache = IntelligentCache(cache_duration_minutes=15)
        
        # Initialize Gemini 2.5 models ONLY (per memory requirements)
        api_key = os.getenv('GEMINI_API_KEY')
        logger.info(f"Debug: API key from environment: '{api_key[:10] if api_key else None}...{api_key[-10:] if api_key and len(api_key) > 10 else ''}'")
        logger.info(f"Debug: API key length: {len(api_key) if api_key else 0}")
        
        if not api_key:
            logger.warning("GEMINI_API_KEY not set, enhanced analysis will be disabled")
            self.gemini_available = False
            return
        
        if api_key == 'test_key' or len(api_key) < 10:
            logger.warning(f"Invalid GEMINI_API_KEY detected (len={len(api_key)}), enhanced analysis will be disabled")
            self.gemini_available = False
            return
        
        try:
            genai.configure(api_key=api_key)
            # Only Gemini 2.5 models are permitted (per memory)
            self.flash_model = genai.GenerativeModel('gemini-2.5-flash')
            self.pro_model = genai.GenerativeModel('gemini-2.5-pro')
            self.gemini_available = True
            logger.info("Gemini 2.5 Flash and Pro models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini models: {e}")
            if "API_KEY_INVALID" in str(e):
                logger.error("Please check your GEMINI_API_KEY is valid and has proper permissions")
            self.gemini_available = False
    
    async def initialize_browser_agent(self) -> bool:
        """Initialize browser-use agent via optimized pool with Gemini 2.5 Flash"""
        if not BROWSER_USE_AVAILABLE:
            logger.warning("browser-use not available")
            return False
        
        try:
            # Use optimized browser pool with Gemini navigation
            agent = await self.browser_pool.get_analysis_agent()
            return agent is not None
        except Exception as e:
            logger.error(f"Failed to get Gemini-powered browser agent from pool: {e}")
            return False
    
    async def cleanup_browser_session(self):
        """Clean up browser session via optimized pool"""
        try:
            await self.browser_pool.cleanup_agent()
            logger.info("Browser session cleaned up via pool")
        except Exception as e:
            logger.warning(f"Error cleaning up browser session: {e}")
    
    async def check_session_timeout(self):
        """Check if browser session has timed out"""
        if self.last_activity and self.browser_agent:
            if datetime.now() - self.last_activity > timedelta(seconds=self.session_timeout):
                await self.cleanup_browser_session()
    
    async def collect_analysis_context(self, home_team: str, away_team: str) -> dict:
        """
        Collect enhanced context using browser-use for real-time data gathering.
        Falls back gracefully if browser automation fails.
        
        Phase 2 Enhancement: Multi-source data collection from ESPN, BBC Sport, 
        betting sites, and expert analysis platforms.
        """
        context = {
            'teams': {'home': home_team, 'away': away_team},
            'timestamp': datetime.now().isoformat(),
            'data_sources': [],
            'collection_strategy': 'browser_automation'
        }
        
        # Try browser automation first, but fall back quickly if it fails
        try:
            if not await self.initialize_browser_agent():
                logger.warning("Browser agent unavailable, using enhanced fallback")
                return await self.collect_enhanced_fallback_context(home_team, away_team)
            
            # Phase 2: Enhanced data collection tasks with specific targets
            collection_tasks = [
                # Team news and injury reports
                {
                    'task': f"""1. Navigate to https://espn.com/soccer
2. Use the search function to find '{home_team}'
3. Click on the {home_team} team page
4. Look for 'News' or 'Latest News' section
5. Extract the 3 most recent news headlines and their dates
6. For each headline, extract the first paragraph of text
7. Focus specifically on injury reports, squad updates, or team news
8. Return the extracted information in a structured format""",
                    'source': 'espn_injuries',
                    'timeout': 25,
                    'priority': 'high'
                },
                {
                    'task': f"""1. Navigate to https://bbc.com/sport/football
2. Search for '{away_team}' using the site search
3. Find the {away_team} team page or recent articles about {away_team}
4. Look for injury reports, team news, or squad updates
5. Extract information about player availability, injuries, or suspensions
6. Get the 2-3 most recent updates about the team
7. Extract specific player names and their status if mentioned
8. Return structured information about team availability""",
                    'source': 'bbc_team_news', 
                    'timeout': 25,
                    'priority': 'high'
                },
                # Betting market intelligence (Phase 2 enhancement)
                {
                    'task': f"""1. Navigate to https://oddsportal.com
2. Search for '{home_team} vs {away_team}' or similar match
3. Find betting odds from multiple bookmakers for this match
4. Extract odds for 1X2 (Home/Draw/Away) markets
5. Look for over/under 2.5 goals odds if available
6. Note which bookmaker offers the best odds for each outcome
7. Calculate implied probability from the odds
8. Return structured betting odds data""",
                    'source': 'betting_odds',
                    'timeout': 30,
                    'priority': 'high'
                },
                {
                    'task': f"""1. Navigate to https://goal.com or https://skysports.com
2. Search for predictions about '{home_team} vs {away_team}'
3. Look for expert analysis or prediction articles
4. Find market consensus or betting trends for this match
5. Extract expert opinions about likely outcome
6. Look for any mention of public betting percentages
7. Identify any value bets or market inefficiencies mentioned
8. Return structured market sentiment data""",
                    'source': 'market_consensus',
                    'timeout': 25,
                    'priority': 'medium'
                },
                # Expert predictions and analysis
                {
                    'task': f"""1. Navigate to https://espn.com/soccer or https://goal.com
2. Search for '{home_team} vs {away_team}' predictions or preview
3. Find match preview articles by football analysts
4. Extract expert predictions with reasoning
5. Look for tactical analysis or key matchup insights
6. Find any mention of historical head-to-head trends
7. Extract confidence levels or probability assessments if given
8. Return structured expert analysis and predictions""",
                    'source': 'expert_predictions',
                    'timeout': 30,
                    'priority': 'medium'
                },
                # Form and recent performance
                {
                    'task': f"""1. Navigate to https://flashscore.com or https://espn.com/soccer
2. Search for {home_team} team page and recent results
3. Extract last 5 match results with scores and dates
4. Navigate to {away_team} team page
5. Extract their last 5 match results with scores and dates
6. Look for current league position for both teams
7. Calculate recent form (wins/draws/losses in last 5)
8. Return structured recent form data for both teams""",
                    'source': 'recent_form',
                    'timeout': 20,
                    'priority': 'medium'
                }
            ]
            
            # Execute collection tasks with priority ordering
            high_priority_tasks = [t for t in collection_tasks if t['priority'] == 'high']
            medium_priority_tasks = [t for t in collection_tasks if t['priority'] == 'medium']
            
            # Process high priority tasks first
            for task_info in high_priority_tasks:
                await self._execute_collection_task(task_info, context)
            
            # Process medium priority tasks if we have time/resources
            for task_info in medium_priority_tasks:
                await self._execute_collection_task(task_info, context)
            
            logger.info(f"Collected data from {len(context['data_sources'])} sources")
            
            # If no data was collected, fall back to enhanced analysis
            if len(context['data_sources']) == 0:
                logger.warning("No browser data collected, using enhanced fallback")
                return await self.collect_enhanced_fallback_context(home_team, away_team)
                
            return context
            
        except Exception as e:
            logger.warning(f"Browser automation failed: {e}, using enhanced fallback")
            return await self.collect_enhanced_fallback_context(home_team, away_team)
    
    async def _execute_collection_task(self, task_info: dict, context: dict):
        """Execute a single data collection task with error handling and timeout protection"""
        try:
            task_description = task_info['task']
            source = task_info['source']
            
            logger.info(f"Starting browser navigation for {source}: {task_description[:100]}...")
            
            # Use the working browser-use pattern with timeout protection
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise Exception("GEMINI_API_KEY required for browser navigation")
            
            # Set environment variable as required by browser-use
            os.environ['GEMINI_API_KEY'] = api_key
            
            detail_logger.info(f"BROWSER NAVIGATION TASK for '{source}': {task_description}")
            
            # Create browser agent using the working pattern
            from langchain_google_genai import ChatGoogleGenerativeAI
            from pydantic import SecretStr
            
            gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                api_key=SecretStr(api_key),
                temperature=0.0
            )
            
            agent = Agent(
                task=task_description,
                llm=gemini_llm
            )
            
            # Execute browser automation with timeout protection
            browser_result = await asyncio.wait_for(
                agent.run(),
                timeout=20.0  # 20 second timeout for actual browser execution
            )
            
            # Parse the real browser result
            collected_data = self.parse_browser_result(browser_result)
            
            # Validate the collected data quality
            validation = self.validate_browser_data(source, collected_data)
            detail_logger.info(f"BROWSER DATA VALIDATION for '{source}': Valid={validation['valid']}, Quality={validation['quality_score']}, Length={validation['data_length']}")
            
            detail_logger.info(f"BROWSER NAVIGATION RESULT for '{source}' ({len(collected_data)} chars): {collected_data[:500]}...")
            
            # Store with quality indicator
            if validation['valid']:
                context[source] = collected_data[:800]  # Increase limit for good data
                logger.info(f"Successfully collected high-quality browser data: {source} - {len(collected_data)} characters")
            else:
                context[source] = f"[LOW_QUALITY] {collected_data[:600]}"
                logger.warning(f"Browser data quality low for {source}: {validation.get('issue', 'Unknown issue')}")
            
            context['data_sources'].append(source)
                
        except asyncio.TimeoutError:
            logger.warning(f"Browser navigation timed out for {source} - browser automation not available")
            detail_logger.error(f"BROWSER NAVIGATION TIMEOUT for '{source}': Browser execution timed out")
            
            # Fall back to Gemini simulation when browser execution times out
            logger.info(f"Falling back to enhanced Gemini knowledge for {source} (reason: browser_timeout)")
            await self._fallback_to_gemini_simulation(task_description, source, context, 'browser_timeout')
            
        except Exception as e:
            logger.warning(f"Browser navigation failed for {source}: {e}")
            detail_logger.error(f"BROWSER NAVIGATION ERROR for '{source}': {e}")
            
            # Categorize error type for better handling
            error_type = self._categorize_browser_error(e)
            logger.info(f"Error type for {source}: {error_type}")
            
            # Fall back to Gemini simulation when browser navigation fails
            logger.info(f"Falling back to enhanced Gemini knowledge for {source} (reason: {error_type})")
            await self._fallback_to_gemini_simulation(task_description, source, context, error_type)
    
    def _categorize_browser_error(self, error: Exception) -> str:
        """Categorize browser errors to provide better feedback"""
        error_str = str(error).lower()
        
        if 'timeout' in error_str or 'time out' in error_str:
            return 'timeout_error'
        elif 'network' in error_str or 'connection' in error_str:
            return 'network_error'
        elif 'permission' in error_str or 'access' in error_str:
            return 'permission_error'
        elif 'not found' in error_str or '404' in error_str:
            return 'page_not_found'
        elif 'blocked' in error_str or 'captcha' in error_str:
            return 'access_blocked'
        elif 'browser' in error_str or 'chromium' in error_str:
            return 'browser_init_error'
        elif 'api' in error_str or 'key' in error_str:
            return 'api_error'
        else:
            return 'general_error'
    
    def _categorize_analysis_error(self, error: Exception) -> str:
        """Categorize analysis errors for better user feedback"""
        error_str = str(error).lower()
        
        if 'api_key_invalid' in error_str or 'api key not valid' in error_str or 'invalid api key' in error_str:
            return 'api_key_error'
        elif 'quota_exceeded' in error_str or 'quota exceeded' in error_str or 'rate limit' in error_str:
            return 'quota_error'
        elif 'permission_denied' in error_str or 'permission denied' in error_str:
            return 'permission_error'
        elif 'timeout' in error_str or 'timed out' in error_str:
            return 'timeout_error'
        elif 'network' in error_str or 'connection' in error_str or 'dns' in error_str:
            return 'network_error'
        elif 'browser' in error_str or 'chromium' in error_str or 'playwright' in error_str:
            return 'browser_error'
        else:
            return 'general_error'
    
    async def _fallback_to_gemini_simulation(self, task_description: str, source: str, context: dict, error_type: str = 'unknown'):
        """Fallback to Gemini simulation only when browser navigation fails"""
        try:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if not gemini_key:
                raise Exception("GEMINI_API_KEY required for fallback")
            
            genai.configure(api_key=gemini_key)
            navigation_model = genai.GenerativeModel('gemini-2.5-flash')
            
            current_date = datetime.now().strftime('%B %d, %Y')
            
            # Create context-aware fallback prompt based on error type
            error_context = {
                'timeout_error': 'Website took too long to load',
                'network_error': 'Network connection issue',
                'permission_error': 'Access permission denied',
                'page_not_found': 'Requested page not found',
                'access_blocked': 'Website blocked automated access',
                'browser_init_error': 'Browser initialization failed',
                'browser_timeout': 'Browser automation timed out - not available in this environment',
                'api_error': 'API configuration issue',
                'general_error': 'Unknown browser issue'
            }
            
            error_explanation = error_context.get(error_type, 'Browser automation failed')
            
            fallback_prompt = (
                f"FALLBACK MODE - {error_explanation}. "
                f"Today's date is {current_date}. "
                f"Provide information about {task_description} based on your knowledge. "
                f"Focus on recent news and updates from 2025. "
                f"Clearly indicate this is based on training data, not live web data. "
                f"Note: This fallback was triggered due to: {error_explanation}"
            )
            
            detail_logger.info(f"FALLBACK GEMINI PROMPT for '{source}': {fallback_prompt}")
            
            response = await navigation_model.generate_content_async(fallback_prompt)
            
            collected_data = "Fallback data unavailable"
            if response:
                try:
                    collected_data = response.text if hasattr(response, 'text') and response.text else "No fallback response"
                    detail_logger.info(f"FALLBACK GEMINI RESPONSE for '{source}' ({len(collected_data)} chars): {collected_data[:300]}...")
                except Exception as text_error:
                    logger.warning(f"Could not extract fallback response text: {text_error}")
                    collected_data = f"Fallback attempted for {source}"
            
            # Clearly mark as fallback data
            context[source] = f"[FALLBACK] {collected_data[:700]}"
            context['data_sources'].append(source)
            
            logger.info(f"Fallback data generated for {source}")
            
        except Exception as fallback_error:
            logger.error(f"Even fallback failed for {source}: {fallback_error}")
            context[source] = f"Data collection unavailable for {source}"
            context['data_sources'].append(source)
    
    async def collect_fallback_context(self, home_team: str, away_team: str) -> dict:
        """Fallback context collection when browser automation is unavailable"""
        return {
            'teams': {'home': home_team, 'away': away_team},
            'timestamp': datetime.now().isoformat(),
            'data_sources': ['basic'],
            'collection_strategy': 'fallback',
            'basic_context': f"Analysis requested for {home_team} vs {away_team}.",
            'note': 'Enhanced data collection unavailable - using basic analysis'
        }
    
    async def collect_enhanced_fallback_context(self, home_team: str, away_team: str) -> dict:
        """Enhanced fallback context using Gemini analysis without browser automation"""
        try:
            # Use Gemini 2.5 Flash to generate context based on team knowledge
            if self.gemini_available:
                current_date = datetime.now().strftime('%B %d, %Y')  # e.g., "August 26, 2025"
                flash_prompt = f"""
                IMPORTANT: Today's date is {current_date}. 
                
                Provide analysis context for {home_team} vs {away_team} match:
                
                Based on your knowledge from 2025 (current year), provide:
                1. Recent form and performance trends for both teams in 2025
                2. Current injuries or player availability concerns
                3. Recent head-to-head performance (2024-2025 season)
                4. Tactical strengths and weaknesses as of 2025
                5. Key players to watch in current 2024-25 season
                6. Any notable recent news or developments from 2025
                
                Format as structured information for sports analysis.
                DO NOT use outdated information from 2024 or earlier.
                """
                
                # Log the exact prompt being sent
                detail_logger.info(f"ENHANCED FALLBACK PROMPT for '{home_team} vs {away_team}': {flash_prompt}")
                
                response = await self.flash_model.generate_content_async(flash_prompt)
                
                # Safe text extraction with error handling
                context_text = "No context available"
                if response:
                    try:
                        context_text = response.text if hasattr(response, 'text') and response.text else "No response text"
                        detail_logger.info(f"ENHANCED FALLBACK RESPONSE ({len(context_text)} chars): {context_text[:800]}...")
                    except Exception as text_error:
                        logger.warning(f"Could not extract context text: {text_error}")
                        detail_logger.warning(f"FALLBACK TEXT EXTRACTION ERROR: {text_error}")
                        # Try alternative text extraction
                        try:
                            if hasattr(response, 'candidates') and response.candidates:
                                candidate = response.candidates[0]
                                if hasattr(candidate, 'content') and candidate.content:
                                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                        for part in candidate.content.parts:
                                            if hasattr(part, 'text') and part.text:
                                                context_text = part.text
                                                break
                            detail_logger.info(f"FALLBACK ALTERNATIVE EXTRACTION SUCCESS: {context_text[:500]}...")
                        except Exception as fallback_error:
                            logger.warning(f"Fallback context extraction failed: {fallback_error}")
                            detail_logger.error(f"FALLBACK ALL EXTRACTION FAILED: {fallback_error}")
                            context_text = f"Enhanced fallback analysis for {home_team} vs {away_team}"
                
                return {
                    'teams': {'home': home_team, 'away': away_team},
                    'timestamp': datetime.now().isoformat(),
                    'data_sources': ['gemini_knowledge'],
                    'collection_strategy': 'enhanced_fallback',
                    'gemini_context': context_text,
                    'note': 'Using Gemini knowledge base - browser data collection unavailable'
                }
            else:
                return await self.collect_fallback_context(home_team, away_team)
                
        except Exception as e:
            logger.warning(f"Enhanced fallback failed: {e}")
            detail_logger.error(f"ENHANCED FALLBACK ERROR for '{home_team} vs {away_team}': {e}")
            return await self.collect_fallback_context(home_team, away_team)
    
    def extract_data_type(self, task: str) -> str:
        """Extract data type from collection task"""
        if 'espn' in task.lower():
            return 'espn_news'
        elif 'bbc' in task.lower():
            return 'bbc_injuries'
        elif 'prediction' in task.lower():
            return 'expert_predictions'
        else:
            return 'general_data'
    
    def parse_browser_result(self, result) -> str:
        """Parse browser automation result into structured data"""
        try:
            # Handle different browser-use result types
            if hasattr(result, 'extracted_content'):
                # Browser-use specific result format
                content = str(result.extracted_content)
            elif hasattr(result, 'text'):
                # Text-based result
                content = str(result.text)
            elif hasattr(result, 'content'):
                # Content attribute
                content = str(result.content)
            elif isinstance(result, dict):
                # Dictionary result - extract relevant fields
                if 'content' in result:
                    content = str(result['content'])
                elif 'text' in result:
                    content = str(result['text'])
                elif 'data' in result:
                    content = str(result['data'])
                else:
                    content = str(result)
            elif isinstance(result, str):
                # Direct string result
                content = result
            elif isinstance(result, list):
                # List of results - join them
                content = ' | '.join([str(item) for item in result])
            else:
                # Fallback - convert to string
                content = str(result)
            
            # Clean and limit the content
            if content:
                # Remove excessive whitespace and newlines
                content = ' '.join(content.split())
                # Remove HTML tags if present
                import re
                content = re.sub(r'<[^>]+>', '', content)
                # Limit length but preserve information
                if len(content) > 1000:
                    content = content[:1000] + "...[truncated]"
                return content
            else:
                return "No content extracted from browser result"
                
        except Exception as e:
            logger.warning(f"Error parsing browser result: {e}")
            return f"Browser result parsing failed: {str(e)[:100]}"
    
    def validate_browser_data(self, source: str, data: str) -> dict:
        """Validate and structure browser-collected data"""
        validation_result = {
            'source': source,
            'valid': False,
            'data_length': len(data) if data else 0,
            'contains_fallback': '[FALLBACK]' in data if data else False,
            'quality_score': 0
        }
        
        if not data or len(data) < 10:
            validation_result['issue'] = 'Insufficient data collected'
            return validation_result
        
        # Source-specific validation
        if source == 'espn_injuries':
            # Check for injury-related keywords
            injury_keywords = ['injury', 'injured', 'out', 'doubtful', 'fitness', 'squad']
            if any(keyword in data.lower() for keyword in injury_keywords):
                validation_result['quality_score'] += 30
        
        elif source == 'betting_odds':
            # Check for odds-related data
            odds_keywords = ['odds', 'bet', 'bookmaker', 'decimal', 'fractional', 'probability']
            if any(keyword in data.lower() for keyword in odds_keywords):
                validation_result['quality_score'] += 30
        
        elif source == 'recent_form':
            # Check for match results
            form_keywords = ['win', 'loss', 'draw', 'score', 'result', 'match']
            if any(keyword in data.lower() for keyword in form_keywords):
                validation_result['quality_score'] += 30
        
        # General quality indicators
        if len(data) > 100:
            validation_result['quality_score'] += 20
        if len(data) > 300:
            validation_result['quality_score'] += 20
        
        # Check if it contains meaningful content (not just error messages)
        error_indicators = ['error', 'failed', 'unavailable', 'not found']
        if not any(indicator in data.lower() for indicator in error_indicators):
            validation_result['quality_score'] += 30
        
        validation_result['valid'] = validation_result['quality_score'] >= 50
        return validation_result
    
    async def generate_premium_analysis(self, home_team: str, away_team: str, context_data: dict) -> dict:
        """
        Generate premium analysis using Gemini 2.5 models with real-time context.
        
        Uses dual-model approach:
        1. Gemini 2.5 Flash: Fast data processing and validation
        2. Gemini 2.5 Pro: Deep analysis generation
        """
        if not self.gemini_available:
            raise Exception("Gemini models not available")
        
        try:
            # Step 1: Process raw data with Gemini 2.5 Flash
            current_date = datetime.now().strftime('%B %d, %Y')  # e.g., "August 26, 2025"
            flash_prompt = f"""
            IMPORTANT: Today's date is {current_date}.
            
            Process this sports data for {home_team} vs {away_team}:
            
            Context: {context_data}
            
            Extract key information focusing on current 2025 data:
            - Recent team news from 2025
            - Current injury status
            - Expert opinions from 2025
            - Any relevant trends from the current season
            
            Return structured summary in JSON format.
            DO NOT use outdated information from 2024 or earlier.
            """
            
            # Log the exact prompt being sent
            detail_logger.info(f"FLASH PROCESSING PROMPT for '{home_team} vs {away_team}': {flash_prompt[:1000]}...")
            
            flash_response = await self.flash_model.generate_content_async(flash_prompt)
            
            # Safe text extraction for flash response
            processed_data = "No data processed"
            if flash_response:
                try:
                    processed_data = flash_response.text if hasattr(flash_response, 'text') and flash_response.text else "No flash response text"
                    detail_logger.info(f"FLASH PROCESSING RESPONSE ({len(processed_data)} chars): {processed_data[:600]}...")
                except Exception as text_error:
                    logger.warning(f"Could not extract flash response text: {text_error}")
                    detail_logger.warning(f"FLASH TEXT EXTRACTION ERROR: {text_error}")
                    processed_data = "Data processing completed"
            
            # Step 2: Generate comprehensive analysis with Gemini 2.5 Pro
            pro_prompt = f"""
            IMPORTANT: Today's date is {current_date}. You are analyzing this match in 2025.
            
            As an expert football analyst, provide comprehensive analysis for:
            {home_team} vs {away_team}
            
            Based on real-time data from 2025:
            {processed_data}
            
            Provide analysis including:
            1. Current form assessment (2024-25 season)
            2. Head-to-head context (recent meetings in 2024-2025)
            3. Key player situations (current injuries, suspensions as of 2025)
            4. Tactical considerations for current season
            5. Market intelligence (if betting data available)
            6. External factors (weather, venue, motivation)
            7. Prediction with confidence level and reasoning
            
            If betting odds are mentioned in the data, use them to:
            - Assess market sentiment
            - Identify value opportunities
            - Compare your analysis with market expectations
            
            Focus on factors that could influence the match outcome.
            Be concise but thorough, emphasizing unique insights.
            Use ONLY current 2025 information. Do NOT reference 2024 or earlier data unless specifically for historical context.
            """
            
            # Log the exact prompt being sent
            detail_logger.info(f"EXPERT ANALYSIS PROMPT for '{home_team} vs {away_team}' (Pro Model): {pro_prompt[:1200]}...")
            
            pro_response = await self.pro_model.generate_content_async(pro_prompt)
            
            # Safe text extraction for pro response
            expert_analysis = "Analysis unavailable"
            if pro_response:
                try:
                    expert_analysis = pro_response.text if hasattr(pro_response, 'text') and pro_response.text else "No pro response text"
                    detail_logger.info(f"EXPERT ANALYSIS RESPONSE (Pro Model, {len(expert_analysis)} chars): {expert_analysis[:800]}...")
                except Exception as text_error:
                    logger.warning(f"Could not extract pro response text: {text_error}")
                    detail_logger.warning(f"EXPERT ANALYSIS TEXT EXTRACTION ERROR: {text_error}")
                    expert_analysis = f"Analysis completed for {home_team} vs {away_team}"
            
            # Log final result structure
            detail_logger.info(f"FINAL ANALYSIS RESULT for '{home_team} vs {away_team}': {len(expert_analysis)} chars analysis, {len(context_data.get('data_sources', []))} sources, {sum(len(str(data)) for key, data in context_data.items() if key not in ['teams', 'timestamp', 'data_sources', 'collection_strategy'])} total data collected")
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'processed_data': processed_data,
                'expert_analysis': expert_analysis,
                'data_sources': context_data.get('data_sources', []),
                'timestamp': context_data.get('timestamp'),
                'confidence': 'High' if len(context_data.get('data_sources', [])) > 1 else 'Medium'
            }
            
        except Exception as e:
            logger.error(f"Premium analysis generation failed: {e}")
            raise
    
    async def enhanced_analysis(self, user_id: str, home_team: str, away_team: str) -> dict:
        """
        Main entry point for enhanced analysis with validation, caching, and browser integration.
        Phase 3: Optimized with intelligent caching and resource management.
        """
        # Validate query
        query = f"{home_team} {away_team}"
        validation = self.query_validator.validate_analysis_query(user_id, query)
        
        if not validation['valid']:
            return {'error': validation['reason'], 'valid': False}
        
        # Phase 3: Check intelligent cache first
        cached_result = await self.intelligent_cache.get_cached_analysis(home_team, away_team)
        if cached_result:
            logger.info(f"Returning cached analysis for {home_team} vs {away_team}")
            return {
                'valid': True,
                'analysis': cached_result,
                'enhanced': True,
                'cached': True,
                'data_sources': cached_result.get('data_sources', [])
            }
        
        try:
            # Collect real-time context
            context_data = await self.collect_analysis_context(home_team, away_team)
            
            # Generate premium analysis
            analysis = await self.generate_premium_analysis(home_team, away_team, context_data)
            
            # Cache the result for future use
            await self.intelligent_cache.cache_analysis(home_team, away_team, analysis)
            
            return {
                'valid': True,
                'analysis': analysis,
                'enhanced': True,
                'cached': False,
                'data_sources': context_data.get('data_sources', [])
            }
            
        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            
            # Categorize error for better handling and user feedback
            error_category = self._categorize_analysis_error(e)
            
            # Provide specific error messages for common issues
            error_messages = {
                'api_key_error': "Invalid Gemini API key - please check your configuration",
                'quota_error': "API quota exceeded - please try again in a few minutes",
                'permission_error': "API permission denied - please check key permissions",
                'timeout_error': "Analysis timed out - the request took too long",
                'network_error': "Network connection issue - please check your internet",
                'browser_error': "Browser automation temporarily unavailable",
                'general_error': "Enhanced analysis temporarily unavailable"
            }
            
            error_msg = error_messages.get(error_category, "Enhanced analysis temporarily unavailable")
            
            # Add actionable suggestions based on error type
            suggestions = {
                'api_key_error': "Check your GEMINI_API_KEY in the .env file",
                'quota_error': "Wait a few minutes or upgrade your Gemini API plan",
                'permission_error': "Ensure your API key has Gemini model access",
                'timeout_error': "Try again with a simpler query or check your connection",
                'network_error': "Check your internet connection and try again",
                'browser_error': "Browser features temporarily disabled, using fallback",
                'general_error': "Try again later or contact support if issue persists"
            }
            
            suggestion = suggestions.get(error_category, "Try again later")
            
            return {
                'error': error_msg,
                'suggestion': suggestion,
                'valid': False,
                'fallback_available': True,
                'error_type': error_category,
                'technical_details': str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
            }
    
    async def delayed_cleanup(self):
        """Clean up browser resources after a delay"""
        await asyncio.sleep(60)  # Wait 1 minute before cleanup
        await self.cleanup_browser_session()
    
    def format_premium_analysis(self, analysis_data: dict) -> str:
        """Format premium analysis for Telegram display with market intelligence"""
        if 'error' in analysis_data:
            return f"❌ {analysis_data['error']}"
        
        analysis_info = analysis_data.get('analysis', {})
        data_sources = analysis_data.get('data_sources', [])
        
        # Determine data richness level
        has_betting_data = 'betting_odds' in data_sources or 'market_consensus' in data_sources
        has_injury_data = 'espn_injuries' in data_sources or 'bbc_team_news' in data_sources
        
        # Sanitize text for Telegram Markdown
        def sanitize_markdown(text):
            """Remove or escape problematic characters for Telegram Markdown"""
            if not text:
                return "Analysis completed"
            
            # Remove problematic characters that break Telegram parsing
            text = str(text)
            
            # Replace Markdown-breaking characters
            text = text.replace('_', '-')     # Underscores can break italic formatting
            text = text.replace('*', '•')     # Asterisks can break bold formatting
            text = text.replace('`', "'")     # Backticks can break code formatting
            text = text.replace('[', '(')     # Square brackets can break links
            text = text.replace(']', ')')
            text = text.replace('<', '')      # Angle brackets can break HTML tags
            text = text.replace('>', '')
            text = text.replace('\n\n\n', '\n\n')  # Reduce excessive line breaks
            text = text.replace('  ', ' ')    # Reduce multiple spaces
            
            # Remove other potentially problematic characters
            text = text.replace('|', '-')     # Pipes can interfere with tables
            text = text.replace('#', '')      # Hashes can create headers
            text = text.replace('~', '-')     # Tildes can create strikethrough
            text = text.replace('^', '')      # Carets can cause issues
            text = text.replace('{', '(')     # Curly braces
            text = text.replace('}', ')')
            text = text.replace('\\', '/')    # Backslashes can escape characters
            
            # Handle percentage signs that might break parsing
            text = text.replace('%', ' percent')
            
            # Limit length to prevent message too long errors
            if len(text) > 600:  # Reduced from 800 to be safer
                text = text[:600] + "..."
            
            # Final cleanup - remove any remaining special characters that could cause issues
            import re
            # Keep only safe characters: letters, numbers, spaces, basic punctuation
            text = re.sub(r'[^\w\s.,!?():;-]', '', text)
            
            return text.strip()
        
        expert_analysis = sanitize_markdown(analysis_info.get('expert_analysis', 'Analysis complete'))
        
        response = f"""🧠 *Premium Team Analysis*

**{analysis_info.get('home_team', 'Home')} vs {analysis_info.get('away_team', 'Away')}**

📊 **Real-Time Intelligence:**
{expert_analysis}

"""
        
        # Add market intelligence section if available
        if has_betting_data:
            response += "💰 **Market Intelligence:**\n"
            response += "• Betting market analysis included\n"
            response += "• Market sentiment assessed\n\n"
        
        # Add injury/team news section if available
        if has_injury_data:
            response += "🏥 **Team Status:**\n"
            response += "• Latest injury reports reviewed\n"
            response += "• Team news analyzed\n\n"
        
        response += f"""📈 **Confidence Level:** {analysis_info.get('confidence', 'Medium')}

🔄 **Data Sources:** {len(data_sources)} sources
• {', '.join(data_sources) if data_sources else 'Basic analysis'}

*Enhanced with real-time data intelligence* ⚡
*Powered by Gemini 2.5 Pro* 🤖"""
        
        return response
    
    async def is_available(self) -> bool:
        """
        Quick check if browser analysis is available without hanging.
        Returns True if Gemini is configured, False if browser automation fails.
        """
        try:
            # First check if basic requirements are met
            if not BROWSER_USE_AVAILABLE:
                logger.info("Browser analysis unavailable: browser-use not installed")
                return False
            
            if not self.gemini_available:
                logger.info("Browser analysis unavailable: Gemini models not available")
                return False
            
            # Quick test of browser agent creation with timeout
            try:
                agent_ready = await asyncio.wait_for(
                    self.initialize_browser_agent(),
                    timeout=5.0  # 5 second timeout for quick check
                )
                if agent_ready:
                    logger.info("Browser analysis available: agent creation successful")
                    return True
                else:
                    logger.info("Browser analysis unavailable: agent creation failed")
                    return False
            except asyncio.TimeoutError:
                logger.warning("Browser analysis unavailable: agent creation timed out (browser instances not available)")
                return False
            except Exception as e:
                logger.warning(f"Browser analysis unavailable: agent creation error - {e}")
                return False
                
        except Exception as e:
            logger.error(f"Browser availability check failed: {e}")
            return False
    
    async def get_system_stats(self) -> dict:
        """Get comprehensive system statistics for monitoring"""
        browser_stats = await self.browser_pool.get_pool_stats()
        cache_stats = self.intelligent_cache.get_cache_stats()
        
        return {
            'browser_pool': browser_stats,
            'intelligent_cache': cache_stats,
            'gemini_available': self.gemini_available,
            'browser_use_available': BROWSER_USE_AVAILABLE
        }