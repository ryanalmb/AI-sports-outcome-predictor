"""
Stream renderer for Flash Live Degen feature.
This module formats real-time updates in degen language for Telegram.
"""

import logging
from typing import Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class StreamRenderer:
    """
    Renderer for live streaming updates in degen language.
    """
    
    def __init__(self):
        """
        Initialize the StreamRenderer.
        """
        pass
    
    def format_update(self, update_data: Dict[str, Any]) -> str:
        """
        Format a live update in degen language.
        
        Args:
            update_data: Dictionary containing update information
            
        Returns:
            Formatted string for Telegram
        """
        update_type = update_data.get("type", "unknown")
        
        if update_type == "update":
            return self._format_source_update(update_data)
        elif update_type == "error":
            return self._format_error(update_data)
        elif update_type == "session_end":
            return self._format_session_end(update_data)
        elif update_type == "final_summary":
            return self._format_final_summary(update_data)
        else:
            return f"🔥 *Unknown Update*: {update_data.get('message', 'No message')}"
    
    def _format_source_update(self, update_data: Dict[str, Any]) -> str:
        """
        Format a source update message.
        
        Args:
            update_data: Dictionary containing source update information
            
        Returns:
            Formatted string for Telegram
        """
        message = update_data.get("message", "")
        source = update_data.get("source", {})
        url = source.get("url", "")
        
        # Extract domain for display
        domain = ""
        if url:
            try:
                domain = urlparse(url).hostname or ""
            except Exception:
                domain = "source"
        
        # Format based on message content
        if "injury" in message.lower():
            return f"🩹 *Injury Alert* on [{domain}]({url}): Fresh deets dropping!"
        elif "bookie" in message.lower() or "odds" in message.lower() or "line" in message.lower():
            return f"📈 *Bookie Shift* on [{domain}]({url}): Market vibes changing!"
        elif "press" in message.lower() or "quote" in message.lower():
            return f"🎙️ *Press Drop* on [{domain}]({url}): Coach just spilled the tea!"
        else:
            return f"🔍 *Scouted* on [{domain}]({url}): Something's brewing..."
    
    def _format_error(self, update_data: Dict[str, Any]) -> str:
        """
        Format an error message.
        
        Args:
            update_data: Dictionary containing error information
            
        Returns:
            Formatted string for Telegram
        """
        message = update_data.get("message", "Unknown error")
        return f"🚨 *Research Error*: {message} (NFA)"
    
    def _format_session_end(self, update_data: Dict[str, Any]) -> str:
        """
        Format a session end message.
        
        Args:
            update_data: Dictionary containing session end information
            
        Returns:
            Formatted string for Telegram
        """
        message = update_data.get("message", "Research session completed")
        return f"🏁 *Session Complete*: {message}"
    
    def _format_final_summary(self, update_data: Dict[str, Any]) -> str:
        """
        Format a final summary message.
        
        Args:
            update_data: Dictionary containing final summary information
            
        Returns:
            Formatted string for Telegram
        """
        message = update_data.get("message", "Research complete!")
        sources_count = len(update_data.get("sources", []))
        return f"📊 *Final Summary*: {message}\n\nFound {sources_count} spicy sources to cook up the degen playbook! (NFA)"