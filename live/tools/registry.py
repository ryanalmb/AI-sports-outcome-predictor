"""
Tool registry for Flash Live Degen feature.
This module provides a registry for managing tools used in live sessions.
"""

import logging
from typing import Dict, Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    A registry for managing tools used in live sessions.
    """
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_descriptions: Dict[str, str] = {}
    
    def register(self, name: str, description: str = ""):
        """
        Register a tool with the registry.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        def decorator(func: Callable):
            self._tools[name] = func
            self._tool_descriptions[name] = description
            logger.info(f"Registered tool: {name}")
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Get a registered tool by name.
        
        Args:
            name: The name of the tool
            
        Returns:
            The tool function, or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self) -> Dict[str, str]:
        """
        List all registered tools with their descriptions.
        
        Returns:
            A dictionary mapping tool names to descriptions
        """
        return self._tool_descriptions.copy()
    
    def call_tool(self, name: str, *args, **kwargs) -> Any:
        """
        Call a registered tool by name.
        
        Args:
            name: The name of the tool
            *args: Positional arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool
            
        Returns:
            The result of calling the tool
            
        Raises:
            ValueError: If the tool is not registered
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' is not registered")
        
        logger.info(f"Calling tool: {name}")
        return tool(*args, **kwargs)