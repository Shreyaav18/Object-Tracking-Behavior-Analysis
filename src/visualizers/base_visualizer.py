"""
Base classes for all visualizers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from ..trackers.base_tracker import Track


class BaseVisualizer(ABC):
    """
    Abstract base class for all visualizers.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def visualize(self, *args, **kwargs):
        """Create visualization."""
        pass