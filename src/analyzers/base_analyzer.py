"""
Base classes for all analyzers.
Analyzers extract features and insights from tracked data.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from ..trackers.base_tracker import Track


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def analyze(self, tracks: List[Track]) -> Dict[str, Any]:
        """
        Analyze tracks and extract features/insights.
        
        Args:
            tracks: List of Track objects
            
        Returns:
            Dictionary of analysis results
        """
        pass
    
    def analyze_single(self, track: Track) -> Dict[str, Any]:
        """
        Analyze a single track.
        
        Args:
            track: Single Track object
            
        Returns:
            Dictionary of features for this track
        """
        return self.analyze([track])