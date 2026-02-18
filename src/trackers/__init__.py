"""
Tracker module initialization and factory.
"""

from .base_tracker import BaseTracker, Track
from .simple_tracker import SimpleTracker
from .trackpy_tracker import TrackPyTracker, TRACKPY_AVAILABLE
from .sort_tracker import SORTTracker


class TrackerFactory:
    """
    Factory for creating trackers.
    """
    
    TRACKERS = {
        'simple': SimpleTracker,
        'trackpy': TrackPyTracker,
        'sort': SORTTracker,
    }
    
    @classmethod
    def create(cls, tracker_type: str, config: dict = None) -> BaseTracker:
        """
        Create a tracker instance.
        
        Args:
            tracker_type: Type of tracker ('simple', 'trackpy', 'sort')
            config: Configuration dictionary
            
        Returns:
            Tracker instance
        """
        if tracker_type not in cls.TRACKERS:
            raise ValueError(f"Unknown tracker type: {tracker_type}. "
                           f"Available: {list(cls.TRACKERS.keys())}")
        
        if tracker_type == 'trackpy' and not TRACKPY_AVAILABLE:
            raise ImportError("trackpy not installed. Run: pip install trackpy")
        
        tracker_class = cls.TRACKERS[tracker_type]
        return tracker_class(config)
    
    @classmethod
    def list_available(cls) -> list:
        """List all available tracker types."""
        return list(cls.TRACKERS.keys())


__all__ = [
    'BaseTracker',
    'Track',
    'SimpleTracker',
    'TrackPyTracker',
    'SORTTracker',
    'TrackerFactory'
]