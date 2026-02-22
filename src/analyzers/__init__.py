"""
Analyzer module initialization and factory.
"""

from .base_analyzer import BaseAnalyzer
from .motion_analyzer import MotionAnalyzer
from .trajectory_analyzer import TrajectoryAnalyzer
from .behavior_classifier import BehaviorClassifier, SKLEARN_AVAILABLE
from .statistical_analyzer import StatisticalAnalyzer


class AnalyzerFactory:
    """Factory for creating analyzers."""
    
    ANALYZERS = {
        'motion': MotionAnalyzer,
        'trajectory': TrajectoryAnalyzer,
        'behavior': BehaviorClassifier,
        'statistical': StatisticalAnalyzer,
    }
    
    @classmethod
    def create(cls, analyzer_type: str, config: dict = None) -> BaseAnalyzer:
        """Create an analyzer instance."""
        if analyzer_type not in cls.ANALYZERS:
            raise ValueError(f"Unknown analyzer: {analyzer_type}")
        
        analyzer_class = cls.ANALYZERS[analyzer_type]
        return analyzer_class(config)


__all__ = [
    'BaseAnalyzer',
    'MotionAnalyzer',
    'TrajectoryAnalyzer',
    'BehaviorClassifier',
    'StatisticalAnalyzer',
    'AnalyzerFactory'
]