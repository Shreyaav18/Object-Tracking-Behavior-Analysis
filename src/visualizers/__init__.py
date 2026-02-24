"""
Visualizer module initialization and factory.
"""

from .base_visualizer import BaseVisualizer
from .video_annotator import VideoAnnotator, DetectionAnnotator
from .trajectory_plotter import TrajectoryPlotter
from .heatmap_generator import HeatmapGenerator
from .dashboard import DashboardGenerator


class VisualizerFactory:
    """Factory for creating visualizers."""
    
    VISUALIZERS = {
        'video': VideoAnnotator,
        'detection': DetectionAnnotator,
        'trajectory': TrajectoryPlotter,
        'heatmap': HeatmapGenerator,
        'dashboard': DashboardGenerator,
    }
    
    @classmethod
    def create(cls, visualizer_type: str, config: dict = None) -> BaseVisualizer:
        """Create a visualizer instance."""
        if visualizer_type not in cls.VISUALIZERS:
            raise ValueError(f"Unknown visualizer: {visualizer_type}")
        
        visualizer_class = cls.VISUALIZERS[visualizer_type]
        return visualizer_class(config)


__all__ = [
    'BaseVisualizer',
    'VideoAnnotator',
    'DetectionAnnotator',
    'TrajectoryPlotter',
    'HeatmapGenerator',
    'DashboardGenerator',
    'VisualizerFactory'
]