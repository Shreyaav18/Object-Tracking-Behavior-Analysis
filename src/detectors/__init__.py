"""
Detector module initialization and factory.
"""

from .base_detector import BaseDetector, Detection
from .blob_detector import BlobDetector, AdvancedBlobDetector
from .yolo_detector import YOLODetector, CustomYOLODetector, YOLO_AVAILABLE
from .motion_detector import MotionDetector


class DetectorFactory:
    """
    Factory for creating detectors based on config or type.
    """
    
    DETECTORS = {
        'blob': BlobDetector,
        'blob_advanced': AdvancedBlobDetector,
        'yolo': YOLODetector,
        'custom_yolo': CustomYOLODetector,
        'motion': MotionDetector,
    }
    
    @classmethod
    def create(cls, detector_type: str, config: dict = None) -> BaseDetector:
        """
        Create a detector instance.
        
        Args:
            detector_type: Type of detector ('blob', 'yolo', etc.)
            config: Configuration dictionary
            
        Returns:
            Detector instance
        """
        if detector_type not in cls.DETECTORS:
            raise ValueError(f"Unknown detector type: {detector_type}. "
                           f"Available: {list(cls.DETECTORS.keys())}")
        
        detector_class = cls.DETECTORS[detector_type]
        return detector_class(config)
    
    @classmethod
    def list_available(cls) -> list:
        """List all available detector types."""
        return list(cls.DETECTORS.keys())


__all__ = [
    'BaseDetector',
    'Detection',
    'BlobDetector',
    'AdvancedBlobDetector',
    'YOLODetector',
    'CustomYOLODetector',
    'MotionDetector',
    'DetectorFactory'
]