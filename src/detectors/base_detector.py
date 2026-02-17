"""
Abstract base class for all detectors.
Defines the standard interface for object detection.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    """
    Container for a single detection.
    """
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    centroid: Tuple[float, float]     # (x, y)
    confidence: float = 1.0           # Detection confidence score
    class_id: int = 0                 # Class ID (for multi-class detection)
    class_name: str = "object"        # Class name
    mask: np.ndarray = None           # Optional segmentation mask
    features: Dict[str, Any] = None   # Additional features (area, etc.)
    
    def __post_init__(self):
        if self.features is None:
            self.features = {}
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return self.bbox[2] * self.bbox[3]
    
    @property
    def center(self) -> Tuple[float, float]:
        """Alias for centroid."""
        return self.centroid
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'bbox': self.bbox,
            'centroid': self.centroid,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'area': self.area,
            'features': self.features
        }


class BaseDetector(ABC):
    """
    Abstract base class for all detectors.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of Detection objects
        """
        pass
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect objects in multiple frames.
        
        Args:
            frames: List of frames
            
        Returns:
            List of detection lists (one per frame)
        """
        return [self.detect(frame) for frame in frames]
    
    def __call__(self, frame: np.ndarray) -> List[Detection]:
        """Allow detector to be called as a function."""
        return self.detect(frame)
    
    @staticmethod
    def _calculate_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Calculate centroid from bounding box.
        
        Args:
            bbox: (x, y, width, height)
            
        Returns:
            (center_x, center_y)
        """
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)