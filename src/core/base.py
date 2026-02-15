"""
Base classes for all framework components.
Defines standard interfaces for detectors, trackers, analyzers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np


class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Processed frame
        """
        pass
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Allow preprocessor to be called as a function."""
        return self.process(frame)


class FrameSequence:
    """Container for a sequence of frames with metadata."""
    
    def __init__(self, frames: List[np.ndarray], fps: float = None, 
                 metadata: Dict = None):
        """
        Initialize frame sequence.
        
        Args:
            frames: List of frames
            fps: Frames per second
            metadata: Additional metadata (resolution, etc.)
        """
        self.frames = frames
        self.fps = fps or 30.0
        self.metadata = metadata or {}
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return self.frames[idx]
    
    @property
    def shape(self):
        """Get shape of frames."""
        if self.frames:
            return self.frames[0].shape
        return None
    
    @property
    def duration(self):
        """Get duration in seconds."""
        return len(self.frames) / self.fps if self.fps else None