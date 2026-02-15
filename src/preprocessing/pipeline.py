"""
Preprocessing pipeline that combines multiple preprocessors.
"""

from typing import List
import numpy as np
from ..core.base import BasePreprocessor


class PreprocessingPipeline:
    """
    Chain multiple preprocessors together.
    """
    
    def __init__(self, preprocessors: List[BasePreprocessor] = None):
        """
        Initialize pipeline.
        
        Args:
            preprocessors: List of preprocessor objects
        """
        self.preprocessors = preprocessors or []
    
    def add(self, preprocessor: BasePreprocessor):
        """Add a preprocessor to the pipeline."""
        self.preprocessors.append(preprocessor)
        return self
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame through all preprocessors in sequence.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        result = frame.copy()
        
        for preprocessor in self.preprocessors:
            result = preprocessor.process(result)
        
        return result
    
    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process multiple frames.
        
        Args:
            frames: List of frames
            
        Returns:
            List of processed frames
        """
        return [self.process(frame) for frame in frames]
    
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Allow pipeline to be called as a function."""
        return self.process(frame)