"""
Motion-based detector using background subtraction.
Detects moving objects (people, vehicles, etc.)
"""

import numpy as np
import cv2
from typing import List, Dict, Any
from .base_detector import BaseDetector, Detection


class MotionDetector(BaseDetector):
    """
    Detect moving objects using background subtraction.
    Ideal for static camera surveillance scenarios.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize motion detector.
        
        Config options:
            - method: 'mog2' or 'knn'
            - history: Number of frames for background model
            - var_threshold: Threshold for MOG2
            - detect_shadows: Whether to detect shadows
            - min_area: Minimum object area
            - learning_rate: Background update rate (-1 = auto)
        """
        super().__init__(config)
        
        method = self.config.get('method', 'mog2')
        history = self.config.get('history', 500)
        detect_shadows = self.config.get('detect_shadows', False)
        self.min_area = self.config.get('min_area', 500)
        self.learning_rate = self.config.get('learning_rate', -1)
        
        # Initialize background subtractor
        if method == 'mog2':
            var_threshold = self.config.get('var_threshold', 16)
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows
            )
        elif method == 'knn':
            dist2threshold = self.config.get('dist2threshold', 400.0)
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=dist2threshold,
                detectShadows=detect_shadows
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.is_initialized = True
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect moving objects.
        
        Args:
            frame: Input frame
            
        Returns:
            List of Detection objects
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learning_rate)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to detections
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, w, h)
            centroid = self._calculate_centroid(bbox)
            
            features = {
                'area': area,
                'contour': contour
            }
            
            detection = Detection(
                bbox=bbox,
                centroid=centroid,
                confidence=1.0,
                class_name='moving_object',
                features=features
            )
            
            detections.append(detection)
        
        return detections