"""
Blob detector for detecting circular/elliptical objects.
Ideal for cells, particles, bacteria, nuclei, etc.
"""

import numpy as np
import cv2
from typing import List, Dict, Any
from .base_detector import BaseDetector, Detection


class BlobDetector(BaseDetector):
    """
    Detect blob-like objects (cells, particles, etc.)
    Uses traditional CV methods: thresholding + contour detection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize blob detector.
        
        Config options:
            - min_area: Minimum blob area (pixels)
            - max_area: Maximum blob area (pixels)
            - min_circularity: Minimum circularity (0-1)
            - threshold_method: 'otsu', 'adaptive', 'manual'
            - threshold_value: Value for manual thresholding
            - invert: Whether to invert image (for dark objects on light bg)
            - morphology: Apply morphological operations
        """
        super().__init__(config)
        
        # Default parameters
        self.min_area = self.config.get('min_area', 10)
        self.max_area = self.config.get('max_area', 5000)
        self.min_circularity = self.config.get('min_circularity', 0.3)
        self.threshold_method = self.config.get('threshold_method', 'otsu')
        self.threshold_value = self.config.get('threshold_value', 127)
        self.invert = self.config.get('invert', False)
        self.use_morphology = self.config.get('morphology', True)
        
        self.is_initialized = True
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect blobs in frame.
        
        Args:
            frame: Input frame (grayscale or color)
            
        Returns:
            List of Detection objects
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply thresholding
        binary = self._threshold(gray)
        
        # Invert if needed
        if self.invert:
            binary = cv2.bitwise_not(binary)
        
        # Apply morphological operations
        if self.use_morphology:
            binary = self._apply_morphology(binary)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and convert to detections
        detections = []
        for contour in contours:
            detection = self._contour_to_detection(contour)
            if detection is not None:
                detections.append(detection)
        
        return detections
    
    def _threshold(self, gray: np.ndarray) -> np.ndarray:
        """Apply thresholding to grayscale image."""
        if self.threshold_method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif self.threshold_method == 'adaptive':
            block_size = self.config.get('block_size', 11)
            c = self.config.get('c', 2)
            binary = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block_size, c)
        
        elif self.threshold_method == 'manual':
            _, binary = cv2.threshold(gray, self.threshold_value, 255,
                                     cv2.THRESH_BINARY)
        
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        
        return binary
    
    def _apply_morphology(self, binary: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up binary image."""
        kernel_size = self.config.get('morph_kernel_size', 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Opening to remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill holes
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return binary
    
    def _contour_to_detection(self, contour: np.ndarray) -> Detection:
        """
        Convert contour to Detection object.
        
        Args:
            contour: OpenCV contour
            
        Returns:
            Detection object or None if filtered out
        """
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < self.min_area or area > self.max_area:
            return None
        
        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Filter by circularity
        if circularity < self.min_circularity:
            return None
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, w, h)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = x + w/2, y + h/2
        
        centroid = (cx, cy)
        
        # Additional features
        features = {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': w / h if h > 0 else 0,
            'contour': contour  # Store for later use if needed
        }
        
        return Detection(
            bbox=bbox,
            centroid=centroid,
            confidence=1.0,
            class_name='blob',
            features=features
        )


class AdvancedBlobDetector(BaseDetector):
    """
    Advanced blob detector using Difference of Gaussians (DoG)
    or Laplacian of Gaussian (LoG).
    Better for noisy images or overlapping objects.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize advanced blob detector.
        
        Config options:
            - method: 'dog' or 'log'
            - min_sigma: Minimum sigma for Gaussian
            - max_sigma: Maximum sigma for Gaussian
            - num_sigma: Number of sigma values
            - threshold: Detection threshold
            - overlap: Maximum overlap between blobs (0-1)
        """
        super().__init__(config)
        
        self.method = self.config.get('method', 'log')
        self.min_sigma = self.config.get('min_sigma', 1)
        self.max_sigma = self.config.get('max_sigma', 50)
        self.num_sigma = self.config.get('num_sigma', 10)
        self.threshold = self.config.get('threshold', 0.1)
        self.overlap = self.config.get('overlap', 0.5)
        
        self.is_initialized = True
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect blobs using scale-space methods.
        
        Args:
            frame: Input frame (grayscale or color)
            
        Returns:
            List of Detection objects
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Normalize to 0-1 range
        gray = gray.astype(np.float32) / 255.0
        
        # Detect blobs
        if self.method == 'log':
            blobs = self._detect_log(gray)
        elif self.method == 'dog':
            blobs = self._detect_dog(gray)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Convert to Detection objects
        detections = []
        for blob in blobs:
            y, x, r = blob
            
            # Calculate bounding box
            x_min = max(0, int(x - r))
            y_min = max(0, int(y - r))
            w = int(2 * r)
            h = int(2 * r)
            bbox = (x_min, y_min, w, h)
            
            centroid = (float(x), float(y))
            
            features = {
                'radius': float(r),
                'area': np.pi * r * r
            }
            
            detection = Detection(
                bbox=bbox,
                centroid=centroid,
                confidence=1.0,
                class_name='blob',
                features=features
            )
            detections.append(detection)
        
        return detections
    
    def _detect_log(self, image: np.ndarray) -> np.ndarray:
        """Detect blobs using Laplacian of Gaussian."""
        from skimage.feature import blob_log
        
        blobs = blob_log(
            image,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            num_sigma=self.num_sigma,
            threshold=self.threshold,
            overlap=self.overlap
        )
        
        # Compute radii
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
        
        return blobs
    
    def _detect_dog(self, image: np.ndarray) -> np.ndarray:
        """Detect blobs using Difference of Gaussians."""
        from skimage.feature import blob_dog
        
        blobs = blob_dog(
            image,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
            threshold=self.threshold,
            overlap=self.overlap
        )
        
        # Compute radii
        blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
        
        return blobs