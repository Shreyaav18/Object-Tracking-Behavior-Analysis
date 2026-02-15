"""
Image preprocessing operations.
Noise reduction, contrast enhancement, background subtraction, etc.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import Optional, Tuple
from ..core.base import BasePreprocessor


class ImagePreprocessor(BasePreprocessor):
    """
    Comprehensive image preprocessing pipeline.
    Applies multiple operations in sequence.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize preprocessor.
        
        Config options:
            - denoise: bool or dict with method and params
            - enhance_contrast: bool or dict with method and params
            - normalize: bool
            - blur: bool or dict with kernel_size
            - threshold: bool or dict with method and params
            - morphology: dict with operations
        """
        super().__init__(config)
        self.operations = self._setup_operations()
    
    def _setup_operations(self) -> list:
        """Setup preprocessing operations based on config."""
        ops = []
        
        # Noise reduction
        if self.config.get('denoise', False):
            ops.append(self._denoise)
        
        # Contrast enhancement
        if self.config.get('enhance_contrast', False):
            ops.append(self._enhance_contrast)
        
        # Gaussian blur
        if self.config.get('blur', False):
            ops.append(self._blur)
        
        # Normalization
        if self.config.get('normalize', False):
            ops.append(self._normalize)
        
        # Thresholding
        if self.config.get('threshold', False):
            ops.append(self._threshold)
        
        # Morphological operations
        if self.config.get('morphology', False):
            ops.append(self._morphology)
        
        return ops
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply all preprocessing operations.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        processed = frame.copy()
        
        for operation in self.operations:
            processed = operation(processed)
        
        return processed
    
    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        """Apply denoising."""
        config = self.config.get('denoise', {})
        if isinstance(config, bool):
            config = {}
        
        method = config.get('method', 'gaussian')
        
        if method == 'gaussian':
            kernel_size = config.get('kernel_size', 5)
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        elif method == 'median':
            kernel_size = config.get('kernel_size', 5)
            return cv2.medianBlur(frame, kernel_size)
        
        elif method == 'bilateral':
            d = config.get('d', 9)
            sigma_color = config.get('sigma_color', 75)
            sigma_space = config.get('sigma_space', 75)
            return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
        
        elif method == 'nlm':  # Non-local means
            h = config.get('h', 10)
            template_size = config.get('template_size', 7)
            search_size = config.get('search_size', 21)
            if len(frame.shape) == 2:
                return cv2.fastNlMeansDenoising(frame, None, h, 
                                                template_size, search_size)
            else:
                return cv2.fastNlMeansDenoisingColored(frame, None, h, h,
                                                       template_size, search_size)
        
        return frame
    
    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        config = self.config.get('enhance_contrast', {})
        if isinstance(config, bool):
            config = {}
        
        method = config.get('method', 'clahe')
        
        if method == 'clahe':  # Contrast Limited Adaptive Histogram Equalization
            clip_limit = config.get('clip_limit', 2.0)
            tile_size = config.get('tile_size', 8)
            
            if len(frame.shape) == 2:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                                       tileGridSize=(tile_size, tile_size))
                return clahe.apply(frame)
            else:
                # Apply to each channel
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                       tileGridSize=(tile_size, tile_size))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        elif method == 'histogram_eq':
            if len(frame.shape) == 2:
                return cv2.equalizeHist(frame)
            else:
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
        elif method == 'gamma':
            gamma = config.get('gamma', 1.5)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in range(256)]).astype("uint8")
            return cv2.LUT(frame, table)
        
        return frame
    
    def _blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        config = self.config.get('blur', {})
        if isinstance(config, bool):
            config = {}
        
        kernel_size = config.get('kernel_size', 5)
        sigma = config.get('sigma', 0)
        
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        """Normalize pixel values to 0-255 range."""
        return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    
    def _threshold(self, frame: np.ndarray) -> np.ndarray:
        """Apply thresholding."""
        config = self.config.get('threshold', {})
        if isinstance(config, bool):
            config = {}
        
        method = config.get('method', 'otsu')
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        if method == 'otsu':
            _, thresh = cv2.threshold(gray, 0, 255, 
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif method == 'adaptive':
            block_size = config.get('block_size', 11)
            c = config.get('c', 2)
            thresh = cv2.adaptiveThreshold(gray, 255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block_size, c)
        
        elif method == 'manual':
            threshold_value = config.get('value', 127)
            _, thresh = cv2.threshold(gray, threshold_value, 255, 
                                     cv2.THRESH_BINARY)
        
        else:
            thresh = gray
        
        return thresh
    
    def _morphology(self, frame: np.ndarray) -> np.ndarray:
        """Apply morphological operations."""
        config = self.config.get('morphology', {})
        
        operations = config.get('operations', ['opening'])
        kernel_size = config.get('kernel_size', 3)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        result = frame.copy()
        
        for op in operations:
            if op == 'erosion':
                result = cv2.erode(result, kernel, iterations=1)
            elif op == 'dilation':
                result = cv2.dilate(result, kernel, iterations=1)
            elif op == 'opening':
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            elif op == 'closing':
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        return result


class BackgroundSubtractor(BasePreprocessor):
    """
    Background subtraction for detecting moving objects.
    Useful for tracking in surveillance, etc.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        
        method = self.config.get('method', 'mog2')
        
        if method == 'mog2':
            history = self.config.get('history', 500)
            var_threshold = self.config.get('var_threshold', 16)
            detect_shadows = self.config.get('detect_shadows', True)
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows
            )
        
        elif method == 'knn':
            history = self.config.get('history', 500)
            dist2threshold = self.config.get('dist2threshold', 400.0)
            detect_shadows = self.config.get('detect_shadows', True)
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=dist2threshold,
                detectShadows=detect_shadows
            )
        
        else:
            raise ValueError(f"Unknown background subtraction method: {method}")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply background subtraction.
        
        Returns:
            Foreground mask
        """
        return self.subtractor.apply(frame)