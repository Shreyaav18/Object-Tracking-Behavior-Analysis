"""
YOLO-based object detector.
Uses YOLOv8 via ultralytics for detecting people, vehicles, etc.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from .base_detector import BaseDetector, Detection

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO detector unavailable.")


class YOLODetector(BaseDetector):
    """
    Object detector using YOLOv8.
    Can detect 80 COCO classes: person, car, dog, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize YOLO detector.
        
        Config options:
            - model: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
            - conf_threshold: Confidence threshold (0-1)
            - iou_threshold: IoU threshold for NMS (0-1)
            - classes: List of class IDs to detect (None = all)
            - device: 'cpu' or 'cuda'
        """
        super().__init__(config)
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        model_name = self.config.get('model', 'yolov8n.pt')
        self.conf_threshold = self.config.get('conf_threshold', 0.25)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.classes = self.config.get('classes', None)
        self.device = self.config.get('device', 'cpu')
        
        # Load model
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        self.is_initialized = True
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects using YOLO.
        
        Args:
            frame: Input frame (BGR or RGB)
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False
        )[0]
        
        # Convert to Detection objects
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Convert to xywh format
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                bbox = (x, y, w, h)
                
                # Get centroid
                centroid = self._calculate_centroid(bbox)
                
                # Get confidence and class
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = self.model.names[class_id]
                
                detection = Detection(
                    bbox=bbox,
                    centroid=centroid,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                
                detections.append(detection)
        
        return detections


class CustomYOLODetector(BaseDetector):
    """
    YOLO detector with custom trained weights.
    Use this if you've trained YOLO on your own dataset.
    """
    
    def __init__(self, weights_path: str, config: Dict[str, Any] = None):
        """
        Initialize custom YOLO detector.
        
        Args:
            weights_path: Path to custom trained weights (.pt file)
            config: Configuration dictionary
        """
        super().__init__(config)
        
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed")
        
        self.weights_path = weights_path
        self.conf_threshold = self.config.get('conf_threshold', 0.25)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.device = self.config.get('device', 'cpu')
        
        # Load custom model
        self.model = YOLO(weights_path)
        self.model.to(self.device)
        
        self.is_initialized = True
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects using custom YOLO model."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        detections = []
        
        if results.boxes is not None:
            boxes = results.boxes
            
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                x = int(x1)
                y = int(y1)
                w = int(x2 - x1)
                h = int(y2 - y1)
                bbox = (x, y, w, h)
                
                centroid = self._calculate_centroid(bbox)
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = self.model.names[class_id]
                
                detection = Detection(
                    bbox=bbox,
                    centroid=centroid,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                
                detections.append(detection)
        
        return detections