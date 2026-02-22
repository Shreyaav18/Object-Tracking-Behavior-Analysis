"""
Data loading utilities for videos, image sequences, and single images.
Supports multiple formats: mp4, avi, tif stacks, image folders, etc.
"""

import os
from pathlib import Path
from typing import Union, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


class DataLoader:
    """
    Universal data loader for videos and images.
    Handles multiple input formats.
    """
    
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    def __init__(self, source: str | Path, 
                 max_frames: Optional[int] = None,
                 start_frame: int = 0,
                 grayscale: bool = False,
                 resize: Optional[Tuple[int, int]] = None):
        """
        Initialize data loader.
        
        Args:
            source: Path to video file, image file, or directory of images
            max_frames: Maximum number of frames to load (None = all)
            start_frame: Frame index to start from
            grayscale: Convert to grayscale
            resize: Resize to (width, height), None keeps original
        """
        self.source = Path(source)
        self.max_frames = max_frames
        self.start_frame = start_frame
        self.grayscale = grayscale
        self.resize = resize
        
        self.source_type = self._detect_source_type()
        
    def _detect_source_type(self) -> str:
        """Detect if source is video, image, or directory."""
        if not self.source.exists():
            raise FileNotFoundError(f"Source not found: {self.source}")
        
        if self.source.is_dir():
            return 'directory'
        
        suffix = self.source.suffix.lower()
        if suffix in self.SUPPORTED_VIDEO_FORMATS:
            return 'video'
        elif suffix in self.SUPPORTED_IMAGE_FORMATS:
            return 'image'
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def load(self) -> Tuple[List[np.ndarray], dict]:
        """
        Load data based on source type.
        
        Returns:
            frames: List of frames as numpy arrays
            metadata: Dictionary with fps, resolution, etc.
        """
        if self.source_type == 'video':
            return self._load_video()
        elif self.source_type == 'directory':
            return self._load_image_sequence()
        else:  # single image
            return self._load_single_image()
    
    def _load_video(self) -> Tuple[List[np.ndarray], dict]:
        """Load video file."""
        cap = cv2.VideoCapture(str(self.source))
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.source}")
        
        # Get metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata = {
            'fps': fps,
            'total_frames': total_frames,
            'resolution': (width, height),
            'source': str(self.source),
            'source_type': 'video'
        }
        
        # Set start frame
        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        # Determine how many frames to load
        frames_to_load = total_frames - self.start_frame
        if self.max_frames:
            frames_to_load = min(frames_to_load, self.max_frames)
        
        # Load frames
        frames = []
        pbar = tqdm(total=frames_to_load, desc="Loading video")
        
        while len(frames) < frames_to_load:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame = self._process_frame(frame)
            frames.append(frame)
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        print(f"Loaded {len(frames)} frames from video")
        return frames, metadata
    
    def _load_image_sequence(self) -> Tuple[List[np.ndarray], dict]:
        """Load sequence of images from directory."""
        # Get all image files
        image_files = []
        for ext in self.SUPPORTED_IMAGE_FORMATS:
            image_files.extend(self.source.glob(f'*{ext}'))
        
        image_files = sorted(image_files)  # Sort alphabetically
        
        if not image_files:
            raise ValueError(f"No images found in {self.source}")
        
        # Apply start_frame and max_frames
        image_files = image_files[self.start_frame:]
        if self.max_frames:
            image_files = image_files[:self.max_frames]
        
        # Load images
        frames = []
        for img_path in tqdm(image_files, desc="Loading images"):
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"Warning: Could not load {img_path}")
                continue
            frame = self._process_frame(frame)
            frames.append(frame)
        
        metadata = {
            'fps': 30.0,  # Default, can be overridden
            'total_frames': len(frames),
            'resolution': frames[0].shape[:2][::-1] if frames else None,
            'source': str(self.source),
            'source_type': 'image_sequence'
        }
        
        print(f"Loaded {len(frames)} images from directory")
        return frames, metadata
    
    def _load_single_image(self) -> Tuple[List[np.ndarray], dict]:
        """Load a single image."""
        frame = cv2.imread(str(self.source))
        if frame is None:
            raise IOError(f"Cannot load image: {self.source}")
        
        frame = self._process_frame(frame)
        
        metadata = {
            'fps': None,
            'total_frames': 1,
            'resolution': frame.shape[:2][::-1],
            'source': str(self.source),
            'source_type': 'single_image'
        }
        
        return [frame], metadata
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing to frame."""
        # Grayscale conversion
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize
        if self.resize:
            frame = cv2.resize(frame, self.resize)
        
        return frame