"""
Base classes for all trackers.
Defines standard interfaces and Track object.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
from ..detectors.base_detector import Detection


@dataclass
class Track:
    """
    Represents a tracked object across frames.
    """
    track_id: int                              # Unique track ID
    detections: List[Detection] = field(default_factory=list)  # History of detections
    frames: List[int] = field(default_factory=list)            # Frame numbers
    age: int = 0                               # Frames since first detection
    hits: int = 0                              # Number of successful updates
    time_since_update: int = 0                 # Frames since last detection
    state: str = "tentative"                   # "tentative", "confirmed", "deleted"
    
    def __post_init__(self):
        if not isinstance(self.detections, list):
            self.detections = []
        if not isinstance(self.frames, list):
            self.frames = []
    
    @property
    def is_confirmed(self) -> bool:
        """Check if track is confirmed (not tentative)."""
        return self.state == "confirmed"
    
    @property
    def is_deleted(self) -> bool:
        """Check if track is marked for deletion."""
        return self.state == "deleted"
    
    @property
    def is_tentative(self) -> bool:
        """Check if track is tentative (new, not yet confirmed)."""
        return self.state == "tentative"
    
    @property
    def current_detection(self) -> Detection:
        """Get most recent detection."""
        return self.detections[-1] if self.detections else None
    
    @property
    def current_position(self) -> Tuple[float, float]:
        """Get current centroid position."""
        if self.current_detection:
            return self.current_detection.centroid
        return None
    
    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        """Get full trajectory (list of centroids)."""
        return [det.centroid for det in self.detections]
    
    @property
    def bboxes(self) -> List[Tuple[int, int, int, int]]:
        """Get all bounding boxes."""
        return [det.bbox for det in self.detections]
    
    def update(self, detection: Detection, frame_num: int):
        """
        Update track with new detection.
        
        Args:
            detection: New detection
            frame_num: Current frame number
        """
        self.detections.append(detection)
        self.frames.append(frame_num)
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        
        # Confirm track after sufficient hits
        if self.hits >= 3 and self.state == "tentative":
            self.state = "confirmed"
    
    def mark_missed(self):
        """Mark that detection was missed in current frame."""
        self.time_since_update += 1
        self.age += 1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            'track_id': self.track_id,
            'trajectory': self.trajectory,
            'frames': self.frames,
            'age': self.age,
            'hits': self.hits,
            'state': self.state,
            'detections': [det.to_dict() for det in self.detections]
        }


class BaseTracker(ABC):
    """
    Abstract base class for all trackers.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.tracks: List[Track] = []
        self.next_track_id = 0
        self.frame_count = 0
    
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of active tracks
        """
        pass
    
    def get_tracks(self, confirmed_only: bool = False) -> List[Track]:
        """
        Get current tracks.
        
        Args:
            confirmed_only: If True, return only confirmed tracks
            
        Returns:
            List of tracks
        """
        if confirmed_only:
            return [t for t in self.tracks if t.is_confirmed]
        return self.tracks
    
    def _create_track(self, detection: Detection) -> Track:
        """Create new track from detection."""
        track = Track(track_id=self.next_track_id)
        track.update(detection, self.frame_count)
        self.next_track_id += 1
        return track
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_track_id = 0
        self.frame_count = 0