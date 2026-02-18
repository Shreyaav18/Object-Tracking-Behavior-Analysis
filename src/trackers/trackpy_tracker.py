"""
Wrapper around TrackPy library for particle tracking.
Specialized for cells, particles, and point-like objects.
"""

import numpy as np
import pandas as pd
from typing import List
from .base_tracker import BaseTracker, Track
from ..detectors.base_detector import Detection

try:
    import trackpy as tp
    TRACKPY_AVAILABLE = True
except ImportError:
    TRACKPY_AVAILABLE = False
    print("Warning: trackpy not installed. Run: pip install trackpy")


class TrackPyTracker(BaseTracker):
    """
    Tracker using TrackPy library.
    Excellent for particle/cell tracking with established algorithms.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize TrackPy tracker.
        
        Config options:
            - search_range: Maximum displacement between frames
            - memory: Frames to remember lost particles
            - adaptive_stop: Adaptive search range reduction
            - adaptive_step: Step size for adaptive search
        """
        super().__init__(config)
        
        if not TRACKPY_AVAILABLE:
            raise ImportError("trackpy not installed")
        
        self.search_range = self.config.get('search_range', 15)
        self.memory = self.config.get('memory', 3)
        self.adaptive_stop = self.config.get('adaptive_stop', None)
        self.adaptive_step = self.config.get('adaptive_step', 0.95)
        
        # Store detection history for batch linking
        self.detection_history = []
        self.frame_detections = {}  # frame_num -> detections
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracks with new detections.
        
        Note: TrackPy works best in batch mode, but we provide
        online tracking by maintaining a sliding window.
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Store detections for this frame
        self.frame_detections[self.frame_count] = detections
        
        # Convert detections to DataFrame format expected by TrackPy
        if len(detections) == 0:
            return self.get_tracks()
        
        # Build DataFrame from detection history
        df = self._build_dataframe()
        
        # Link particles
        if len(df) > 0:
            linked = tp.link(
                df,
                search_range=self.search_range,
                memory=self.memory,
                adaptive_stop=self.adaptive_stop,
                adaptive_step=self.adaptive_step
            )
            
            # Convert linked DataFrame back to Track objects
            self._update_tracks_from_dataframe(linked)
        
        return self.get_tracks()
    
    def _build_dataframe(self) -> pd.DataFrame:
        """
        Build DataFrame from detection history.
        
        Returns:
            DataFrame with columns: x, y, frame, (optional: mass, size, etc.)
        """
        rows = []
        
        for frame_num, detections in self.frame_detections.items():
            for det in detections:
                row = {
                    'x': det.centroid[0],
                    'y': det.centroid[1],
                    'frame': frame_num,
                }
                
                # Add optional features
                if det.features:
                    if 'area' in det.features:
                        row['mass'] = det.features['area']
                    if 'radius' in det.features:
                        row['size'] = det.features['radius']
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _update_tracks_from_dataframe(self, linked_df: pd.DataFrame):
        """
        Update Track objects from linked DataFrame.
        
        Args:
            linked_df: DataFrame with 'particle' column from tp.link()
        """
        # Group by particle ID
        grouped = linked_df.groupby('particle')
        
        # Create/update tracks
        new_tracks = []
        
        for particle_id, group in grouped:
            # Sort by frame
            group = group.sort_values('frame')
            
            # Check if track already exists
            existing_track = None
            for track in self.tracks:
                if track.track_id == particle_id:
                    existing_track = track
                    break
            
            if existing_track is None:
                # Create new track
                track = Track(track_id=int(particle_id))
                new_tracks.append(track)
            else:
                track = existing_track
            
            # Add detections to track
            for _, row in group.iterrows():
                frame_num = int(row['frame'])
                
                # Find corresponding detection
                frame_detections = self.frame_detections.get(frame_num, [])
                
                # Find detection closest to this position
                pos = (row['x'], row['y'])
                min_dist = float('inf')
                best_det = None
                
                for det in frame_detections:
                    dist = np.sqrt((det.centroid[0] - pos[0])**2 + 
                                 (det.centroid[1] - pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_det = det
                
                if best_det and frame_num not in track.frames:
                    track.update(best_det, frame_num)
        
        # Update tracks list
        if new_tracks:
            self.tracks.extend(new_tracks)
            self.next_track_id = max(t.track_id for t in self.tracks) + 1
    
    def link_batch(self, all_detections: List[List[Detection]]) -> List[Track]:
        """
        Link detections across all frames in batch mode.
        More accurate than online mode.
        
        Args:
            all_detections: List of detection lists (one per frame)
            
        Returns:
            List of tracks
        """
        # Reset
        self.reset()
        self.frame_detections = {}
        
        # Store all detections
        for frame_num, detections in enumerate(all_detections):
            self.frame_detections[frame_num] = detections
        
        # Build DataFrame
        df = self._build_dataframe()
        
        if len(df) == 0:
            return []
        
        # Link all at once
        linked = tp.link(
            df,
            search_range=self.search_range,
            memory=self.memory,
            adaptive_stop=self.adaptive_stop,
            adaptive_step=self.adaptive_step
        )
        
        # Convert to tracks
        self._update_tracks_from_dataframe(linked)
        
        return self.get_tracks()