"""
Simple tracker using nearest neighbor matching.
Fast and effective for non-overlapping objects with slow motion.
"""

import numpy as np
from typing import List
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from .base_tracker import BaseTracker, Track
from ..detectors.base_detector import Detection


class SimpleTracker(BaseTracker):
    """
    Nearest neighbor tracker with Hungarian algorithm for optimal assignment.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize simple tracker.
        
        Config options:
            - max_distance: Maximum distance for matching (pixels)
            - max_age: Maximum frames to keep track without detection
            - min_hits: Minimum hits before confirming track
        """
        super().__init__(config)
        
        self.max_distance = self.config.get('max_distance', 50)
        self.max_age = self.config.get('max_age', 30)
        self.min_hits = self.config.get('min_hits', 3)
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Handle empty cases
        if len(self.tracks) == 0:
            # No existing tracks - create new tracks for all detections
            for detection in detections:
                track = self._create_track(detection)
                self.tracks.append(track)
            return self.get_tracks()
        
        if len(detections) == 0:
            # No detections - mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()
            self._remove_old_tracks()
            return self.get_tracks()
        
        # Calculate distance matrix
        distance_matrix = self._calculate_distances(detections)
        
        # Perform assignment
        matched_indices, unmatched_tracks, unmatched_detections = \
            self._assign_detections_to_tracks(distance_matrix)
        
        # Update matched tracks
        for track_idx, det_idx in matched_indices:
            self.tracks[track_idx].update(detections[det_idx], self.frame_count)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            track = self._create_track(detections[det_idx])
            self.tracks.append(track)
        
        # Remove old tracks
        self._remove_old_tracks()
        
        return self.get_tracks()
    
    def _calculate_distances(self, detections: List[Detection]) -> np.ndarray:
        """
        Calculate distance matrix between tracks and detections.
        
        Args:
            detections: List of current detections
            
        Returns:
            Distance matrix of shape (num_tracks, num_detections)
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return np.array([])
        
        # Get track positions (last known position)
        track_positions = np.array([t.current_position for t in self.tracks])
        
        # Get detection positions
        detection_positions = np.array([d.centroid for d in detections])
        
        # Calculate Euclidean distances
        distances = cdist(track_positions, detection_positions, metric='euclidean')
        
        return distances
    
    def _assign_detections_to_tracks(self, distance_matrix: np.ndarray):
        """
        Assign detections to tracks using Hungarian algorithm.
        
        Args:
            distance_matrix: Distance matrix (tracks x detections)
            
        Returns:
            matched_indices: List of (track_idx, detection_idx) pairs
            unmatched_tracks: List of unmatched track indices
            unmatched_detections: List of unmatched detection indices
        """
        if distance_matrix.size == 0:
            return [], list(range(len(self.tracks))), list(range(0))
        
        # Apply distance threshold
        distance_matrix = distance_matrix.copy()
        distance_matrix[distance_matrix > self.max_distance] = 1e6  # Large value
        
        # Hungarian algorithm
        track_indices, detection_indices = linear_sum_assignment(distance_matrix)
        
        # Filter out assignments with large distance
        matched_indices = []
        for t_idx, d_idx in zip(track_indices, detection_indices):
            if distance_matrix[t_idx, d_idx] < self.max_distance:
                matched_indices.append((t_idx, d_idx))
        
        # Find unmatched tracks and detections
        matched_track_indices = [t for t, _ in matched_indices]
        matched_detection_indices = [d for _, d in matched_indices]
        
        unmatched_tracks = [i for i in range(len(self.tracks)) 
                          if i not in matched_track_indices]
        unmatched_detections = [i for i in range(distance_matrix.shape[1])
                               if i not in matched_detection_indices]
        
        return matched_indices, unmatched_tracks, unmatched_detections
    
    def _remove_old_tracks(self):
        """Remove tracks that haven't been updated for max_age frames."""
        self.tracks = [t for t in self.tracks 
                      if t.time_since_update < self.max_age]