"""
SORT: Simple Online Realtime Tracking
Uses Kalman Filter for motion prediction and Hungarian algorithm for assignment.
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from .base_tracker import BaseTracker, Track
from ..detectors.base_detector import Detection


class KalmanBoxTracker:
    """
    Kalman Filter for tracking bounding boxes.
    State: [x, y, w, h, vx, vy, vw, vh] (position + velocity)
    """
    
    count = 0  # Global counter for unique IDs
    
    def __init__(self, bbox: Tuple[int, int, int, int]):
        """
        Initialize Kalman filter with bounding box.
        
        Args:
            bbox: (x, y, width, height)
        """
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ])
        
        # Measurement matrix (we only observe position, not velocity)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        # Measurement noise
        self.kf.R *= 10.0
        
        # Process noise
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty in velocity
        self.kf.P *= 10.0
        
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """
        Update Kalman filter with new bounding box.
        
        Args:
            bbox: (x, y, width, height)
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
    
    def predict(self) -> np.ndarray:
        """
        Predict next state.
        
        Returns:
            Predicted bounding box [x, y, w, h]
        """
        # Prevent negative width/height
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] = 0
        if self.kf.x[3] + self.kf.x[7] <= 0:
            self.kf.x[7] = 0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self) -> np.ndarray:
        """
        Get current bounding box.
        
        Returns:
            Current bbox [x, y, w, h]
        """
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Convert bbox [x, y, w, h] to measurement [x, y, w, h].
        """
        return np.array([bbox[0], bbox[1], bbox[2], bbox[3]]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
        """
        Convert state x to bbox [x, y, w, h].
        """
        return np.array([x[0], x[1], x[2], x[3]]).flatten()


class SORTTracker(BaseTracker):
    """
    SORT: Simple Online and Realtime Tracking
    Combines Kalman filter prediction with Hungarian algorithm assignment.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize SORT tracker.
        
        Config options:
            - max_age: Maximum frames to keep track alive without detections
            - min_hits: Minimum hits before track is considered confirmed
            - iou_threshold: IOU threshold for matching
        """
        super().__init__(config)
        
        self.max_age = self.config.get('max_age', 30)
        self.min_hits = self.config.get('min_hits', 3)
        self.iou_threshold = self.config.get('iou_threshold', 0.3)
        
        self.kalman_trackers = []
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.kalman_trackers), 4))
        to_del = []
        
        for t, trk in enumerate(trks):
            pos = self.kalman_trackers[t].predict()
            trk[:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # Remove invalid trackers
        for t in reversed(to_del):
            self.kalman_trackers.pop(t)
        
        # Convert detections to bbox array
        dets = np.array([d.bbox for d in detections]) if detections else np.empty((0, 4))
        
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )
        
        # Update matched trackers
        for m in matched:
            self.kalman_trackers[m[0]].update(detections[m[1]].bbox)
        
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i].bbox)
            self.kalman_trackers.append(trk)
        
        # Convert Kalman trackers to Track objects
        self._update_tracks_from_kalman(detections)
        
        # Remove dead tracks
        self._remove_dead_kalman_trackers()
        
        return self.get_tracks(confirmed_only=True)
    
    def _associate_detections_to_trackers(self, detections: np.ndarray, 
                                          trackers: np.ndarray, 
                                          iou_threshold: float = 0.3):
        """
        Assign detections to tracked objects using IOU and Hungarian algorithm.
        
        Returns:
            matched: List of [tracker_idx, detection_idx] pairs
            unmatched_detections: List of detection indices
            unmatched_trackers: List of tracker indices
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk)
        
        # Hungarian algorithm (maximize IOU = minimize negative IOU)
        if min(iou_matrix.shape) > 0:
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # Filter matches by IOU threshold
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                continue
            matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        # Find unmatched detections and trackers
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matches[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matches[:, 1]:
                unmatched_trackers.append(t)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IOU) of two bounding boxes.
        
        Args:
            bbox1, bbox2: [x, y, w, h]
            
        Returns:
            IOU value (0-1)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _update_tracks_from_kalman(self, detections: List[Detection]):
        """Update Track objects from Kalman trackers."""
        # Create mapping from Kalman tracker IDs to Track objects
        track_map = {t.track_id: t for t in self.tracks}
        
        new_tracks = []
        
        for kf_tracker in self.kalman_trackers:
            # Only include confirmed tracks
            if kf_tracker.hit_streak < self.min_hits:
                continue
            
            track_id = kf_tracker.id
            
            # Get or create Track object
            if track_id not in track_map:
                track = Track(track_id=track_id)
                new_tracks.append(track)
                track_map[track_id] = track
            else:
                track = track_map[track_id]
            
            # Update track state
            track.state = "confirmed"
            track.age = kf_tracker.age
            track.hits = kf_tracker.hits
            track.time_since_update = kf_tracker.time_since_update
        
        if new_tracks:
            self.tracks.extend(new_tracks)
    
    def _remove_dead_kalman_trackers(self):
        """Remove Kalman trackers that are too old."""
        self.kalman_trackers = [
            t for t in self.kalman_trackers
            if t.time_since_update < self.max_age
        ]
        
        # Also remove corresponding Track objects
        active_ids = {t.id for t in self.kalman_trackers}
        self.tracks = [t for t in self.tracks if t.track_id in active_ids]