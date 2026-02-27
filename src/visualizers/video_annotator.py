"""
Annotate videos with tracking results.
Draw bounding boxes, trajectories, IDs, etc.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
from .base_visualizer import BaseVisualizer
from ..trackers.base_tracker import Track
from ..detectors.base_detector import Detection


class VideoAnnotator(BaseVisualizer):
    """
    Annotate video frames with tracking results.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize video annotator.
        
        Config options:
            - show_bbox: Show bounding boxes
            - show_id: Show track IDs
            - show_trajectory: Show trajectory trails
            - trail_length: Length of trajectory trail (frames)
            - bbox_color: Color for bounding boxes (BGR)
            - text_color: Color for text (BGR)
            - trajectory_color: Color for trajectories (BGR)
            - thickness: Line thickness
            - font_scale: Text size
        """
        super().__init__(config)
        
        self.show_bbox = self.config.get('show_bbox', True)
        self.show_id = self.config.get('show_id', True)
        self.show_trajectory = self.config.get('show_trajectory', True)
        self.trail_length = self.config.get('trail_length', 30)
        self.bbox_color = self.config.get('bbox_color', (0, 255, 0))  # Green
        self.text_color = self.config.get('text_color', (255, 255, 255))  # White
        self.trajectory_color = self.config.get('trajectory_color', (0, 0, 255))  # Red
        self.thickness = self.config.get('thickness', 2)
        self.font_scale = self.config.get('font_scale', 0.5)
        
        # Color palette for different tracks
        self.colors = self._generate_colors(100)
    
    def annotate_frame(self, frame: np.ndarray, tracks: List[Track], 
                      frame_num: int = None) -> np.ndarray:
        """
        Annotate a single frame.
        
        Args:
            frame: Input frame
            tracks: List of active tracks
            frame_num: Current frame number
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for track in tracks:
            if not track.is_confirmed:
                continue
            
            color = self.colors[track.track_id % len(self.colors)]
            
            # Draw trajectory
            if self.show_trajectory and len(track.trajectory) > 1:
                points = track.trajectory[-self.trail_length:]
                for i in range(1, len(points)):
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    cv2.line(annotated, pt1, pt2, color, self.thickness)
            
            # Get current detection
            current_det = track.current_detection
            if current_det is None:
                continue
            
            # Draw bounding box
            if self.show_bbox:
                x, y, w, h = current_det.bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), 
                            color, self.thickness)
            
            # Draw ID
            if self.show_id:
                cx, cy = current_det.centroid
                text = f"ID: {track.track_id}"
                
                # Background for text
                (text_w, text_h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
                )
                cv2.rectangle(annotated, 
                            (int(cx) - 5, int(cy) - text_h - 10),
                            (int(cx) + text_w + 5, int(cy) - 5),
                            color, -1)
                
                # Text
                cv2.putText(annotated, text, (int(cx), int(cy) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                          self.text_color, 1)
        
        # Add frame number
        if frame_num is not None:
            cv2.putText(annotated, f"Frame: {frame_num}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, self.text_color, 2)
        
        # Add track count
        n_tracks = sum(1 for t in tracks if t.is_confirmed)
        cv2.putText(annotated, f"Tracks: {n_tracks}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, self.text_color, 2)
        
        return annotated
    
    def annotate_video(self, frames: List[np.ndarray], 
                      all_tracks: List[List[Track]],
                      output_path: str) -> str:
        """
        Annotate entire video and save.
        
        Args:
            frames: List of video frames
            all_tracks: List of track lists (one per frame)
            output_path: Path to save annotated video
            
        Returns:
            Path to saved video
        """
        if len(frames) == 0:
            raise ValueError("No frames to annotate")
        
        if len(frames) != len(all_tracks):
            raise ValueError("Number of frames must match number of track lists")
        
        # Setup video writer
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.config.get('fps', 30)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Annotate each frame
        for i, (frame, tracks) in enumerate(tqdm(zip(frames, all_tracks), 
                                                  total=len(frames),
                                                  desc="Annotating video")):
            annotated = self.annotate_frame(frame, tracks, frame_num=i)
            writer.write(annotated)
        
        writer.release()
        print(f"Annotated video saved to: {output_path}")
        
        return str(output_path)
    
    def visualize(self, frames: List[np.ndarray], 
                 all_tracks: List[List[Track]],
                 output_path: str) -> str:
        """Alias for annotate_video."""
        return self.annotate_video(frames, all_tracks, output_path)
    
    @staticmethod
    def _generate_colors(n: int) -> List[Tuple[int, int, int]]:
        """
        Generate n distinct colors.
        
        Args:
            n: Number of colors
            
        Returns:
            List of BGR color tuples
        """
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            # Convert HSV to BGR
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, bgr)))
        return colors


class DetectionAnnotator(BaseVisualizer):
    """
    Annotate frames with detection results (before tracking).
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        self.bbox_color = self.config.get('bbox_color', (255, 0, 0))  # Blue
        self.centroid_color = self.config.get('centroid_color', (0, 255, 255))  # Yellow
        self.thickness = self.config.get('thickness', 2)
    
    def annotate_frame(self, frame: np.ndarray, 
                      detections: List[Detection]) -> np.ndarray:
        """
        Annotate frame with detections.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            # Draw bounding box
            x, y, w, h = det.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h),
                        self.bbox_color, self.thickness)
            
            # Draw centroid
            cx, cy = det.centroid
            cv2.circle(annotated, (int(cx), int(cy)), 3,
                      self.centroid_color, -1)
            
            # Draw confidence if available
            if det.confidence < 1.0:
                text = f"{det.confidence:.2f}"
                cv2.putText(annotated, text, (x, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                          (255, 255, 255), 1)
        
        # Add detection count
        cv2.putText(annotated, f"Detections: {len(detections)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)
        
        return annotated
    
    def visualize(self, frames: List[np.ndarray],
                 all_detections: List[List[Detection]],
                 output_path: str) -> str:
        """Save annotated video with detections."""
        if len(frames) == 0:
            raise ValueError("No frames provided")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.config.get('fps', 30)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame, detections in tqdm(zip(frames, all_detections),
                                     total=len(frames),
                                     desc="Annotating detections"):
            annotated = self.annotate_frame(frame, detections)
            writer.write(annotated)
        
        writer.release()
        print(f"Detection video saved to: {output_path}")
        
        return str(output_path)