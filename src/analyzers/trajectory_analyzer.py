"""
Trajectory shape analysis: turning angles, straightness, tortuosity, etc.
"""

import numpy as np
from typing import List, Dict, Any
from .base_analyzer import BaseAnalyzer
from ..trackers.base_tracker import Track


class TrajectoryAnalyzer(BaseAnalyzer):
    """
    Analyze trajectory shape and geometry.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.pixel_size = self.config.get('pixel_size', 1.0)
    
    def analyze_single(self, track: Track) -> Dict[str, Any]:
        """
        Analyze trajectory shape for a single track.
        
        Returns:
            Dictionary of trajectory features
        """
        if len(track.trajectory) < 3:
            return {}
        
        trajectory = np.array(track.trajectory)
        features = {}
        
        # Turning angles
        angles = self._calculate_turning_angles(trajectory)
        features['turning_angles'] = angles
        features['mean_turning_angle'] = np.mean(np.abs(angles))
        features['std_turning_angle'] = np.std(angles)
        
        # Straightness index (net displacement / path length)
        net_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
        path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        
        if path_length > 0:
            features['straightness'] = net_displacement / path_length
        else:
            features['straightness'] = 0.0
        
        # Tortuosity (inverse of straightness)
        features['tortuosity'] = 1.0 / features['straightness'] if features['straightness'] > 0 else float('inf')
        
        # Sinuosity (path length / direct distance)
        features['sinuosity'] = path_length / net_displacement if net_displacement > 0 else float('inf')
        
        # Bounding box
        min_x, min_y = np.min(trajectory, axis=0)
        max_x, max_y = np.max(trajectory, axis=0)
        features['bounding_box'] = {
            'width': (max_x - min_x) * self.pixel_size,
            'height': (max_y - min_y) * self.pixel_size,
            'area': (max_x - min_x) * (max_y - min_y) * (self.pixel_size ** 2)
        }
        
        # Convex hull area
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(trajectory)
            features['convex_hull_area'] = hull.volume * (self.pixel_size ** 2)  # volume = area in 2D
        except:
            features['convex_hull_area'] = None
        
        # Asymmetry (difference between forward and backward motion)
        features['asymmetry'] = self._calculate_asymmetry(trajectory)
        
        # Fractal dimension (box-counting)
        features['fractal_dimension'] = self._estimate_fractal_dimension(trajectory)
        
        return features
    
    def analyze(self, tracks: List[Track]) -> Dict[str, Any]:
        """Analyze trajectories for all tracks."""
        results = {'tracks': [], 'summary': {}}
        
        all_straightness = []
        all_turning_angles = []
        
        for track in tracks:
            track_features = self.analyze_single(track)
            results['tracks'].append({
                'track_id': track.track_id,
                'features': track_features
            })
            
            if 'straightness' in track_features:
                all_straightness.append(track_features['straightness'])
            if 'turning_angles' in track_features:
                all_turning_angles.extend(track_features['turning_angles'])
        
        if all_straightness:
            results['summary']['mean_straightness'] = np.mean(all_straightness)
            results['summary']['std_straightness'] = np.std(all_straightness)
        
        if all_turning_angles:
            results['summary']['mean_turning_angle'] = np.mean(np.abs(all_turning_angles))
        
        return results
    
    @staticmethod
    def _calculate_turning_angles(trajectory: np.ndarray) -> np.ndarray:
        """
        Calculate turning angles at each point.
        
        Args:
            trajectory: Array of shape (n_points, 2)
            
        Returns:
            Array of turning angles in radians
        """
        if len(trajectory) < 3:
            return np.array([])
        
        # Calculate vectors
        vectors = np.diff(trajectory, axis=0)
        
        # Calculate angles between consecutive vectors
        angles = []
        for i in range(len(vectors) - 1):
            v1 = vectors[i]
            v2 = vectors[i + 1]
            
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            # Determine sign using cross product
            cross = np.cross(v1, v2)
            if cross < 0:
                angle = -angle
            
            angles.append(angle)
        
        return np.array(angles)
    
    @staticmethod
    def _calculate_asymmetry(trajectory: np.ndarray) -> float:
        """
        Calculate asymmetry between forward and backward motion.
        
        Args:
            trajectory: Array of shape (n_points, 2)
            
        Returns:
            Asymmetry value (0 = symmetric, 1 = completely asymmetric)
        """
        if len(trajectory) < 4:
            return 0.0
        
        # Split trajectory in half
        mid = len(trajectory) // 2
        first_half = trajectory[:mid]
        second_half = trajectory[mid:]
        
        # Calculate distances from centroid
        centroid = np.mean(trajectory, axis=0)
        
        dist_first = np.mean(np.linalg.norm(first_half - centroid, axis=1))
        dist_second = np.mean(np.linalg.norm(second_half - centroid, axis=1))
        
        # Asymmetry
        total_dist = dist_first + dist_second
        if total_dist > 0:
            asymmetry = abs(dist_first - dist_second) / total_dist
        else:
            asymmetry = 0.0
        
        return asymmetry
    
    @staticmethod
    def _estimate_fractal_dimension(trajectory: np.ndarray, max_box_size: int = None) -> float:
        """
        Estimate fractal dimension using box-counting method.
        
        Args:
            trajectory: Array of shape (n_points, 2)
            max_box_size: Maximum box size (None = auto)
            
        Returns:
            Estimated fractal dimension
        """
        if len(trajectory) < 10:
            return 1.0
        
        # Normalize trajectory to unit square
        min_vals = np.min(trajectory, axis=0)
        max_vals = np.max(trajectory, axis=0)
        range_vals = max_vals - min_vals
        
        if np.any(range_vals == 0):
            return 1.0
        
        normalized = (trajectory - min_vals) / range_vals
        
        # Box sizes
        if max_box_size is None:
            max_box_size = len(trajectory) // 4
        
        box_sizes = np.logspace(-2, 0, num=10)  # From 0.01 to 1.0
        counts = []
        
        for box_size in box_sizes:
            # Count boxes containing trajectory points
            grid = {}
            for point in normalized:
                box_idx = tuple((point / box_size).astype(int))
                grid[box_idx] = True
            counts.append(len(grid))
        
        # Fit log-log plot
        log_sizes = np.log(box_sizes)
        log_counts = np.log(counts)
        
        # Linear regression
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        fractal_dim = -coeffs[0]
        
        return fractal_dim