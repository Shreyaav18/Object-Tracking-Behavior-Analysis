"""
Motion analysis: velocity, acceleration, displacement, MSD, etc.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from .base_analyzer import BaseAnalyzer
from ..trackers.base_tracker import Track


class MotionAnalyzer(BaseAnalyzer):
    """
    Analyze motion characteristics of tracks.
    Calculates velocity, acceleration, displacement, MSD, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize motion analyzer.
        
        Config options:
            - fps: Frames per second (for time calculations)
            - pixel_size: Pixel size in microns (for real-world units)
            - smooth_window: Window size for smoothing (None = no smoothing)
        """
        super().__init__(config)
        
        self.fps = self.config.get('fps', 30.0)
        self.pixel_size = self.config.get('pixel_size', 1.0)  # microns per pixel
        self.smooth_window = self.config.get('smooth_window', None)
    
    def analyze(self, tracks: List[Track]) -> Dict[str, Any]:
        """
        Analyze motion for all tracks.
        
        Returns:
            Dictionary with per-track and aggregate statistics
        """
        results = {
            'tracks': [],
            'summary': {}
        }
        
        all_velocities = []
        all_displacements = []
        
        for track in tracks:
            track_features = self.analyze_single(track)
            results['tracks'].append({
                'track_id': track.track_id,
                'features': track_features
            })
            
            if 'velocities' in track_features:
                all_velocities.extend(track_features['velocities'])
            if 'total_displacement' in track_features:
                all_displacements.append(track_features['total_displacement'])
        
        # Aggregate statistics
        if all_velocities:
            results['summary']['mean_velocity'] = np.mean(all_velocities)
            results['summary']['std_velocity'] = np.std(all_velocities)
            results['summary']['median_velocity'] = np.median(all_velocities)
        
        if all_displacements:
            results['summary']['mean_displacement'] = np.mean(all_displacements)
            results['summary']['std_displacement'] = np.std(all_displacements)
        
        return results
    
    def analyze_single(self, track: Track) -> Dict[str, Any]:
        """
        Analyze motion for a single track.
        
        Returns:
            Dictionary of motion features
        """
        if len(track.trajectory) < 2:
            return {}
        
        trajectory = np.array(track.trajectory)
        
        # Calculate time array
        dt = 1.0 / self.fps
        times = np.arange(len(trajectory)) * dt
        
        # Smooth trajectory if requested
        if self.smooth_window and len(trajectory) > self.smooth_window:
            trajectory = self._smooth_trajectory(trajectory)
        
        features = {}
        
        # Displacement (frame-to-frame)
        displacements = self._calculate_displacements(trajectory)
        features['displacements'] = displacements * self.pixel_size
        
        # Total displacement (start to end)
        total_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
        features['total_displacement'] = total_displacement * self.pixel_size
        
        # Path length (total distance traveled)
        path_length = np.sum(displacements)
        features['path_length'] = path_length * self.pixel_size
        
        # Velocity
        velocities = displacements / dt
        features['velocities'] = velocities * self.pixel_size
        features['mean_velocity'] = np.mean(velocities) * self.pixel_size
        features['max_velocity'] = np.max(velocities) * self.pixel_size
        features['min_velocity'] = np.min(velocities) * self.pixel_size
        
        # Acceleration
        if len(velocities) > 1:
            accelerations = np.diff(velocities) / dt
            features['accelerations'] = accelerations * self.pixel_size
            features['mean_acceleration'] = np.mean(np.abs(accelerations)) * self.pixel_size
        
        # Mean Squared Displacement (MSD)
        msd = self._calculate_msd(trajectory)
        features['msd'] = msd * (self.pixel_size ** 2)
        
        # Directionality (net displacement / path length)
        if path_length > 0:
            directionality = total_displacement / path_length
            features['directionality'] = directionality
        else:
            features['directionality'] = 0.0
        
        # Confinement ratio (radius of gyration / path length)
        radius_of_gyration = self._calculate_radius_of_gyration(trajectory)
        features['radius_of_gyration'] = radius_of_gyration * self.pixel_size
        if path_length > 0:
            features['confinement_ratio'] = radius_of_gyration / path_length
        
        return features
    
    @staticmethod
    def _calculate_displacements(trajectory: np.ndarray) -> np.ndarray:
        """
        Calculate frame-to-frame displacements.
        
        Args:
            trajectory: Array of shape (n_points, 2)
            
        Returns:
            Array of displacements (length n_points - 1)
        """
        diffs = np.diff(trajectory, axis=0)
        displacements = np.linalg.norm(diffs, axis=1)
        return displacements
    
    @staticmethod
    def _calculate_msd(trajectory: np.ndarray, max_tau: int = None) -> np.ndarray:
        """
        Calculate Mean Squared Displacement.
        
        Args:
            trajectory: Array of shape (n_points, 2)
            max_tau: Maximum time lag (None = half of trajectory length)
            
        Returns:
            Array of MSD values for each time lag
        """
        n = len(trajectory)
        if max_tau is None:
            max_tau = n // 2
        
        msd = np.zeros(max_tau)
        
        for tau in range(1, max_tau):
            displacements = []
            for i in range(n - tau):
                displacement = np.linalg.norm(trajectory[i + tau] - trajectory[i])
                displacements.append(displacement ** 2)
            msd[tau] = np.mean(displacements)
        
        return msd
    
    @staticmethod
    def _calculate_radius_of_gyration(trajectory: np.ndarray) -> float:
        """
        Calculate radius of gyration (spread of trajectory around centroid).
        
        Args:
            trajectory: Array of shape (n_points, 2)
            
        Returns:
            Radius of gyration
        """
        centroid = np.mean(trajectory, axis=0)
        squared_distances = np.sum((trajectory - centroid) ** 2, axis=1)
        return np.sqrt(np.mean(squared_distances))
    
    @staticmethod
    def _smooth_trajectory(trajectory: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Smooth trajectory using moving average.
        
        Args:
            trajectory: Array of shape (n_points, 2)
            window_size: Size of smoothing window
            
        Returns:
            Smoothed trajectory
        """
        from scipy.ndimage import uniform_filter1d
        
        smoothed = np.zeros_like(trajectory)
        smoothed[:, 0] = uniform_filter1d(trajectory[:, 0], size=window_size, mode='nearest')
        smoothed[:, 1] = uniform_filter1d(trajectory[:, 1], size=window_size, mode='nearest')
        
        return smoothed
    
    def fit_diffusion_model(self, track: Track) -> Dict[str, float]:
        """
        Fit diffusion model to MSD curve: MSD = 4*D*t^alpha
        
        Args:
            track: Track object
            
        Returns:
            Dictionary with diffusion coefficient and alpha
        """
        features = self.analyze_single(track)
        msd = features.get('msd')
        
        if msd is None or len(msd) < 3:
            return {'diffusion_coefficient': None, 'alpha': None}
        
        # Time points
        dt = 1.0 / self.fps
        t = np.arange(1, len(msd)) * dt
        msd = msd[1:]  # Skip tau=0
        
        # Fit log(MSD) = log(4D) + alpha * log(t)
        log_t = np.log(t)
        log_msd = np.log(msd + 1e-10)  # Add small value to avoid log(0)
        
        # Linear regression
        coeffs = np.polyfit(log_t, log_msd, 1)
        alpha = coeffs[0]
        log_4D = coeffs[1]
        D = np.exp(log_4D) / 4.0
        
        return {
            'diffusion_coefficient': D,
            'alpha': alpha,
            'motion_type': self._classify_motion_type(alpha)
        }
    
    @staticmethod
    def _classify_motion_type(alpha: float) -> str:
        """
        Classify motion type based on alpha value.
        
        Args:
            alpha: Exponent from MSD fit
            
        Returns:
            Motion type classification
        """
        if alpha < 0.9:
            return 'subdiffusive'  # Confined/anomalous
        elif 0.9 <= alpha <= 1.1:
            return 'brownian'  # Normal diffusion
        else:
            return 'superdiffusive'  # Directed/active transport