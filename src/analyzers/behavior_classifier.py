"""
Machine learning-based behavior classification.
Classifies movement patterns: random walk, directed motion, confined, etc.
"""

import numpy as np
from typing import List, Dict, Any
from .base_analyzer import BaseAnalyzer
from .motion_analyzer import MotionAnalyzer
from .trajectory_analyzer import TrajectoryAnalyzer
from ..trackers.base_tracker import Track

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Run: pip install scikit-learn")


class BehaviorClassifier(BaseAnalyzer):
    """
    Classify track behaviors using machine learning.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize behavior classifier.
        
        Config options:
            - method: 'supervised' or 'unsupervised'
            - n_clusters: Number of clusters for unsupervised (default: 3)
            - model: Pre-trained model (optional)
        """
        super().__init__(config)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed")
        
        self.method = self.config.get('method', 'unsupervised')
        self.n_clusters = self.config.get('n_clusters', 3)
        self.model = self.config.get('model', None)
        self.scaler = StandardScaler()
        
        self.motion_analyzer = MotionAnalyzer(config)
        self.trajectory_analyzer = TrajectoryAnalyzer(config)
        
        # Initialize model
        if self.model is None:
            if self.method == 'unsupervised':
                self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def analyze(self, tracks: List[Track]) -> Dict[str, Any]:
        """
        Classify behaviors for all tracks.
        
        Returns:
            Dictionary with classifications and features
        """
        if len(tracks) == 0:
            return {'tracks': [], 'summary': {}}
        
        # Extract features for all tracks
        features_list = []
        track_ids = []
        
        for track in tracks:
            if len(track.trajectory) < 5:  # Need minimum length
                continue
            
            features = self._extract_features(track)
            features_list.append(features)
            track_ids.append(track.track_id)
        
        if len(features_list) == 0:
            return {'tracks': [], 'summary': {}}
        
        # Convert to array
        X = np.array(features_list)
        
        # Classify
        if self.method == 'unsupervised':
            labels = self._cluster(X)
            behavior_names = [f'cluster_{i}' for i in labels]
        else:
            labels = self.model.predict(X)
            behavior_names = [self._get_behavior_name(label) for label in labels]
        
        # Prepare results
        results = {
            'tracks': [],
            'summary': {}
        }
        
        for i, track_id in enumerate(track_ids):
            results['tracks'].append({
                'track_id': track_id,
                'behavior': behavior_names[i],
                'label': int(labels[i]),
                'features': features_list[i].tolist()
            })
        
        # Summary statistics
        unique_behaviors, counts = np.unique(behavior_names, return_counts=True)
        results['summary']['behavior_distribution'] = {
            behavior: int(count) 
            for behavior, count in zip(unique_behaviors, counts)
        }
        
        return results
    
    def _extract_features(self, track: Track) -> np.ndarray:
        """
        Extract feature vector for a track.
        
        Returns:
            Feature vector as numpy array
        """
        # Get motion features
        motion_features = self.motion_analyzer.analyze_single(track)
        diffusion = self.motion_analyzer.fit_diffusion_model(track)
        
        # Get trajectory features
        traj_features = self.trajectory_analyzer.analyze_single(track)
        
        # Build feature vector
        features = [
            motion_features.get('mean_velocity', 0),
            motion_features.get('max_velocity', 0),
            motion_features.get('directionality', 0),
            motion_features.get('path_length', 0),
            motion_features.get('total_displacement', 0),
            diffusion.get('alpha', 1.0),
            diffusion.get('diffusion_coefficient', 0),
            traj_features.get('straightness', 0),
            traj_features.get('mean_turning_angle', 0),
            traj_features.get('fractal_dimension', 1.0),
        ]
        
        return np.array(features)
    
    def _cluster(self, X: np.ndarray) -> np.ndarray:
        """
        Perform unsupervised clustering.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cluster
        labels = self.model.fit_predict(X_scaled)
        
        return labels
    
    def train(self, tracks: List[Track], labels: List[int]):
        """
        Train supervised classifier.
        
        Args:
            tracks: List of tracks
            labels: Ground truth labels
        """
        if self.method != 'supervised':
            raise ValueError("Can only train in supervised mode")
        
        # Extract features
        features_list = []
        valid_labels = []
        
        for track, label in zip(tracks, labels):
            if len(track.trajectory) < 5:
                continue
            features = self._extract_features(track)
            features_list.append(features)
            valid_labels.append(label)
        
        X = np.array(features_list)
        y = np.array(valid_labels)
        
        # Normalize and train
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    @staticmethod
    def _get_behavior_name(label: int) -> str:
        """Map numeric label to behavior name."""
        behavior_map = {
            0: 'random_walk',
            1: 'directed_motion',
            2: 'confined',
            3: 'mixed'
        }
        return behavior_map.get(label, f'behavior_{label}')