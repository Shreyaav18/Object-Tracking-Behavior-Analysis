"""
Statistical analysis and hypothesis testing.
"""

import numpy as np
from typing import List, Dict, Any
from .base_analyzer import BaseAnalyzer
from ..trackers.base_tracker import Track

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class StatisticalAnalyzer(BaseAnalyzer):
    """
    Perform statistical tests and generate summary statistics.
    """
    
    def analyze(self, tracks: List[Track]) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for tracks.
        
        Returns:
            Dictionary of statistical summaries
        """
        if not tracks:
            return {}
        
        results = {
            'n_tracks': len(tracks),
            'track_lengths': [],
            'duration': [],
        }
        
        for track in tracks:
            results['track_lengths'].append(len(track.trajectory))
            results['duration'].append(track.age)
        
        # Summary statistics
        results['length_stats'] = self._describe(results['track_lengths'])
        results['duration_stats'] = self._describe(results['duration'])
        
        return results
    
    @staticmethod
    def _describe(data: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics."""
        if not data:
            return {}
        
        arr = np.array(data)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'q25': float(np.percentile(arr, 25)),
            'q75': float(np.percentile(arr, 75)),
        }
    
    def compare_groups(self, tracks_a: List[Track], tracks_b: List[Track], 
                      feature: str = 'velocity') -> Dict[str, Any]:
        """
        Compare two groups of tracks statistically.
        
        Args:
            tracks_a: First group
            tracks_b: Second group
            feature: Feature to compare
            
        Returns:
            Dictionary with test results
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy not installed")
        
        # Extract feature values
        from .motion_analyzer import MotionAnalyzer
        analyzer = MotionAnalyzer()
        
        values_a = []
        values_b = []
        
        for track in tracks_a:
            features = analyzer.analyze_single(track)
            if f'mean_{feature}' in features:
                values_a.append(features[f'mean_{feature}'])
        
        for track in tracks_b:
            features = analyzer.analyze_single(track)
            if f'mean_{feature}' in features:
                values_b.append(features[f'mean_{feature}'])
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(values_a, values_b)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(values_a, values_b)
        
        return {
            'feature': feature,
            'group_a': self._describe(values_a),
            'group_b': self._describe(values_b),
            't_test': {'statistic': float(t_stat), 'p_value': float(p_value)},
            'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p_value)},
            'significant': p_value < 0.05
        }