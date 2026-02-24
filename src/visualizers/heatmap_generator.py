"""
Generate heatmaps showing spatial density and motion patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import List, Tuple
from pathlib import Path
from .base_visualizer import BaseVisualizer
from ..trackers.base_tracker import Track


class HeatmapGenerator(BaseVisualizer):
    """
    Generate spatial heatmaps from tracking data.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        
        self.figsize = self.config.get('figsize', (10, 8))
        self.dpi = self.config.get('dpi', 100)
        self.cmap = self.config.get('cmap', 'hot')
        self.sigma = self.config.get('sigma', 5)  # Gaussian smoothing
    
    def generate_density_heatmap(self, tracks: List[Track],
                                 frame_shape: Tuple[int, int],
                                 output_path: str = None) -> plt.Figure:
        """
        Generate spatial density heatmap.
        
        Args:
            tracks: List of tracks
            frame_shape: (height, width) of video frames
            output_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        height, width = frame_shape
        
        # Create density map
        density = np.zeros((height, width), dtype=np.float32)
        
        for track in tracks:
            for x, y in track.trajectory:
                ix, iy = int(x), int(y)
                if 0 <= ix < width and 0 <= iy < height:
                    density[iy, ix] += 1
        
        # Apply Gaussian smoothing
        density_smooth = gaussian_filter(density, sigma=self.sigma)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        im = ax.imshow(density_smooth, cmap=self.cmap, interpolation='bilinear')
        ax.set_title('Spatial Density Heatmap')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density')
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Density heatmap saved to: {output_path}")
        
        return fig
    
    def generate_velocity_heatmap(self, tracks: List[Track],
                                  frame_shape: Tuple[int, int],
                                  output_path: str = None) -> plt.Figure:
        """
        Generate velocity heatmap.
        
        Args:
            tracks: List of tracks
            frame_shape: (height, width) of frames
            output_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        from ..analyzers.motion_analyzer import MotionAnalyzer
        
        analyzer = MotionAnalyzer(self.config)
        height, width = frame_shape
        
        # Create velocity map
        velocity_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        for track in tracks:
            features = analyzer.analyze_single(track)
            velocities = features.get('velocities', [])
            
            if len(velocities) == 0:
                continue
            
            trajectory = track.trajectory[1:]  # Skip first point (no velocity)
            
            for (x, y), vel in zip(trajectory, velocities):
                ix, iy = int(x), int(y)
                if 0 <= ix < width and 0 <= iy < height:
                    velocity_map[iy, ix] += vel
                    count_map[iy, ix] += 1
        
        # Average velocities
        mask = count_map > 0
        velocity_map[mask] /= count_map[mask]
        
        # Smooth
        velocity_smooth = gaussian_filter(velocity_map, sigma=self.sigma)
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        im = ax.imshow(velocity_smooth, cmap='coolwarm', interpolation='bilinear')
        ax.set_title('Average Velocity Heatmap')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Velocity (pixels/frame)')
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Velocity heatmap saved to: {output_path}")
        
        return fig
    
    def visualize(self, tracks: List[Track], frame_shape: Tuple[int, int],
                 output_path: str = None) -> plt.Figure:
        """Default visualization (density heatmap)."""
        return self.generate_density_heatmap(tracks, frame_shape, output_path)