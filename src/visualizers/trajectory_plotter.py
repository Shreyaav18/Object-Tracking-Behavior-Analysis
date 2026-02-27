"""
Plot trajectories and motion analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import List, Dict, Any
from pathlib import Path
from .base_visualizer import BaseVisualizer
from ..trackers.base_tracker import Track


class TrajectoryPlotter(BaseVisualizer):
    """
    Create trajectory plots and visualizations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        self.figsize = self.config.get('figsize', (10, 8))
        self.dpi = self.config.get('dpi', 100)
        self.cmap = self.config.get('cmap', 'viridis')
    
    def plot_trajectories(self, tracks: List[Track], 
                         output_path: str = None,
                         show_start_end: bool = True,
                         color_by: str = 'track') -> plt.Figure:
        """
        Plot all trajectories.
        
        Args:
            tracks: List of tracks
            output_path: Path to save figure (None = display only)
            show_start_end: Mark start/end points
            color_by: 'track' or 'time' or 'velocity'
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for track in tracks:
            if len(track.trajectory) < 2:
                continue
            
            trajectory = np.array(track.trajectory)
            
            if color_by == 'track':
                # Single color per track
                ax.plot(trajectory[:, 0], trajectory[:, 1],
                       alpha=0.7, linewidth=1.5, label=f'Track {track.track_id}')
            
            elif color_by == 'time':
                # Color gradient by time
                points = trajectory.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                lc = LineCollection(segments, cmap=self.cmap, alpha=0.7)
                lc.set_array(np.arange(len(trajectory)))
                ax.add_collection(lc)
            
            elif color_by == 'velocity':
                # Color by velocity
                if len(trajectory) > 1:
                    velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
                    
                    points = trajectory[:-1].reshape(-1, 1, 2)
                    segments = np.concatenate([points, trajectory[1:].reshape(-1, 1, 2)], axis=1)
                    
                    lc = LineCollection(segments, cmap='coolwarm', alpha=0.7)
                    lc.set_array(velocities)
                    ax.add_collection(lc)
                    plt.colorbar(lc, ax=ax, label='Velocity')
            
            # Mark start and end
            if show_start_end:
                ax.plot(trajectory[0, 0], trajectory[0, 1], 
                       'go', markersize=8, label='Start' if track.track_id == tracks[0].track_id else '')
                ax.plot(trajectory[-1, 0], trajectory[-1, 1],
                       'ro', markersize=8, label='End' if track.track_id == tracks[0].track_id else '')
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(f'Trajectories (n={len(tracks)})')
        ax.invert_yaxis()  # Match image coordinates
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if color_by == 'track' and len(tracks) <= 10:
            ax.legend(loc='best', fontsize=8)
        elif show_start_end:
            ax.legend(loc='best')
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Trajectory plot saved to: {output_path}")
        
        return fig
    
    def plot_msd(self, tracks: List[Track], output_path: str = None) -> plt.Figure:
        """
        Plot Mean Squared Displacement curves.
        
        Args:
            tracks: List of tracks
            output_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        from ..analyzers.motion_analyzer import MotionAnalyzer
        
        analyzer = MotionAnalyzer(self.config)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for track in tracks:
            if len(track.trajectory) < 10:
                continue
            
            features = analyzer.analyze_single(track)
            msd = features.get('msd')
            
            if msd is not None and len(msd) > 1:
                time_lags = np.arange(len(msd))
                ax.plot(time_lags, msd, alpha=0.5, label=f'Track {track.track_id}')
        
        ax.set_xlabel('Time Lag (frames)')
        ax.set_ylabel('MSD (pixels²)')
        ax.set_title('Mean Squared Displacement')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        if len(tracks) <= 10:
            ax.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"MSD plot saved to: {output_path}")
        
        return fig
    
    def plot_velocity_distribution(self, tracks: List[Track],
                                   output_path: str = None) -> plt.Figure:
        """
        Plot velocity distribution histogram.
        
        Args:
            tracks: List of tracks
            output_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        from ..analyzers.motion_analyzer import MotionAnalyzer
        
        analyzer = MotionAnalyzer(self.config)
        
        all_velocities = []
        for track in tracks:
            features = analyzer.analyze_single(track)
            velocities = features.get('velocities', [])
            all_velocities.extend(velocities)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        ax.hist(all_velocities, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(all_velocities), color='r', linestyle='--',
                  label=f'Mean: {np.mean(all_velocities):.2f}')
        ax.axvline(np.median(all_velocities), color='g', linestyle='--',
                  label=f'Median: {np.median(all_velocities):.2f}')
        
        ax.set_xlabel('Velocity (pixels/frame)')
        ax.set_ylabel('Frequency')
        ax.set_title('Velocity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Velocity distribution saved to: {output_path}")
        
        return fig
    
    def visualize(self, tracks: List[Track], output_path: str = None) -> plt.Figure:
        """Default visualization (trajectories)."""
        return self.plot_trajectories(tracks, output_path)