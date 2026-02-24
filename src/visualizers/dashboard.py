"""
Generate comprehensive analysis dashboard.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from pathlib import Path
from .base_visualizer import BaseVisualizer
from ..trackers.base_tracker import Track


class DashboardGenerator(BaseVisualizer):
    """
    Generate multi-panel analysis dashboard.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        
        self.figsize = self.config.get('figsize', (16, 12))
        self.dpi = self.config.get('dpi', 100)
    
    def generate_dashboard(self, tracks: List[Track],
                          frame_shape: tuple = None,
                          output_path: str = None) -> plt.Figure:
        """
        Generate comprehensive dashboard with multiple panels.
        
        Args:
            tracks: List of tracks
            frame_shape: (height, width) for spatial plots
            output_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        from ..analyzers.motion_analyzer import MotionAnalyzer
        from ..analyzers.trajectory_analyzer import TrajectoryAnalyzer
        
        motion_analyzer = MotionAnalyzer(self.config)
        traj_analyzer = TrajectoryAnalyzer(self.config)
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Trajectories
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self._plot_trajectories(ax1, tracks)
        
        # 2. Track length distribution
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_track_lengths(ax2, tracks)
        
        # 3. Velocity distribution
        ax3 = fig.add_subplot(gs[1, 2])
        self._plot_velocity_dist(ax3, tracks, motion_analyzer)
        
        # 4. Displacement over time
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_displacement(ax4, tracks, motion_analyzer)
        
        # 5. Straightness distribution
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_straightness(ax5, tracks, traj_analyzer)
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_summary_stats(ax6, tracks, motion_analyzer)
        
        plt.suptitle(f'Tracking Analysis Dashboard (n={len(tracks)} tracks)',
                    fontsize=16, fontweight='bold')
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Dashboard saved to: {output_path}")
        
        return fig
    
    def _plot_trajectories(self, ax, tracks):
        """Plot trajectories panel."""
        for track in tracks[:20]:  # Limit to avoid clutter
            if len(track.trajectory) < 2:
                continue
            traj = np.array(track.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=1)
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=4)
            ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=4)
        
        ax.set_title('Trajectories')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    def _plot_track_lengths(self, ax, tracks):
        """Plot track length distribution."""
        lengths = [len(t.trajectory) for t in tracks]
        ax.hist(lengths, bins=20, edgecolor='black', alpha=0.7)
        ax.set_title('Track Length Distribution')
        ax.set_xlabel('Length (frames)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    def _plot_velocity_dist(self, ax, tracks, analyzer):
        """Plot velocity distribution."""
        all_velocities = []
        for track in tracks:
            features = analyzer.analyze_single(track)
            all_velocities.extend(features.get('velocities', []))
        
        if all_velocities:
            ax.hist(all_velocities, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(all_velocities), color='r', linestyle='--',
                      label=f'Mean: {np.mean(all_velocities):.2f}')
            ax.set_title('Velocity Distribution')
            ax.set_xlabel('Velocity (px/frame)')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    def _plot_displacement(self, ax, tracks, analyzer):
        """Plot total displacement distribution."""
        displacements = []
        for track in tracks:
            features = analyzer.analyze_single(track)
            displacements.append(features.get('total_displacement', 0))
        
        if displacements:
            ax.hist(displacements, bins=20, edgecolor='black', alpha=0.7)
            ax.set_title('Total Displacement')
            ax.set_xlabel('Displacement (pixels)')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
    
    def _plot_straightness(self, ax, tracks, analyzer):
        """Plot straightness distribution."""
        straightness = []
        for track in tracks:
            features = analyzer.analyze_single(track)
            straightness.append(features.get('straightness', 0))
        
        if straightness:
            ax.hist(straightness, bins=20, edgecolor='black', alpha=0.7, range=(0, 1))
            ax.set_title('Trajectory Straightness')
            ax.set_xlabel('Straightness (0-1)')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)
    
    def _plot_summary_stats(self, ax, tracks, analyzer):
        """Display summary statistics."""
        ax.axis('off')
        
        # Calculate stats
        all_velocities = []
        path_lengths = []
        
        for track in tracks:
            features = analyzer.analyze_single(track)
            all_velocities.extend(features.get('velocities', []))
            path_lengths.append(features.get('path_length', 0))
        
        stats_text = f"""
        Summary Statistics
        
        Total Tracks: {len(tracks)}
        Confirmed: {sum(1 for t in tracks if t.is_confirmed)}
        
        Mean Velocity: {np.mean(all_velocities):.2f} px/frame
        Std Velocity: {np.std(all_velocities):.2f}
        
        Mean Path Length: {np.mean(path_lengths):.2f} px
        Max Path Length: {np.max(path_lengths):.2f} px
        
        Mean Track Duration: {np.mean([t.age for t in tracks]):.1f} frames
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
               family='monospace')
    
    def visualize(self, tracks: List[Track], frame_shape: tuple = None,
                 output_path: str = None) -> plt.Figure:
        """Default visualization (dashboard)."""
        return self.generate_dashboard(tracks, frame_shape, output_path)