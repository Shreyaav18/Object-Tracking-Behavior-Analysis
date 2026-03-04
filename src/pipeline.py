"""
Main pipeline orchestrator.
Coordinates all components: loading, preprocessing, detection, tracking, analysis, visualization.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from .core.config import Config
from .core.data_loader import DataLoader
from .preprocessing.preprocessor import ImagePreprocessor
from .preprocessing.pipeline import PreprocessingPipeline
from .detectors import DetectorFactory, Detection
from .trackers import TrackerFactory, Track
from .analyzers import MotionAnalyzer, TrajectoryAnalyzer, BehaviorClassifier
from .visualizers import VideoAnnotator, TrajectoryPlotter, HeatmapGenerator, DashboardGenerator


class TrackingPipeline:
    """
    Complete tracking pipeline.
    Orchestrates all components from data loading to output generation.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.detector = None
        self.tracker = None
        self.analyzers = {}
        self.visualizers = {}
        
        # Data storage
        self.frames = None
        self.metadata = None
        self.preprocessed_frames = None
        self.all_detections = []
        self.all_tracks = []
        self.tracks = []
        self.analysis_results = {}
        
        # Setup components based on config
        self._setup_components()
    
    def _setup_components(self):
        """Initialize all components based on configuration."""
        
        # Preprocessor
        if self.config.get('preprocessing.enabled', False):
            preproc_config = self.config.get('preprocessing', {})
            self.preprocessor = ImagePreprocessor(preproc_config)
        
        # Detector
        detector_type = self.config.get('detector.type', 'blob')
        detector_config = self.config.get('detector.config', {})
        self.detector = DetectorFactory.create(detector_type, detector_config)
        
        # Tracker
        tracker_type = self.config.get('tracker.type', 'simple')
        tracker_config = self.config.get('tracker.config', {})
        self.tracker = TrackerFactory.create(tracker_type, tracker_config)
        
        # Analyzers
        if self.config.get('analysis.motion.enabled', True):
            motion_config = self.config.get('analysis.motion.config', {})
            self.analyzers['motion'] = MotionAnalyzer(motion_config)
        
        if self.config.get('analysis.trajectory.enabled', False):
            traj_config = self.config.get('analysis.trajectory.config', {})
            self.analyzers['trajectory'] = TrajectoryAnalyzer(traj_config)
        
        if self.config.get('analysis.behavior.enabled', False):
            behavior_config = self.config.get('analysis.behavior.config', {})
            self.analyzers['behavior'] = BehaviorClassifier(behavior_config)
        
        # Visualizers
        if self.config.get('visualization.video.enabled', True):
            video_config = self.config.get('visualization.video.config', {})
            self.visualizers['video'] = VideoAnnotator(video_config)
        
        if self.config.get('visualization.plots.enabled', True):
            plot_config = self.config.get('visualization.plots.config', {})
            self.visualizers['trajectory'] = TrajectoryPlotter(plot_config)
            self.visualizers['heatmap'] = HeatmapGenerator(plot_config)
            self.visualizers['dashboard'] = DashboardGenerator(plot_config)
    
    def load_data(self, source: str, **kwargs) -> Tuple[List[np.ndarray], Dict]:
        """
        Load video/image data.
        
        Args:
            source: Path to video file or image directory
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            frames: List of frames
            metadata: Metadata dictionary
        """
        print(f"\n{'='*60}")
        print(f"LOADING DATA")
        print(f"{'='*60}")
        
        # Merge config and kwargs
        loader_config = {
            'max_frames': self.config.get('data.max_frames'),
            'start_frame': self.config.get('data.start_frame', 0),
            'grayscale': self.config.get('data.grayscale', False),
            'resize': self.config.get('data.resize'),
            **kwargs
        }
        
        self.data_loader = DataLoader(source, **loader_config)
        self.frames, self.metadata = self.data_loader.load()
        
        print(f"✓ Loaded {len(self.frames)} frames")
        print(f"✓ Resolution: {self.metadata.get('resolution')}")
        print(f"✓ FPS: {self.metadata.get('fps')}")
        
        return self.frames, self.metadata
    
    def preprocess(self) -> List[np.ndarray]:
        """
        Preprocess all frames.
        
        Returns:
            Preprocessed frames
        """
        if self.preprocessor is None:
            self.preprocessed_frames = self.frames
            return self.frames
        
        print(f"\n{'='*60}")
        print(f"PREPROCESSING")
        print(f"{'='*60}")
        
        self.preprocessed_frames = []
        for frame in tqdm(self.frames, desc="Preprocessing frames"):
            processed = self.preprocessor.process(frame)
            self.preprocessed_frames.append(processed)
        
        print(f"✓ Preprocessed {len(self.preprocessed_frames)} frames")
        
        return self.preprocessed_frames
    
    def detect(self) -> List[List[Detection]]:
        """
        Detect objects in all frames.
        
        Returns:
            List of detection lists (one per frame)
        """
        print(f"\n{'='*60}")
        print(f"DETECTION")
        print(f"{'='*60}")
        
        frames_to_process = self.preprocessed_frames or self.frames
        
        self.all_detections = []
        total_detections = 0
        
        for frame in tqdm(frames_to_process, desc="Detecting objects"):
            detections = self.detector.detect(frame)
            self.all_detections.append(detections)
            total_detections += len(detections)
        
        avg_per_frame = total_detections / len(frames_to_process) if frames_to_process else 0
        print(f"✓ Total detections: {total_detections}")
        print(f"✓ Average per frame: {avg_per_frame:.1f}")
        
        return self.all_detections
    
    def track(self) -> List[Track]:
        """
        Track objects across frames.
        
        Returns:
            List of tracks
        """
        print(f"\n{'='*60}")
        print(f"TRACKING")
        print(f"{'='*60}")
        
        self.all_tracks = []
        
        for detections in tqdm(self.all_detections, desc="Tracking objects"):
            tracks = self.tracker.update(detections)
            self.all_tracks.append(tracks.copy())
        
        # Get final tracks
        self.tracks = self.tracker.get_tracks(confirmed_only=True)
        
        print(f"✓ Total tracks: {len(self.tracks)}")
        print(f"✓ Confirmed tracks: {sum(1 for t in self.tracks if t.is_confirmed)}")
        
        # Track length statistics
        if self.tracks:
            lengths = [len(t.trajectory) for t in self.tracks]
            print(f"✓ Mean track length: {np.mean(lengths):.1f} frames")
            print(f"✓ Max track length: {np.max(lengths)} frames")
        
        return self.tracks
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze tracking results.
        
        Returns:
            Dictionary of analysis results
        """
        print(f"\n{'='*60}")
        print(f"ANALYSIS")
        print(f"{'='*60}")
        
        self.analysis_results = {}
        
        for name, analyzer in self.analyzers.items():
            print(f"Running {name} analysis...")
            results = analyzer.analyze(self.tracks)
            self.analysis_results[name] = results
        
        print(f"✓ Completed {len(self.analyzers)} analyses")
        
        return self.analysis_results
    
    def visualize(self, output_dir: str = 'outputs') -> Dict[str, str]:
        """
        Generate all visualizations.
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary of output paths
        """
        print(f"\n{'='*60}")
        print(f"VISUALIZATION")
        print(f"{'='*60}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = {}
        
        # Annotated video
        if 'video' in self.visualizers and self.frames:
            print("Generating annotated video...")
            video_path = output_dir / 'annotated_video.mp4'
            self.visualizers['video'].annotate_video(
                self.frames, self.all_tracks, str(video_path)
            )
            output_paths['video'] = str(video_path)
        
        # Trajectory plot
        if 'trajectory' in self.visualizers and self.tracks:
            print("Generating trajectory plot...")
            traj_path = output_dir / 'trajectories.png'
            self.visualizers['trajectory'].plot_trajectories(
                self.tracks, str(traj_path)
            )
            output_paths['trajectories'] = str(traj_path)
            
            # MSD plot
            msd_path = output_dir / 'msd.png'
            self.visualizers['trajectory'].plot_msd(
                self.tracks, str(msd_path)
            )
            output_paths['msd'] = str(msd_path)
            
            # Velocity distribution
            vel_path = output_dir / 'velocity_distribution.png'
            self.visualizers['trajectory'].plot_velocity_distribution(
                self.tracks, str(vel_path)
            )
            output_paths['velocity'] = str(vel_path)
        
        # Heatmap
        if 'heatmap' in self.visualizers and self.tracks and self.frames:
            print("Generating heatmaps...")
            frame_shape = self.frames[0].shape[:2]
            
            density_path = output_dir / 'density_heatmap.png'
            self.visualizers['heatmap'].generate_density_heatmap(
                self.tracks, frame_shape, str(density_path)
            )
            output_paths['density'] = str(density_path)
            
            velocity_path = output_dir / 'velocity_heatmap.png'
            self.visualizers['heatmap'].generate_velocity_heatmap(
                self.tracks, frame_shape, str(velocity_path)
            )
            output_paths['velocity_heatmap'] = str(velocity_path)
        
        # Dashboard
        if 'dashboard' in self.visualizers and self.tracks:
            print("Generating dashboard...")
            frame_shape = self.frames[0].shape[:2] if self.frames else None
            dashboard_path = output_dir / 'dashboard.png'
            self.visualizers['dashboard'].generate_dashboard(
                self.tracks, frame_shape, str(dashboard_path)
            )
            output_paths['dashboard'] = str(dashboard_path)
        
        print(f"✓ Generated {len(output_paths)} visualizations")
        
        return output_paths
    
    def export_results(self, output_dir: str = 'outputs') -> Dict[str, str]:
        """
        Export results to files (CSV, JSON).
        
        Args:
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary of output paths
        """
        print(f"\n{'='*60}")
        print(f"EXPORTING RESULTS")
        print(f"{'='*60}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = {}
        
        # Export tracks to CSV
        import pandas as pd
        
        # Trajectory data
        trajectory_data = []
        for track in self.tracks:
            for i, (frame_num, (x, y)) in enumerate(zip(track.frames, track.trajectory)):
                trajectory_data.append({
                    'track_id': track.track_id,
                    'frame': frame_num,
                    'x': x,
                    'y': y,
                    'state': track.state
                })
        
        if trajectory_data:
            df_traj = pd.DataFrame(trajectory_data)
            traj_path = output_dir / 'trajectories.csv'
            df_traj.to_csv(traj_path, index=False)
            output_paths['trajectories_csv'] = str(traj_path)
            print(f"✓ Exported trajectories to {traj_path}")
        
        # Export analysis results to JSON
        import json
        
        if self.analysis_results:
            # Convert numpy types to native Python types
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            results_clean = convert_types(self.analysis_results)
            
            results_path = output_dir / 'analysis_results.json'
            with open(results_path, 'w') as f:
                json.dump(results_clean, f, indent=2)
            output_paths['analysis_json'] = str(results_path)
            print(f"✓ Exported analysis results to {results_path}")
        
        # Export metadata
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        output_paths['metadata'] = str(metadata_path)
        print(f"✓ Exported metadata to {metadata_path}")
        
        return output_paths
    
    def run(self, source: str, output_dir: str = 'outputs', **kwargs) -> Dict[str, Any]:
        """
        Run complete pipeline end-to-end.
        
        Args:
            source: Path to video/image data
            output_dir: Directory for outputs
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with all results and output paths
        """
        print(f"\n{'#'*60}")
        print(f"# INTELLIGENT OBJECT TRACKING PIPELINE")
        print(f"{'#'*60}")
        
        # 1. Load data
        self.load_data(source, **kwargs)
        
        # 2. Preprocess
        self.preprocess()
        
        # 3. Detect
        self.detect()
        
        # 4. Track
        self.track()
        
        # 5. Analyze
        self.analyze()
        
        # 6. Visualize
        viz_paths = self.visualize(output_dir)
        
        # 7. Export
        export_paths = self.export_results(output_dir)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"✓ Processed {len(self.frames)} frames")
        print(f"✓ Detected {sum(len(d) for d in self.all_detections)} objects")
        print(f"✓ Tracked {len(self.tracks)} objects")
        print(f"✓ Generated {len(viz_paths)} visualizations")
        print(f"✓ Exported {len(export_paths)} data files")
        print(f"\nOutputs saved to: {output_dir}")
        
        return {
            'frames': len(self.frames),
            'tracks': self.tracks,
            'detections': self.all_detections,
            'analysis': self.analysis_results,
            'visualizations': viz_paths,
            'exports': export_paths
        }