"""
Step-by-step example with full control.
"""

from src.pipeline import TrackingPipeline
from src.core.config import Config

# Initialize pipeline
config = Config.from_yaml('configs/biology_cells.yaml')
pipeline = TrackingPipeline(config)

# Step 1: Load data
frames, metadata = pipeline.load_data('data/cells.mp4')
print(f"Loaded {len(frames)} frames at {metadata['fps']} fps")

# Step 2: Preprocess (optional)
preprocessed = pipeline.preprocess()

# Step 3: Detect objects
detections = pipeline.detect()
print(f"Average detections per frame: {sum(len(d) for d in detections) / len(detections):.1f}")

# Step 4: Track objects
tracks = pipeline.track()
print(f"Found {len(tracks)} tracks")

# Step 5: Analyze
analysis = pipeline.analyze()

# Step 6: Visualize
viz_paths = pipeline.visualize('outputs/step_by_step')

# Step 7: Export
export_paths = pipeline.export_results('outputs/step_by_step')

print("\nPipeline complete!")
print(f"Visualizations: {list(viz_paths.keys())}")
print(f"Exports: {list(export_paths.keys())}")