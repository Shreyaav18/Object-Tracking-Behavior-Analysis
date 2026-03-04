"""
Complete example: Cell tracking from microscopy video.
"""

from src.pipeline import TrackingPipeline
from src.core.config import Config

# Load configuration
config = Config.from_yaml('configs/biology_cells.yaml')

# Create pipeline
pipeline = TrackingPipeline(config)

# Run complete pipeline
results = pipeline.run(
    source='data/cells.mp4',
    output_dir='outputs/cell_tracking'
)

# Access results
print(f"\nNumber of tracks: {len(results['tracks'])}")

# Get specific analysis results
if 'motion' in results['analysis']:
    motion = results['analysis']['motion']
    print(f"Mean velocity: {motion['summary'].get('mean_velocity'):.2f} μm/s")

print(f"\nCheck outputs in: outputs/cell_tracking/")