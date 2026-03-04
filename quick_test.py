"""
Quick test script - Run this first!
"""

from src.pipeline import TrackingPipeline
from src.core.config import Config

# Minimal config
config = Config.from_dict({
    'detector': {
        'type': 'blob',
        'config': {
            'min_area': 50,
            'max_area': 2000,
            'min_circularity': 0.5
        }
    },
    'tracker': {
        'type': 'simple',
        'config': {
            'max_distance': 30,
            'max_age': 10
        }
    }
})

# Create pipeline
pipeline = TrackingPipeline(config)

# Run on your video
results = pipeline.run(
    source='data/sample/test_particles.mp4',  
    output_dir='outputs/test'
)

print("\n✓ Done! Check outputs/test/ for results")