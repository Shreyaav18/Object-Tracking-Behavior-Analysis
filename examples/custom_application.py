"""
Example: Building a custom application for IISER project.
"""

from src.pipeline import TrackingPipeline
from src.core.config import Config
from src.analyzers import MotionAnalyzer

# Custom config for microtubule dynamics
config = Config.from_dict({
    'data': {
        'grayscale': True,
        'max_frames': 200
    },
    'preprocessing': {
        'enabled': True,
        'denoise': {'method': 'gaussian', 'kernel_size': 3},
        'enhance_contrast': {'method': 'clahe'}
    },
    'detector': {
        'type': 'blob',
        'config': {
            'min_area': 20,
            'max_area': 500,
            'min_circularity': 0.4,
            'threshold_method': 'adaptive'
        }
    },
    'tracker': {
        'type': 'trackpy',
        'config': {
            'search_range': 10,
            'memory': 5
        }
    },
    'analysis': {
        'motion': {
            'enabled': True,
            'config': {
                'fps': 60,
                'pixel_size': 0.08  # 80nm per pixel
            }
        }
    }
})

# Run pipeline
pipeline = TrackingPipeline(config)
results = pipeline.run(
    source='data/microtubules.tif',
    output_dir='outputs/microtubule_analysis'
)

# Custom analysis
motion_analyzer = MotionAnalyzer(config={'fps': 60, 'pixel_size': 0.08})

print("\n=== Microtubule Dynamics Analysis ===")
for track in results['tracks']:
    if len(track.trajectory) < 10:
        continue
    
    # Fit diffusion model
    diffusion = motion_analyzer.fit_diffusion_model(track)
    
    print(f"\nTrack {track.track_id}:")
    print(f"  Length: {len(track.trajectory)} frames")
    print(f"  Diffusion coefficient: {diffusion['diffusion_coefficient']:.4f} μm²/s")
    print(f"  Alpha: {diffusion['alpha']:.3f}")
    print(f"  Motion type: {diffusion['motion_type']}")

print(f"\nComplete results saved to: outputs/microtubule_analysis/")