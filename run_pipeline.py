#!/usr/bin/env python3
"""
Command-line interface for tracking pipeline.

Usage:
    python run_pipeline.py --config configs/biology_cells.yaml --input data/cells.mp4 --output outputs/
"""

import argparse
from pathlib import Path
from src.pipeline import TrackingPipeline
from src.core.config import Config


def main():
    parser = argparse.ArgumentParser(
        description='Intelligent Object Tracking Pipeline'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input video or image directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading configuration from: {args.config}")
    config = Config.from_yaml(args.config)
    
    # Override max_frames if specified
    if args.max_frames:
        config.set('data.max_frames', args.max_frames)
    
    # Create pipeline
    pipeline = TrackingPipeline(config)
    
    # Run
    print(f"Processing: {args.input}")
    results = pipeline.run(
        source=args.input,
        output_dir=args.output
    )
    
    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()