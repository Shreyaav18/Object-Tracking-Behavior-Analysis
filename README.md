# Intelligent Object Tracking Framework

A modular, reusable framework for object detection, tracking, and behavior analysis across multiple domains (biology, surveillance, sports, etc.).

## Features

- **Multiple Detection Methods**: Blob detection, YOLO, motion-based
- **Advanced Tracking**: Simple, TrackPy, SORT with Kalman filtering
- **Comprehensive Analysis**: Motion metrics, trajectory features, behavior classification
- **Rich Visualizations**: Annotated videos, trajectory plots, heatmaps, dashboards
- **Fully Configurable**: YAML-based configuration system
- **Domain Agnostic**: Works for cells, people, vehicles, particles, etc.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from src.pipeline import TrackingPipeline
from src.core.config import Config

# Load configuration
config = Config.from_yaml('configs/biology_cells.yaml')

# Run pipeline
pipeline = TrackingPipeline(config)
results = pipeline.run('data/video.mp4', output_dir='outputs/')
```

### Command Line
```bash
python run_pipeline.py \
    --config configs/biology_cells.yaml \
    --input data/cells.mp4 \
    --output outputs/cell_tracking
```

## Applications

### 1. Biological Cell Tracking
```bash
python run_pipeline.py -c configs/biology_cells.yaml -i data/cells.mp4
```

### 2. Pedestrian Tracking
```bash
python run_pipeline.py -c configs/pedestrian_tracking.yaml -i data/surveillance.mp4
```

### 3. Custom Application

See `examples/custom_application.py` for building domain-specific applications.

## Configuration

Edit YAML files in `configs/` to customize:
- Detection parameters
- Tracking algorithms
- Analysis metrics
- Visualization options

Example config structure:
```yaml
detector:
  type: blob
  config:
    min_area: 50
    max_area: 2000

tracker:
  type: trackpy
  config:
    search_range: 15
    memory: 3
```

## Output

The pipeline generates:
- Annotated videos with tracking overlays
- Trajectory plots and heatmaps
- Analysis dashboards
- CSV files with trajectory data
- JSON files with analysis results


This demonstrates:
- Particle detection and tracking
- Motion analysis (velocity, MSD, diffusion)
- Behavior classification
- Publication-ready visualizations
