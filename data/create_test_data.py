"""
Create synthetic test video with moving particles.
"""

import numpy as np
import cv2
from pathlib import Path

# Parameters
n_frames = 100
width, height = 640, 480
n_particles = 10

# Create output directory
Path('data/sample').mkdir(parents=True, exist_ok=True)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('data/sample/test_particles.mp4', fourcc, 30, (width, height), False)

# Initialize particle positions
particles = np.random.rand(n_particles, 2) * [width, height]
velocities = (np.random.rand(n_particles, 2) - 0.5) * 4

print("Generating test video...")

for frame_num in range(n_frames):
    # Create blank frame
    frame = np.zeros((height, width), dtype=np.uint8)
    
    # Update particle positions
    particles += velocities
    
    # Bounce off walls
    particles[:, 0] = np.clip(particles[:, 0], 10, width - 10)
    particles[:, 1] = np.clip(particles[:, 1], 10, height - 10)
    
    # Reverse velocity at boundaries
    velocities[particles[:, 0] <= 10, 0] = abs(velocities[particles[:, 0] <= 10, 0])
    velocities[particles[:, 0] >= width-10, 0] = -abs(velocities[particles[:, 0] >= width-10, 0])
    velocities[particles[:, 1] <= 10, 1] = abs(velocities[particles[:, 1] <= 10, 1])
    velocities[particles[:, 1] >= height-10, 1] = -abs(velocities[particles[:, 1] >= height-10, 1])
    
    # Draw particles
    for x, y in particles.astype(int):
        cv2.circle(frame, (x, y), 8, 255, -1)
    
    # Add some noise
    noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
    frame = cv2.add(frame, noise)
    
    out.write(frame)

out.release()
print("✓ Test video created: data/sample/test_particles.mp4")
print("Now run: python quick_test.py")