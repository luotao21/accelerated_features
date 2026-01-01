# Performance Optimization Proposal

## Goal

Reduce the computational load of the XFeat real-time demo to improve framerate and reduce thermal pressure on the device.

## Problem

The current implementation runs XFeat feature detection and matching on every single video frame. This is computationally expensive, especially on CPUs, leading to low FPS and high latency.

## Solution

Implement a "Frame Skipping" mechanism and resolution downsampling for inference.

1. **Frame Skipping**: Run the heavy `detectAndCompute` and matching logic only every N frames (e.g., every 3rd or 4th frame).
2. **Resolution Scaling**: Perform inference on a lower resolution copy of the frame, then scale keypoints back to the display resolution.

## Scope

- Valid for `realtime_demo.py` script.
- Focuses on XFeat method optimization.
