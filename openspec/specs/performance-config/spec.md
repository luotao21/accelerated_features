# performance-config Specification

## Purpose
TBD - created by archiving change optimize-performance. Update Purpose after archive.
## Requirements
### Requirement: Global Frame Skipping

The system MUST support a mechanism to skip the heavy model inference step for a specified number of frames to reduce computational load.

#### Scenario: User wants smooth UI despite heavy processing

- **Given** the user runs the demo with `--skip_frames 2`
- **When** the camera captures 30 frames per second
- **Then** the XFeat inference should only run 10 times per second (once every 3 frames)
- **And** the UI should still update at 30 FPS using the latest estimated homography

### Requirement: Inference Resolution Scaling

The system SHALL support running the feature detection model on a downsampled image and automatically scaling the results back to the original resolution.

#### Scenario: User wants to trade accuracy for speed

- **Given** the user runs the demo with `--infer_scale 0.5`
- **When** the input video is 640x480
- **Then** the XFeat model should process a 320x240 image
- **And** the detected keypoints should be scaled up by 2x to match the display

### Requirement: macOS MPS Acceleration

The system SHALL detect if the code is running on macOS with Apple Silicon and automatically utilize Metal Performance Shaders (MPS) for GPU acceleration instead of CPU.

#### Scenario: User runs on MacBook

- **Given** the user is on a Mac with Apple Silicon
- **When** the application starts
- **Then** the torch device should be set to `mps`
- **And** the inference should run on the GPU

