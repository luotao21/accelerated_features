## Why

The current system allows for real-time homography estimation and hotspot overlay, but lacks interactive feedback. Adding gesture-based audio triggering enhances the user experience by providing auditory information when the user interacts with specific regions (hotspots) on the tracked object.

## What Changes

- Implement a `HandTracker` module using MediaPipe to detect the index finger tip.
- Integrate the hand tracker into the `realtime_demo.py` loop.
- Transform finger coordinates from camera space to reference space using the calculated homography.
- Detect collisions between the finger tip and defined hotspots.
- Trigger audio playback for the corresponding hotspot after a short dwell time.

## Impact

- Users can interact with the book/image using their fingers.
- Adds an educational/interactive layer to the existing AR demo.
- Introduces new dependencies: `mediapipe`, `pygame` (for audio).
