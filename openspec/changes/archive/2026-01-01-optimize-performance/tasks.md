# Tasks

- [x] Add CLI arguments for optimization <!-- id: 0 -->
  - `--skip_frames`: Number of frames to skip processing (default: 0).
  - `--infer_scale`: Scale factor for inference resolution (e.g. 0.5 for half resolution).
- [x] Implement Frame Skipping Logic <!-- id: 1 -->
  - Modify `MatchingDemo` to track frame count.
  - Conditionally run `match_and_draw`.
- [x] Implement Inference Scaling <!-- id: 2 -->
  - Resize image before sending to model.
  - Scale keypoints back after inference.
- [x] Enable MPS Acceleration <!-- id: 3 -->
  - Update `xfeat.py` to check for `torch.backends.mps.is_available()`.
  - Set device to `mps` on supported macOS systems.
- [x] Verify Performance <!-- id: 4 -->
  - Measure FPS with different skip/scale settings.
