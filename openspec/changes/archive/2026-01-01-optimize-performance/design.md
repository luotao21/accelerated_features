# Performance Optimization Design

## Architecture Changes

### Decoupled Processing

Currently, `process()` does both **Processing** (Inference + Matching) and **Rendering** (Drawing Results) in a single synchronous block.

We will introduce a `skip_frames` counter.

- **Heavy Frame**: Update global state (Homography `H`, Keypoints `kp`, Matches). Reset counter.
- **Light Frame**: Reuse last known `H` and keypoints to render on the new frame.

### Inference Downsampling

- Add `inference_width` / `inference_height` parameters.
- Resize frame before passing to `xfeat.detectAndCompute`.
- Scale resulting keypoint coordinates by `display_width / inference_width`.

## Data Flow

1. `FrameGrabber` -> New Frame
2. `MainLoop`:
    - If `frame_count % skip == 0`:
        - Resize frame -> `small_frame`
        - `XFeat(small_frame)` -> `kpts_small`
        - `kpts_small * scale_factor` -> `kpts_display`
        - Match -> Update `self.H`
    - Else:
        - Use cached `self.H`
3. `Render`:
    - Draw cached `H` on new frame.
