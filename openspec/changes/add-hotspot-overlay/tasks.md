# Tasks

- [x] Create SVG hotspot loader module <!-- id: 0 -->
  - Parse SVG files from `assets/hotspot/`
  - Extract path data from `<path d="...">` elements
  - Convert SVG path to OpenCV polygon coordinates

- [x] Implement hotspot mask rendering <!-- id: 1 -->
  - Create BGRA mask at reference resolution (2240x1344)
  - Fill each hotspot polygon with semi-transparent green
  - Resize to display resolution (640x480)

- [x] Integrate hotspot overlay into realtime_demo <!-- id: 2 -->
  - Load hotspots on startup
  - Blend hotspot mask with reference frame
  - Apply Homography transform to current frame overlay

- [x] Add CLI argument for hotspot visibility <!-- id: 3 -->
  - `--show_hotspots`: Enable/disable hotspot overlay
  - Default: enabled when hotspot directory exists

- [x] Verify visual alignment <!-- id: 4 -->
  - Compare overlay positions with expected result image
  - Ensure hotspots follow tracked object correctly
