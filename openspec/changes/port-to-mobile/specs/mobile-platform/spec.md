# Mobile Platform Capability

## ADDED Requirements

### Requirement: Multi-page Book Data Format

The system SHALL provide server-side tools to export all pages of a book as a single unified JSON file containing features and hotspots.

#### Scenario: Book Data Export

- **WHEN** a book with multiple pages is processed on the server
- **THEN** the system SHALL generate a single `book_data.json` file
- **AND** include version info, book ID, and update timestamp
- **AND** contain feature data (keypoints, descriptors, scores) for each page
- **AND** contain hotspot data (name, polygon, audio URL) for each page

#### Scenario: Version-based Caching

- **WHEN** the app starts
- **THEN** it SHALL check the `updatedAt` timestamp against local cache
- **AND** download new version only if server version is newer
- **AND** use local cache for offline access

---

### Requirement: Preserve Video Aspect Ratio

The system SHALL process video streams in their original aspect ratio without cropping or padding to square.

> [!CAUTION]
> Forcing video to square aspect ratio would cause book edge regions to be inaccessible!

#### Scenario: Non-square Video Input

- **WHEN** camera provides 16:9 or 4:3 video stream
- **THEN** MediaPipe and XFeat SHALL process it in original aspect ratio
- **AND** NOT crop, pad, or resize to 1:1 square

#### Scenario: Orientation Coordinate Transform

- **WHEN** device orientation changes between landscape and portrait
- **THEN** the system SHALL apply appropriate coordinate transformation
- **AND** maintain correct mapping between finger position and hotspot regions

#### Scenario: Feature Data Loading

- **WHEN** the app starts
- **THEN** it SHALL load pre-computed feature data from JSON files
- **AND** cache them for offline use

#### Scenario: Real-time Matching

- **WHEN** camera frame is captured at 5 FPS intervals
- **THEN** the system SHALL extract frame features locally
- **AND** match against pre-computed reference features
- **AND** compute homography matrix using RANSAC

---

### Requirement: Mobile Hand Tracking at 30 FPS

The system SHALL detect hand landmarks using MediaPipe on mobile devices with 30 FPS target frame rate.

#### Scenario: iPadOS/iPhone Hand Detection

- **WHEN** camera frame is captured on iOS device
- **THEN** the system SHALL detect index finger tip position at 30 FPS
- **AND** provide coordinates in screen space with < 33ms latency

#### Scenario: Android Hand Detection

- **WHEN** camera frame is captured on Android device
- **THEN** the system SHALL detect index finger tip position at 30 FPS
- **AND** handle camera rotation correctly

---

### Requirement: Frame Rate Separation

The system SHALL prioritize MediaPipe hand tracking at 30 FPS while running XFeat matching at 5 FPS.

#### Scenario: Concurrent Processing

- **WHEN** both hand tracking and image tracking are active
- **THEN** MediaPipe SHALL run at full 30 FPS
- **AND** XFeat matching SHALL run at 5 FPS
- **AND** the two processes SHALL not block each other

#### Scenario: Performance Optimization

- **GIVEN** MediaPipe requires low latency for interactive feedback
- **WHEN** system resources are constrained
- **THEN** MediaPipe 30 FPS SHALL be maintained
- **AND** XFeat MAY drop to lower frame rates if needed

---

### Requirement: Camera Source Abstraction

The system SHALL support multiple camera input sources through a unified abstraction layer.

#### Scenario: Built-in Camera (iPhone)

- **WHEN** running on iPhone
- **THEN** the system SHALL use built-in front or rear camera
- **AND** support camera switching

#### Scenario: UVC Camera (iPad/Android)

- **WHEN** an external UVC camera is connected to iPad or Android device
- **THEN** the system SHALL detect and use the UVC camera
- **AND** handle hot-plug events gracefully

#### Scenario: Fallback to Built-in

- **WHEN** no UVC camera is connected on iPad/Android
- **THEN** the system MAY fall back to built-in camera for development/testing

---

### Requirement: Hotspot Interaction

The system SHALL detect finger hover over predefined hotspot regions and trigger corresponding actions.

#### Scenario: JSON Hotspot Loading

- **WHEN** the app starts
- **THEN** hotspot data SHALL be loaded from JSON format
- **AND** NOT from SVG files

#### Scenario: Dwell Detection

- **WHEN** index finger remains within a hotspot region for ≥ 300ms
- **THEN** the system SHALL trigger the hotspot action
- **AND** prevent repeated triggers for the same hotspot until finger exits

#### Scenario: Coordinate Transformation

- **WHEN** finger position is detected in camera space
- **AND** homography matrix is available
- **THEN** finger position SHALL be transformed to reference image space

---

### Requirement: Audio Playback

The system SHALL play audio files associated with triggered hotspots using platform-native audio APIs.

#### Scenario: iOS Audio

- **WHEN** hotspot is triggered on iOS
- **THEN** corresponding audio file SHALL be played using AVAudioPlayer
- **AND** playback SHALL begin within 100ms of trigger

#### Scenario: Android Audio

- **WHEN** hotspot is triggered on Android
- **THEN** corresponding audio file SHALL be played using MediaPlayer

---

### Requirement: Device Orientation Support

The system SHALL support both landscape and portrait device orientations.

#### Scenario: Orientation Change

- **WHEN** device orientation changes
- **THEN** camera feed SHALL rotate accordingly
- **AND** coordinate transformations SHALL be updated
- **AND** UI overlay SHALL maintain correct alignment

---

### Requirement: Minimum Device Requirements

The system SHALL target devices from the last 3 years (2022+).

#### Scenario: iOS Compatibility

- **GIVEN** target devices are iPhone 12+ and iPad (9th gen+)
- **WHEN** app is installed
- **THEN** it SHALL require iOS 16.0 or later
- **AND** devices with A14 chip or later

#### Scenario: iPad UVC Support

- **GIVEN** UVC camera support is required on iPad
- **WHEN** using external camera
- **THEN** iPadOS 17.0+ SHALL be required

#### Scenario: Android Compatibility

- **GIVEN** target devices are from 2022 or later
- **WHEN** app is installed
- **THEN** it SHALL require Android API 28 (Android 9) or later
- **AND** USB Host mode for UVC camera support
