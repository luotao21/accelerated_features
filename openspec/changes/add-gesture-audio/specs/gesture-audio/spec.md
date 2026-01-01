## ADDED Requirements

### Requirement: Hand Tracking Integration

The system SHALL detect the user's index finger tip position using MediaPipe HandLandmarker and transform the coordinates to the reference image space using the inverse homography matrix.

#### Scenario: Finger detected with valid homography

- **WHEN** the camera captures a hand with visible index finger
- **AND** a valid homography matrix exists
- **THEN** the finger tip position SHALL be transformed to reference coordinates

### Requirement: Dwell-Triggered Audio Playback

The system SHALL play the corresponding audio file when the user's finger dwells over a hotspot for at least 0.3 seconds.

#### Scenario: Audio triggered on dwell

- **WHEN** the finger tip remains over a hotspot for 0.3 seconds
- **THEN** the audio file matching the hotspot name SHALL be played

#### Scenario: No audio on quick pass

- **WHEN** the finger moves across a hotspot in less than 0.3 seconds
- **THEN** no audio SHALL be played
