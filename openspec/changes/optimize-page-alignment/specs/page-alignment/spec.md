## ADDED Requirements

### Requirement: Dual Homography for Folded Books

The system SHALL support computing separate homography matrices for left and right book pages when the book is not fully opened to 180°.

#### Scenario: Book partially opened

- **WHEN** the book is opened at an angle less than 180°
- **AND** XFeat detects sufficient feature points on both pages
- **THEN** the system SHALL compute two independent homographies (H_left, H_right)
- **AND** warp hotspots using the appropriate homography based on their x-position

#### Scenario: Insufficient features on one side

- **WHEN** one page has fewer than 15 matched feature points
- **THEN** the system SHALL fall back to a single homography for the entire spread

### Requirement: Configurable Spine Position

The system SHALL allow configuration of the book spine position for partitioning feature points and hotspots.

#### Scenario: Default spine position

- **WHEN** no spine position is specified
- **THEN** the system SHALL use the horizontal center of the reference image

#### Scenario: Custom spine position

- **GIVEN** a `--spine_position` argument is provided (0.0 to 1.0)
- **WHEN** the system starts
- **THEN** the spine position SHALL be set to that fraction of the image width
