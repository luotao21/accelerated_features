"""
Hand Tracker Module

Uses MediaPipe to detect hand landmarks and extract the index finger tip position.
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. Hand tracking will be disabled.")


class HandTracker:
    """Tracks hand landmarks using MediaPipe and provides index finger tip position."""
    
    # Index finger tip landmark index
    INDEX_FINGER_TIP = 8
    
    def __init__(self, model_path: str = None, num_hands: int = 1, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the hand tracker.
        
        Args:
            model_path: Path to hand_landmarker.task model file. If None, uses default.
            num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for hand tracking.
        """
        self.enabled = MEDIAPIPE_AVAILABLE
        self.detector = None
        self.latest_result = None
        
        if not self.enabled:
            return
        
        try:
            # Use MediaPipe's built-in hand landmarker
            base_options = python.BaseOptions(
                model_asset_path=model_path if model_path else self._get_default_model_path()
            )
            
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                running_mode=vision.RunningMode.IMAGE
            )
            
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Warning: Failed to initialize MediaPipe HandLandmarker: {e}")
            print("Hand tracking will be disabled.")
            self.enabled = False
            self.detector = None
    
    def _get_default_model_path(self) -> str:
        """Get the default model path. Downloads if not present."""
        import os
        # Check common locations
        default_paths = [
            "hand_landmarker.task",
            "models/hand_landmarker.task",
            os.path.expanduser("~/.mediapipe/hand_landmarker.task")
        ]
        for path in default_paths:
            if os.path.exists(path):
                return path
        
        # If no model found, raise error with instructions
        raise FileNotFoundError(
            "hand_landmarker.task model not found. "
            "Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )
    
    def detect(self, frame: np.ndarray) -> list:
        """
        Detect hands in a frame.
        
        Args:
            frame: BGR image from OpenCV.
        
        Returns:
            List of hand landmarks, each as a list of (x, y) normalized coordinates.
        """
        if not self.enabled or self.detector is None:
            return []
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        result = self.detector.detect(mp_image)
        self.latest_result = result
        
        hands = []
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks]
                hands.append(landmarks)
        
        return hands
    
    def get_index_finger_tip(self, frame: np.ndarray) -> tuple:
        """
        Get the index finger tip position in pixel coordinates.
        
        Args:
            frame: BGR image from OpenCV.
        
        Returns:
            (x, y) pixel coordinates of index finger tip, or None if not detected.
        """
        hands = self.detect(frame)
        
        if not hands:
            return None
        
        # Get first hand's index finger tip
        hand = hands[0]
        if len(hand) > self.INDEX_FINGER_TIP:
            norm_x, norm_y = hand[self.INDEX_FINGER_TIP]
            height, width = frame.shape[:2]
            return (int(norm_x * width), int(norm_y * height))
        
        return None
    
    def transform_to_reference(self, point: tuple, H_inv: np.ndarray) -> tuple:
        """
        Transform a point from camera space to reference space using inverse homography.
        
        Args:
            point: (x, y) in camera pixel coordinates.
            H_inv: Inverse homography matrix (3x3).
        
        Returns:
            (x, y) in reference space, or None if transformation fails.
        """
        if point is None or H_inv is None:
            return None
        
        try:
            pt = np.array([[point]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, H_inv)
            return (int(transformed[0, 0, 0]), int(transformed[0, 0, 1]))
        except Exception:
            return None
    
    def cleanup(self):
        """Release resources."""
        if self.detector:
            self.detector.close()
            self.detector = None


class DwellDetector:
    """Detects when a point dwells over a region for a specified duration."""
    
    def __init__(self, dwell_time: float = 0.3, distance_threshold: int = 20):
        """
        Args:
            dwell_time: Time in seconds to trigger dwell.
            distance_threshold: Max pixel distance to consider as "same position".
        """
        self.dwell_time = dwell_time
        self.distance_threshold = distance_threshold
        self.start_time = None
        self.last_position = None
        self.triggered_hotspot = None
    
    def update(self, position: tuple, hotspot_name: str, current_time: float) -> bool:
        """
        Update dwell state and check if dwell is triggered.
        
        Args:
            position: Current (x, y) position.
            hotspot_name: Name of the hotspot being hovered, or None.
            current_time: Current timestamp.
        
        Returns:
            True if dwell was triggered for a new hotspot.
        """
        if position is None or hotspot_name is None:
            self._reset()
            return False
        
        # Check if position moved significantly
        if self.last_position is not None:
            dx = position[0] - self.last_position[0]
            dy = position[1] - self.last_position[1]
            distance = (dx * dx + dy * dy) ** 0.5
            
            if distance > self.distance_threshold:
                self._reset()
        
        self.last_position = position
        
        # Start timing if not already
        if self.start_time is None:
            self.start_time = current_time
            return False
        
        # Check if dwell time reached
        elapsed = current_time - self.start_time
        if elapsed >= self.dwell_time:
            # Only trigger once per hotspot
            if self.triggered_hotspot != hotspot_name:
                self.triggered_hotspot = hotspot_name
                return True
        
        return False
    
    def _reset(self):
        """Reset dwell state."""
        self.start_time = None
        self.last_position = None
        self.triggered_hotspot = None
