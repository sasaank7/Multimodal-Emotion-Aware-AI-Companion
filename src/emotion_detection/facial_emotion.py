"""
Facial Emotion Detection Module.
Uses DeepFace and OpenCV for real-time emotion recognition from webcam feed.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from deepface import DeepFace
import threading
from datetime import datetime

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)


class FacialEmotionDetector:
    """
    Real-time facial emotion detection using DeepFace.
    """

    def __init__(self):
        """Initialize the facial emotion detector."""
        self.config = get_config()
        self.face_config = self.config.get_section("emotion_detection")["face"]

        self.enabled = self.face_config["enabled"]
        self.backend = self.face_config["backend"]
        self.detector_backend = self.face_config["detector"]
        self.confidence_threshold = self.face_config["confidence_threshold"]

        # Current emotion state
        self.current_emotion: Optional[Dict] = None
        self.last_detection_time: Optional[datetime] = None

        # Threading for async processing
        self.lock = threading.Lock()
        self.processing = False

        logger.info(
            f"Facial emotion detector initialized with backend: {self.backend}"
        )

    def detect_emotion(
        self,
        frame: np.ndarray,
        return_face_region: bool = False
    ) -> Optional[Dict]:
        """
        Detect emotion from a video frame.

        Args:
            frame: Video frame (numpy array in BGR format)
            return_face_region: Whether to return face region coordinates

        Returns:
            Dictionary containing:
                - dominant_emotion: The primary detected emotion
                - emotions: Dictionary of all emotion probabilities
                - confidence: Confidence score
                - region: Face region coordinates (if return_face_region=True)
                - timestamp: Detection timestamp
        """
        if not self.enabled:
            return None

        try:
            # Analyze the frame
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True
            )

            # Handle single face or multiple faces
            if isinstance(result, list):
                if len(result) == 0:
                    return None
                result = result[0]  # Use first face if multiple detected

            # Extract emotion data
            emotions = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', 'neutral')

            # Calculate confidence (max probability)
            confidence = max(emotions.values()) / 100.0 if emotions else 0.0

            # Check confidence threshold
            if confidence < self.confidence_threshold:
                logger.debug(
                    f"Emotion confidence {confidence:.2f} below threshold "
                    f"{self.confidence_threshold}"
                )
                return None

            # Build response
            emotion_data = {
                'dominant_emotion': dominant_emotion,
                'emotions': {k: v / 100.0 for k, v in emotions.items()},
                'confidence': confidence,
                'timestamp': datetime.now()
            }

            # Add face region if requested
            if return_face_region and 'region' in result:
                emotion_data['region'] = result['region']

            # Update current state
            with self.lock:
                self.current_emotion = emotion_data
                self.last_detection_time = emotion_data['timestamp']

            logger.debug(
                f"Detected emotion: {dominant_emotion} "
                f"(confidence: {confidence:.2f})"
            )

            return emotion_data

        except Exception as e:
            logger.error(f"Error detecting facial emotion: {e}")
            return None

    def detect_emotion_async(
        self,
        frame: np.ndarray,
        callback: Optional[callable] = None
    ):
        """
        Detect emotion asynchronously in a separate thread.

        Args:
            frame: Video frame
            callback: Optional callback function to call with result
        """
        def process():
            self.processing = True
            result = self.detect_emotion(frame)
            self.processing = False

            if callback and result:
                callback(result)

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def get_current_emotion(self) -> Optional[Dict]:
        """
        Get the most recently detected emotion.

        Returns:
            Current emotion data or None
        """
        with self.lock:
            return self.current_emotion.copy() if self.current_emotion else None

    def draw_emotion_overlay(
        self,
        frame: np.ndarray,
        emotion_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Draw emotion information as an overlay on the frame.

        Args:
            frame: Video frame
            emotion_data: Emotion data to display (uses current if None)

        Returns:
            Frame with emotion overlay
        """
        if emotion_data is None:
            emotion_data = self.get_current_emotion()

        if emotion_data is None:
            return frame

        # Create a copy to avoid modifying original
        overlay_frame = frame.copy()

        # Get emotion colors from config
        emotion_colors = self.config.get(
            "dashboard.emotion_colors",
            {}
        )

        dominant = emotion_data['dominant_emotion']
        confidence = emotion_data['confidence']
        emotions = emotion_data['emotions']

        # Draw emotion label
        label = f"{dominant.upper()} ({confidence:.1%})"
        color_hex = emotion_colors.get(dominant, "#FFFFFF")
        color_bgr = self._hex_to_bgr(color_hex)

        cv2.putText(
            overlay_frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color_bgr,
            2,
            cv2.LINE_AA
        )

        # Draw emotion bar chart
        y_start = 60
        bar_height = 20
        max_bar_width = 200

        sorted_emotions = sorted(
            emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for i, (emotion, prob) in enumerate(sorted_emotions[:5]):
            y = y_start + i * (bar_height + 5)

            # Emotion label
            cv2.putText(
                overlay_frame,
                f"{emotion}:",
                (10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Probability bar
            bar_width = int(prob * max_bar_width)
            color_hex = emotion_colors.get(emotion, "#808080")
            color_bgr = self._hex_to_bgr(color_hex)

            cv2.rectangle(
                overlay_frame,
                (120, y),
                (120 + bar_width, y + bar_height),
                color_bgr,
                -1
            )

            # Percentage text
            cv2.putText(
                overlay_frame,
                f"{prob:.1%}",
                (120 + max_bar_width + 10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Draw face region if available
        if 'region' in emotion_data:
            region = emotion_data['region']
            x, y, w, h = (
                region['x'],
                region['y'],
                region['w'],
                region['h']
            )
            cv2.rectangle(
                overlay_frame,
                (x, y),
                (x + w, y + h),
                color_bgr,
                2
            )

        return overlay_frame

    @staticmethod
    def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        """
        Convert hex color to BGR tuple for OpenCV.

        Args:
            hex_color: Hex color string (e.g., '#FF0000')

        Returns:
            BGR color tuple
        """
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)  # OpenCV uses BGR

    def cleanup(self):
        """Clean up resources."""
        logger.info("Facial emotion detector cleaned up")


class WebcamEmotionStream:
    """
    Manage webcam stream with real-time emotion detection.
    """

    def __init__(self, camera_index: int = 0):
        """
        Initialize webcam stream.

        Args:
            camera_index: Camera device index
        """
        self.config = get_config()
        self.camera_index = camera_index

        # Video settings
        video_config = self.config.get_section("video")
        self.resolution = (
            video_config["resolution"]["width"],
            video_config["resolution"]["height"]
        )
        self.fps = video_config.get("fps", 30)

        # Initialize detector
        self.detector = FacialEmotionDetector()

        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False

        logger.info(f"Webcam emotion stream initialized (camera {camera_index})")

    def start(self) -> bool:
        """
        Start the webcam stream.

        Returns:
            True if started successfully
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if not self.cap.isOpened():
                logger.error("Failed to open webcam")
                return False

            self.running = True
            logger.info("Webcam stream started")
            return True

        except Exception as e:
            logger.error(f"Error starting webcam: {e}")
            return False

    def read_frame(self) -> Optional[Tuple[np.ndarray, Optional[Dict]]]:
        """
        Read a frame and detect emotion.

        Returns:
            Tuple of (frame, emotion_data) or None if failed
        """
        if not self.running or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from webcam")
            return None

        # Detect emotion
        emotion_data = self.detector.detect_emotion(frame, return_face_region=True)

        return frame, emotion_data

    def read_frame_with_overlay(self) -> Optional[np.ndarray]:
        """
        Read frame with emotion overlay drawn.

        Returns:
            Frame with overlay or None
        """
        result = self.read_frame()
        if result is None:
            return None

        frame, emotion_data = result

        # Draw overlay
        if emotion_data:
            frame = self.detector.draw_emotion_overlay(frame, emotion_data)

        return frame

    def stop(self):
        """Stop the webcam stream."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.detector.cleanup()
        logger.info("Webcam stream stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
