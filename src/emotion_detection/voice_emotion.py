"""
Voice Emotion Detection Module.
Uses audio features and deep learning models for emotion recognition from speech.
"""

import numpy as np
import librosa
import torch
from typing import Dict, Optional, List
from collections import deque
from datetime import datetime
import threading

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    AutoModelForAudioClassification,
    AutoFeatureExtractor
)

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)


class VoiceEmotionDetector:
    """
    Voice emotion detection using pre-trained transformers models.
    """

    # Emotion label mapping (adjust based on your model)
    EMOTION_LABELS = {
        0: 'neutral',
        1: 'calm',
        2: 'happy',
        3: 'sad',
        4: 'angry',
        5: 'fearful',
        6: 'disgusted',
        7: 'surprised'
    }

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the voice emotion detector.

        Args:
            model_name: HuggingFace model name or path
        """
        self.config = get_config()
        self.voice_config = self.config.get_section("emotion_detection")["voice"]

        self.enabled = self.voice_config["enabled"]
        self.sample_rate = self.voice_config["sample_rate"]
        self.confidence_threshold = self.voice_config["confidence_threshold"]
        self.buffer_duration = self.voice_config["buffer_duration"]

        # Model setup
        if model_name is None:
            model_name = self.voice_config.get(
                "model",
                "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            )

        self.model_name = model_name
        self.device = self._get_device()

        # Load model and feature extractor
        self.feature_extractor = None
        self.model = None
        self._load_model()

        # Audio buffer for continuous processing
        self.audio_buffer = deque(maxlen=int(self.sample_rate * self.buffer_duration))

        # Current emotion state
        self.current_emotion: Optional[Dict] = None
        self.lock = threading.Lock()

        logger.info(
            f"Voice emotion detector initialized with model: {model_name} "
            f"on device: {self.device}"
        )

    def _get_device(self) -> str:
        """Determine the compute device to use."""
        llm_device = self.config.get("llm.device", "auto")

        if llm_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return llm_device

    def _load_model(self):
        """Load the emotion recognition model."""
        try:
            logger.info(f"Loading voice emotion model: {self.model_name}")

            # Try loading as wav2vec2 model first
            try:
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    self.model_name
                )
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    self.model_name
                )
            except:
                # Fallback to auto classes
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    self.model_name
                )
                self.model = AutoModelForAudioClassification.from_pretrained(
                    self.model_name
                )

            self.model.to(self.device)
            self.model.eval()

            logger.info("Voice emotion model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load voice emotion model: {e}")
            self.enabled = False

    def detect_emotion(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Detect emotion from audio waveform.

        Args:
            audio: Audio waveform (numpy array)
            sample_rate: Audio sample rate (uses default if None)

        Returns:
            Dictionary containing:
                - dominant_emotion: Primary detected emotion
                - emotions: Dictionary of emotion probabilities
                - confidence: Confidence score
                - timestamp: Detection timestamp
        """
        if not self.enabled or self.model is None:
            return None

        try:
            # Resample if necessary
            if sample_rate is None:
                sample_rate = self.sample_rate

            if sample_rate != self.sample_rate:
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate
                )

            # Ensure minimum length
            min_length = int(self.sample_rate * 0.5)  # 0.5 seconds minimum
            if len(audio) < min_length:
                logger.debug("Audio too short for emotion detection")
                return None

            # Extract features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs = probs.cpu().numpy()[0]

            # Get dominant emotion
            dominant_idx = np.argmax(probs)
            dominant_emotion = self.EMOTION_LABELS.get(
                dominant_idx,
                f"emotion_{dominant_idx}"
            )
            confidence = float(probs[dominant_idx])

            # Check confidence threshold
            if confidence < self.confidence_threshold:
                logger.debug(
                    f"Voice emotion confidence {confidence:.2f} below threshold "
                    f"{self.confidence_threshold}"
                )
                return None

            # Build emotion dictionary
            emotions = {
                self.EMOTION_LABELS.get(i, f"emotion_{i}"): float(prob)
                for i, prob in enumerate(probs)
            }

            emotion_data = {
                'dominant_emotion': dominant_emotion,
                'emotions': emotions,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'audio_features': self._extract_audio_features(audio)
            }

            # Update current state
            with self.lock:
                self.current_emotion = emotion_data

            logger.debug(
                f"Detected voice emotion: {dominant_emotion} "
                f"(confidence: {confidence:.2f})"
            )

            return emotion_data

        except Exception as e:
            logger.error(f"Error detecting voice emotion: {e}")
            return None

    def _extract_audio_features(self, audio: np.ndarray) -> Dict:
        """
        Extract additional audio features for analysis.

        Args:
            audio: Audio waveform

        Returns:
            Dictionary of audio features
        """
        try:
            # Energy/loudness
            rms = librosa.feature.rms(y=audio)[0]
            energy = float(np.mean(rms))

            # Zero crossing rate (correlates with noisiness)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = float(np.mean(zcr))

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sample_rate
            )[0]
            spectral_centroid = float(np.mean(spectral_centroids))

            # Pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=self.sample_rate
            )
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            pitch_mean = float(np.mean(pitch_values)) if pitch_values else 0.0

            return {
                'energy': energy,
                'zero_crossing_rate': zcr_mean,
                'spectral_centroid': spectral_centroid,
                'pitch_mean': pitch_mean,
                'duration': len(audio) / self.sample_rate
            }

        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}

    def add_to_buffer(self, audio_chunk: np.ndarray):
        """
        Add audio chunk to the rolling buffer.

        Args:
            audio_chunk: Audio chunk to add
        """
        self.audio_buffer.extend(audio_chunk)

    def detect_from_buffer(self) -> Optional[Dict]:
        """
        Detect emotion from the current audio buffer.

        Returns:
            Emotion data or None
        """
        if len(self.audio_buffer) < int(self.sample_rate * 0.5):
            return None

        audio = np.array(self.audio_buffer)
        return self.detect_emotion(audio)

    def get_current_emotion(self) -> Optional[Dict]:
        """
        Get the most recently detected emotion.

        Returns:
            Current emotion data or None
        """
        with self.lock:
            return self.current_emotion.copy() if self.current_emotion else None

    def reset_buffer(self):
        """Clear the audio buffer."""
        self.audio_buffer.clear()

    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.feature_extractor is not None:
            del self.feature_extractor
            self.feature_extractor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Voice emotion detector cleaned up")


class AudioStreamProcessor:
    """
    Process audio stream in real-time for emotion detection.
    """

    def __init__(self):
        """Initialize the audio stream processor."""
        self.config = get_config()
        self.audio_config = self.config.get_section("audio")
        self.voice_config = self.config.get_section("emotion_detection")["voice"]

        self.sample_rate = self.voice_config["sample_rate"]
        self.chunk_size = self.voice_config["chunk_size"]
        self.channels = self.audio_config.get("channels", 1)

        # Initialize detector
        self.detector = VoiceEmotionDetector()

        # Stream state
        self.running = False
        self.stream = None

        logger.info("Audio stream processor initialized")

    def start_stream(self, callback: Optional[callable] = None):
        """
        Start the audio stream.

        Args:
            callback: Optional callback for emotion detection results
        """
        try:
            import pyaudio

            self.audio = pyaudio.PyAudio()

            # Get input device
            device_index = self.audio_config.get("input_device")

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback if callback else None
            )

            self.running = True
            self.stream.start_stream()

            logger.info("Audio stream started")

        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.running = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream processing."""
        import pyaudio

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize

        # Add to buffer
        self.detector.add_to_buffer(audio_data)

        return (in_data, pyaudio.paContinue)

    def stop_stream(self):
        """Stop the audio stream."""
        self.running = False

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if hasattr(self, 'audio'):
            self.audio.terminate()

        self.detector.cleanup()

        logger.info("Audio stream stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start_stream()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_stream()
