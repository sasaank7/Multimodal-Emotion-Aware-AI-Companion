"""
Speech-to-Text Engine for Voice Input.
Real-time speech recognition with continuous listening.
"""

import numpy as np
import torch
from typing import Optional, Callable, Dict
import queue
import threading
import time
from datetime import datetime

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)


class SpeechToText:
    """
    Speech-to-Text with real-time transcription.
    """

    def __init__(
        self,
        engine: str = 'google',
        model_name: Optional[str] = None,
        language: str = 'en-US'
    ):
        """
        Initialize STT engine.

        Args:
            engine: STT engine ('google', 'whisper', 'sphinx')
            model_name: Model name for Whisper
            language: Language code
        """
        self.config = get_config()
        self.stt_config = self.config.get('stt', {})

        self.engine_type = engine
        self.language = language

        # Initialize recognizer
        if not SR_AVAILABLE:
            raise RuntimeError("speech_recognition not available. Install with: pip install SpeechRecognition")

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Whisper model (if using)
        self.whisper_model = None
        if engine == 'whisper':
            self._initialize_whisper(model_name)

        # Adjust for ambient noise
        logger.info("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        logger.info(f"STT engine initialized with {engine}")

    def _initialize_whisper(self, model_name: Optional[str]):
        """Initialize Whisper model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available for Whisper. Falling back to Google.")
            self.engine_type = 'google'
            return

        try:
            if model_name is None:
                model_name = "openai/whisper-base"

            logger.info(f"Loading Whisper model: {model_name}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.whisper_model = pipeline(
                "automatic-speech-recognition",
                model=model_name,
                device=device
            )

            logger.info("Whisper model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            logger.info("Falling back to Google STT")
            self.engine_type = 'google'

    def transcribe_audio(
        self,
        audio_data: sr.AudioData,
        language: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Transcribe audio to text.

        Args:
            audio_data: Audio data from microphone
            language: Language code (uses default if None)

        Returns:
            Dictionary with transcription and metadata
        """
        if language is None:
            language = self.language

        try:
            start_time = time.time()

            # Transcribe based on engine
            if self.engine_type == 'google':
                text = self.recognizer.recognize_google(audio_data, language=language)
            elif self.engine_type == 'whisper':
                text = self._transcribe_whisper(audio_data)
            elif self.engine_type == 'sphinx':
                text = self.recognizer.recognize_sphinx(audio_data)
            else:
                logger.error(f"Unknown STT engine: {self.engine_type}")
                return None

            duration = time.time() - start_time

            result = {
                'text': text,
                'confidence': 1.0,  # Not all engines provide confidence
                'duration': duration,
                'timestamp': datetime.now(),
                'language': language,
                'engine': self.engine_type
            }

            logger.debug(f"Transcribed: '{text}' ({duration:.2f}s)")

            return result

        except sr.UnknownValueError:
            logger.debug("Speech not understood")
            return None
        except sr.RequestError as e:
            logger.error(f"STT request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def _transcribe_whisper(self, audio_data: sr.AudioData) -> str:
        """Transcribe using Whisper model."""
        # Convert audio data to numpy array
        audio_array = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize

        # Resample if necessary (Whisper expects 16kHz)
        sample_rate = audio_data.sample_rate
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=16000
            )

        # Transcribe
        result = self.whisper_model(audio_array)
        return result['text']

    def listen_once(
        self,
        timeout: Optional[float] = None,
        phrase_time_limit: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Listen for a single phrase and transcribe.

        Args:
            timeout: Maximum time to wait for phrase start
            phrase_time_limit: Maximum phrase duration

        Returns:
            Transcription result
        """
        try:
            logger.info("Listening...")

            with self.microphone as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

            logger.info("Processing speech...")
            return self.transcribe_audio(audio)

        except sr.WaitTimeoutError:
            logger.debug("Listening timed out")
            return None
        except Exception as e:
            logger.error(f"Listen error: {e}")
            return None

    def listen_continuously(
        self,
        callback: Callable[[Dict], None],
        stop_event: Optional[threading.Event] = None
    ):
        """
        Listen continuously and call callback with transcriptions.

        Args:
            callback: Function to call with transcription results
            stop_event: Event to signal stopping
        """
        if stop_event is None:
            stop_event = threading.Event()

        def audio_callback(recognizer, audio):
            """Background callback for continuous listening."""
            try:
                result = self.transcribe_audio(audio)
                if result:
                    callback(result)
            except Exception as e:
                logger.error(f"Transcription callback error: {e}")

        # Start background listening
        stop_listening = self.recognizer.listen_in_background(
            self.microphone,
            audio_callback,
            phrase_time_limit=10
        )

        logger.info("Continuous listening started")

        try:
            # Wait until stop event is set
            stop_event.wait()
        finally:
            stop_listening(wait_for_stop=False)
            logger.info("Continuous listening stopped")

    def get_energy_threshold(self) -> float:
        """Get current energy threshold."""
        return self.recognizer.energy_threshold

    def set_energy_threshold(self, threshold: float):
        """Set energy threshold for voice detection."""
        self.recognizer.energy_threshold = threshold
        logger.info(f"Energy threshold set to: {threshold}")

    def calibrate(self, duration: float = 1.0):
        """
        Calibrate for ambient noise.

        Args:
            duration: Calibration duration in seconds
        """
        logger.info(f"Calibrating for {duration}s...")

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)

        logger.info(f"Calibration complete. Threshold: {self.recognizer.energy_threshold}")


class VoiceCommandProcessor:
    """
    Process voice commands and integrate with companion.
    """

    # Wake words
    WAKE_WORDS = ['hey companion', 'hello companion', 'companion']

    def __init__(self, stt_engine: Optional[SpeechToText] = None):
        """
        Initialize voice command processor.

        Args:
            stt_engine: STT engine (creates new if None)
        """
        self.stt = stt_engine or SpeechToText()
        self.is_listening = False
        self.stop_event = threading.Event()
        self.transcription_queue = queue.Queue()

        logger.info("Voice command processor initialized")

    def start_listening(self, callback: Callable[[str], None]):
        """
        Start listening for voice commands.

        Args:
            callback: Function to call with transcribed text
        """
        if self.is_listening:
            logger.warning("Already listening")
            return

        self.is_listening = True
        self.stop_event.clear()

        def transcription_callback(result: Dict):
            """Handle transcription result."""
            text = result.get('text', '').strip()
            if text:
                logger.info(f"Voice input: {text}")
                callback(text)

        # Start continuous listening in a thread
        self.listen_thread = threading.Thread(
            target=self.stt.listen_continuously,
            args=(transcription_callback, self.stop_event),
            daemon=True
        )
        self.listen_thread.start()

        logger.info("Voice listening started")

    def stop_listening(self):
        """Stop listening for voice commands."""
        if not self.is_listening:
            return

        self.stop_event.set()
        self.is_listening = False

        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=2)

        logger.info("Voice listening stopped")

    def listen_for_wake_word(self, timeout: float = 30.0) -> bool:
        """
        Listen for wake word.

        Args:
            timeout: Maximum wait time

        Returns:
            True if wake word detected
        """
        logger.info("Listening for wake word...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            result = self.stt.listen_once(timeout=5.0, phrase_time_limit=3.0)

            if result:
                text = result['text'].lower()

                for wake_word in self.WAKE_WORDS:
                    if wake_word in text:
                        logger.info(f"Wake word detected: {wake_word}")
                        return True

        logger.info("Wake word timeout")
        return False

    def process_voice_command(self) -> Optional[str]:
        """
        Listen for and process a voice command.

        Returns:
            Transcribed command text
        """
        result = self.stt.listen_once(timeout=10.0, phrase_time_limit=10.0)

        if result:
            return result['text']

        return None


class VoiceActivityDetector:
    """
    Detect voice activity in audio stream.
    """

    def __init__(self, energy_threshold: float = 300):
        """
        Initialize VAD.

        Args:
            energy_threshold: Energy threshold for voice detection
        """
        self.energy_threshold = energy_threshold
        logger.info(f"VAD initialized with threshold: {energy_threshold}")

    def is_speech(self, audio_data: np.ndarray) -> bool:
        """
        Detect if audio contains speech.

        Args:
            audio_data: Audio array

        Returns:
            True if speech detected
        """
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        energy = rms * 32768  # Scale to int16 range

        return energy > self.energy_threshold

    def get_speech_segments(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        frame_duration: float = 0.03
    ) -> list:
        """
        Get speech segments from audio.

        Args:
            audio_data: Audio array
            sample_rate: Sample rate
            frame_duration: Frame duration in seconds

        Returns:
            List of (start, end) tuples for speech segments
        """
        frame_length = int(sample_rate * frame_duration)
        segments = []
        in_speech = False
        segment_start = 0

        for i in range(0, len(audio_data), frame_length):
            frame = audio_data[i:i + frame_length]

            if len(frame) < frame_length:
                break

            is_speech = self.is_speech(frame)

            if is_speech and not in_speech:
                # Start of speech segment
                segment_start = i / sample_rate
                in_speech = True

            elif not is_speech and in_speech:
                # End of speech segment
                segment_end = i / sample_rate
                segments.append((segment_start, segment_end))
                in_speech = False

        # Handle case where speech continues to end
        if in_speech:
            segments.append((segment_start, len(audio_data) / sample_rate))

        return segments
