"""
Text-to-Speech Engine with Emotional Tone.
Synthesizes speech with emotion-aware prosody and tone.
"""

import torch
import numpy as np
from typing import Optional, Dict, Literal
import io
import soundfile as sf
from datetime import datetime

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)


class EmotionalTTS:
    """
    Text-to-Speech with emotion-aware voice synthesis.
    """

    # Emotion to voice parameter mapping
    EMOTION_VOICE_PARAMS = {
        'happy': {'rate': 1.15, 'pitch': 1.1, 'energy': 1.2},
        'sad': {'rate': 0.85, 'pitch': 0.9, 'energy': 0.7},
        'angry': {'rate': 1.2, 'pitch': 1.15, 'energy': 1.3},
        'fearful': {'rate': 1.1, 'pitch': 1.05, 'energy': 0.9},
        'calm': {'rate': 0.9, 'pitch': 0.95, 'energy': 0.8},
        'neutral': {'rate': 1.0, 'pitch': 1.0, 'energy': 1.0},
        'excited': {'rate': 1.25, 'pitch': 1.2, 'energy': 1.4},
        'supportive': {'rate': 0.95, 'pitch': 1.0, 'energy': 1.0},
        'cheerful': {'rate': 1.1, 'pitch': 1.15, 'energy': 1.2}
    }

    def __init__(
        self,
        engine: Literal['coqui', 'pyttsx3', 'auto'] = 'auto',
        model_name: Optional[str] = None
    ):
        """
        Initialize the TTS engine.

        Args:
            engine: TTS engine to use ('coqui', 'pyttsx3', 'auto')
            model_name: Specific model name for Coqui TTS
        """
        self.config = get_config()
        self.tts_config = self.config.get('tts', {})

        # Determine which engine to use
        if engine == 'auto':
            if TTS_AVAILABLE:
                engine = 'coqui'
            elif PYTTSX3_AVAILABLE:
                engine = 'pyttsx3'
            else:
                raise RuntimeError("No TTS engine available. Install TTS or pyttsx3.")

        self.engine_type = engine
        self.tts_engine = None

        # Initialize the selected engine
        self._initialize_engine(model_name)

        logger.info(f"Emotional TTS initialized with {self.engine_type} engine")

    def _initialize_engine(self, model_name: Optional[str]):
        """Initialize the TTS engine."""
        if self.engine_type == 'coqui':
            self._initialize_coqui(model_name)
        elif self.engine_type == 'pyttsx3':
            self._initialize_pyttsx3()

    def _initialize_coqui(self, model_name: Optional[str]):
        """Initialize Coqui TTS engine."""
        if not TTS_AVAILABLE:
            raise RuntimeError("Coqui TTS not available. Install with: pip install TTS")

        try:
            if model_name is None:
                # Use a good default model
                model_name = "tts_models/en/ljspeech/tacotron2-DDC"

            logger.info(f"Loading Coqui TTS model: {model_name}")

            # Initialize TTS
            self.tts_engine = TTS(model_name=model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.tts_engine.to('cuda')

            logger.info("Coqui TTS engine loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Coqui TTS: {e}")
            raise

    def _initialize_pyttsx3(self):
        """Initialize pyttsx3 engine."""
        if not PYTTSX3_AVAILABLE:
            raise RuntimeError("pyttsx3 not available. Install with: pip install pyttsx3")

        try:
            self.tts_engine = pyttsx3.init()

            # Set default properties
            self.tts_engine.setProperty('rate', 150)  # Speed
            self.tts_engine.setProperty('volume', 0.9)  # Volume

            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break

            logger.info("pyttsx3 engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            raise

    def synthesize(
        self,
        text: str,
        emotion: Optional[str] = None,
        save_path: Optional[str] = None,
        return_audio: bool = True
    ) -> Optional[np.ndarray]:
        """
        Synthesize speech from text with emotional tone.

        Args:
            text: Text to synthesize
            emotion: Emotion to apply (happy, sad, angry, etc.)
            save_path: Path to save audio file
            return_audio: Whether to return audio array

        Returns:
            Audio array (numpy) if return_audio=True
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None

        try:
            # Get emotion parameters
            emotion_params = self._get_emotion_params(emotion)

            if self.engine_type == 'coqui':
                return self._synthesize_coqui(
                    text, emotion_params, save_path, return_audio
                )
            elif self.engine_type == 'pyttsx3':
                return self._synthesize_pyttsx3(
                    text, emotion_params, save_path, return_audio
                )

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

    def _get_emotion_params(self, emotion: Optional[str]) -> Dict:
        """Get voice parameters for emotion."""
        if emotion is None:
            emotion = 'neutral'

        emotion = emotion.lower()
        return self.EMOTION_VOICE_PARAMS.get(emotion, self.EMOTION_VOICE_PARAMS['neutral'])

    def _synthesize_coqui(
        self,
        text: str,
        emotion_params: Dict,
        save_path: Optional[str],
        return_audio: bool
    ) -> Optional[np.ndarray]:
        """Synthesize using Coqui TTS."""
        try:
            # Generate speech
            wav = self.tts_engine.tts(text)

            # Convert to numpy array
            if isinstance(wav, list):
                wav = np.array(wav, dtype=np.float32)

            # Apply emotion parameters
            wav = self._apply_emotion_effects(wav, emotion_params)

            # Save if requested
            if save_path:
                sf.write(save_path, wav, 22050)
                logger.info(f"Audio saved to: {save_path}")

            return wav if return_audio else None

        except Exception as e:
            logger.error(f"Coqui TTS synthesis failed: {e}")
            return None

    def _synthesize_pyttsx3(
        self,
        text: str,
        emotion_params: Dict,
        save_path: Optional[str],
        return_audio: bool
    ) -> Optional[np.ndarray]:
        """Synthesize using pyttsx3."""
        try:
            # Apply emotion parameters to voice
            base_rate = 150
            new_rate = int(base_rate * emotion_params['rate'])

            self.tts_engine.setProperty('rate', new_rate)

            # Generate speech
            if save_path:
                self.tts_engine.save_to_file(text, save_path)
                self.tts_engine.runAndWait()
                logger.info(f"Audio saved to: {save_path}")

                # Load and return if requested
                if return_audio:
                    audio, sr = sf.read(save_path)
                    return audio
            else:
                # Speak directly (for real-time use)
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()

            return None

        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return None

    def _apply_emotion_effects(
        self,
        audio: np.ndarray,
        emotion_params: Dict
    ) -> np.ndarray:
        """
        Apply emotion-based effects to audio.

        Args:
            audio: Input audio array
            emotion_params: Emotion parameters

        Returns:
            Modified audio array
        """
        try:
            import librosa

            # Time stretching (rate)
            if emotion_params['rate'] != 1.0:
                audio = librosa.effects.time_stretch(
                    audio,
                    rate=1.0 / emotion_params['rate']
                )

            # Pitch shifting
            if emotion_params['pitch'] != 1.0:
                n_steps = (emotion_params['pitch'] - 1.0) * 12  # semitones
                audio = librosa.effects.pitch_shift(
                    audio,
                    sr=22050,
                    n_steps=n_steps
                )

            # Energy/volume adjustment
            audio = audio * emotion_params['energy']

            # Normalize
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val

            return audio

        except Exception as e:
            logger.warning(f"Failed to apply emotion effects: {e}")
            return audio

    def speak_with_emotion(
        self,
        text: str,
        emotion: Optional[str] = None
    ):
        """
        Speak text immediately with emotion (blocking).

        Args:
            text: Text to speak
            emotion: Emotion to apply
        """
        if self.engine_type == 'pyttsx3':
            emotion_params = self._get_emotion_params(emotion)
            base_rate = 150
            new_rate = int(base_rate * emotion_params['rate'])

            self.tts_engine.setProperty('rate', new_rate)
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        else:
            # For Coqui, generate and play
            import sounddevice as sd

            audio = self.synthesize(text, emotion, return_audio=True)
            if audio is not None:
                sd.play(audio, 22050)
                sd.wait()

    def get_available_voices(self) -> list:
        """
        Get list of available voices.

        Returns:
            List of voice information
        """
        if self.engine_type == 'pyttsx3':
            voices = self.tts_engine.getProperty('voices')
            return [
                {
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages
                }
                for voice in voices
            ]
        elif self.engine_type == 'coqui':
            # Return available Coqui models
            return TTS().list_models()

        return []

    def set_voice(self, voice_id: str):
        """
        Set the voice to use.

        Args:
            voice_id: Voice identifier
        """
        if self.engine_type == 'pyttsx3':
            self.tts_engine.setProperty('voice', voice_id)
            logger.info(f"Voice changed to: {voice_id}")

    def cleanup(self):
        """Clean up resources."""
        if self.engine_type == 'pyttsx3' and self.tts_engine:
            self.tts_engine.stop()

        if self.engine_type == 'coqui' and self.tts_engine:
            del self.tts_engine

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("TTS engine cleaned up")


class EmotionAwareVoiceResponse:
    """
    Combines LLM response generation with emotional TTS.
    """

    def __init__(self):
        """Initialize emotion-aware voice response system."""
        self.tts = EmotionalTTS()
        logger.info("Emotion-aware voice response system initialized")

    def generate_and_speak(
        self,
        response_text: str,
        emotion_context: Optional[Dict] = None,
        response_mode: Optional[str] = None
    ):
        """
        Generate speech with appropriate emotion.

        Args:
            response_text: Text to speak
            emotion_context: Current emotion context
            response_mode: Response mode (supportive, cheerful, etc.)
        """
        # Determine emotion for TTS
        emotion = self._select_tts_emotion(emotion_context, response_mode)

        logger.info(f"Speaking with emotion: {emotion}")

        # Speak
        self.tts.speak_with_emotion(response_text, emotion)

    def _select_tts_emotion(
        self,
        emotion_context: Optional[Dict],
        response_mode: Optional[str]
    ) -> str:
        """
        Select appropriate TTS emotion.

        Args:
            emotion_context: Emotion detection result
            response_mode: Response mode

        Returns:
            Emotion name for TTS
        """
        # Map response mode to TTS emotion
        mode_to_emotion = {
            'supportive': 'supportive',
            'cheerful': 'cheerful',
            'calm': 'calm',
            'concise': 'neutral',
            'neutral': 'neutral'
        }

        if response_mode and response_mode in mode_to_emotion:
            return mode_to_emotion[response_mode]

        # Fallback to neutral
        return 'neutral'

    def cleanup(self):
        """Clean up resources."""
        self.tts.cleanup()
