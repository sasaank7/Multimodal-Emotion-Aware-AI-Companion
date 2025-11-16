"""
Adaptive Response System.
Manages response generation based on multimodal emotion signals and user context.
"""

from typing import Dict, Optional, List
from datetime import datetime

from .emotion_aware_llm import EmotionAwareLLM, determine_response_mode
from ..emotion_detection.multimodal_fusion import MultimodalEmotionFusion
from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)


class AdaptiveResponseSystem:
    """
    Coordinates emotion detection, fusion, and adaptive response generation.
    """

    def __init__(self):
        """Initialize the adaptive response system."""
        self.config = get_config()
        self.adaptive_config = self.config.get_section("adaptive_response")

        # Initialize components
        self.emotion_fusion = MultimodalEmotionFusion()
        self.llm = EmotionAwareLLM()

        # Response mode tracking
        self.current_mode: str = 'neutral'
        self.mode_history: List[Dict] = []

        # Context tracking
        self.context_window = self.adaptive_config.get('context_window', 10)
        self.mood_sensitivity = self.adaptive_config.get('mood_sensitivity', 0.3)

        logger.info("Adaptive response system initialized")

    def generate_response(
        self,
        user_message: str,
        face_emotion: Optional[Dict] = None,
        voice_emotion: Optional[Dict] = None,
        text_emotion: Optional[Dict] = None
    ) -> Dict:
        """
        Generate an adaptive response based on multimodal emotions.

        Args:
            user_message: User's text message
            face_emotion: Facial emotion data
            voice_emotion: Voice emotion data
            text_emotion: Text sentiment data

        Returns:
            Dictionary containing:
                - response: Generated response text
                - emotion_context: Fused emotion data
                - response_mode: Selected response mode
                - confidence: Overall confidence
                - metadata: Additional information
        """
        try:
            # Fuse multimodal emotions
            fused_emotion = self.emotion_fusion.fuse_emotions(
                face_emotion=face_emotion,
                voice_emotion=voice_emotion,
                text_emotion=text_emotion
            )

            # Enhance with temporal context
            if fused_emotion:
                temporal_emotion = self.emotion_fusion.get_temporal_emotion(
                    window_seconds=10
                )
                if temporal_emotion:
                    fused_emotion['temporal_context'] = temporal_emotion

                # Check for emotion transitions
                transition = self.emotion_fusion.get_emotion_transition()
                if transition:
                    fused_emotion['transition'] = transition

                # Get stability
                stability = self.emotion_fusion.get_emotion_stability()
                fused_emotion['stability'] = stability

            # Determine response mode
            response_mode = self._select_response_mode(fused_emotion)

            # Adjust LLM generation parameters
            self.llm.adjust_generation_params(response_mode)

            # Generate response
            response_text = self.llm.generate_response(
                user_message=user_message,
                emotion_context=fused_emotion,
                response_mode=response_mode,
                include_history=True
            )

            # Build result
            result = {
                'response': response_text,
                'emotion_context': fused_emotion,
                'response_mode': response_mode,
                'confidence': fused_emotion.get('confidence', 0.0) if fused_emotion else 0.0,
                'metadata': {
                    'timestamp': datetime.now(),
                    'user_message': user_message,
                    'modalities_used': fused_emotion.get('modalities_used', []) if fused_emotion else []
                }
            }

            # Track mode change
            self._track_mode_change(response_mode, fused_emotion)

            logger.info(
                f"Generated adaptive response (mode: {response_mode}, "
                f"emotion: {fused_emotion.get('dominant_emotion', 'none') if fused_emotion else 'none'})"
            )

            return result

        except Exception as e:
            logger.error(f"Error generating adaptive response: {e}")
            return {
                'response': "I'm here to help. Could you tell me more?",
                'emotion_context': None,
                'response_mode': 'neutral',
                'confidence': 0.0,
                'metadata': {
                    'error': str(e),
                    'timestamp': datetime.now()
                }
            }

    def _select_response_mode(
        self,
        emotion_data: Optional[Dict]
    ) -> str:
        """
        Select appropriate response mode based on emotion and context.

        Args:
            emotion_data: Fused emotion data

        Returns:
            Selected response mode
        """
        # Determine base mode from emotion
        base_mode = determine_response_mode(emotion_data)

        # Check if mode should change based on sensitivity
        if self._should_change_mode(emotion_data, base_mode):
            self.current_mode = base_mode

        return self.current_mode

    def _should_change_mode(
        self,
        emotion_data: Optional[Dict],
        proposed_mode: str
    ) -> bool:
        """
        Determine if response mode should change.

        Args:
            emotion_data: Current emotion data
            proposed_mode: Proposed new mode

        Returns:
            True if mode should change
        """
        # Always change if no current mode
        if not self.current_mode:
            return True

        # Change if proposed mode is different and emotion is stable enough
        if proposed_mode != self.current_mode:
            if emotion_data is None:
                return False

            # Check emotion stability
            stability = emotion_data.get('stability', 0.5)
            confidence = emotion_data.get('confidence', 0.0)

            # Change if emotion is strong and stable
            if confidence >= self.mood_sensitivity and stability >= 0.5:
                return True

        return False

    def _track_mode_change(
        self,
        mode: str,
        emotion_data: Optional[Dict]
    ):
        """
        Track response mode changes.

        Args:
            mode: Current mode
            emotion_data: Current emotion data
        """
        # Check if mode changed
        if not self.mode_history or self.mode_history[-1]['mode'] != mode:
            self.mode_history.append({
                'mode': mode,
                'emotion': emotion_data.get('dominant_emotion', 'unknown') if emotion_data else 'unknown',
                'timestamp': datetime.now()
            })

            # Trim history
            if len(self.mode_history) > 50:
                self.mode_history = self.mode_history[-50:]

            logger.debug(f"Response mode changed to: {mode}")

    def get_emotion_summary(self, window_seconds: int = 60) -> Dict:
        """
        Get emotion summary over time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Emotion summary
        """
        return self.emotion_fusion.get_emotion_summary(window_seconds)

    def get_mode_history(self) -> List[Dict]:
        """
        Get response mode history.

        Returns:
            List of mode changes
        """
        return self.mode_history.copy()

    def clear_history(self):
        """Clear all history."""
        self.emotion_fusion.clear_history()
        self.llm.clear_history()
        self.mode_history.clear()
        logger.info("Adaptive response system history cleared")

    def cleanup(self):
        """Clean up resources."""
        self.llm.cleanup()
        logger.info("Adaptive response system cleaned up")


class ResponseRecommendations:
    """
    Generate recommendations based on emotional state.
    """

    RECOMMENDATIONS = {
        'sad': [
            "Would you like me to suggest some uplifting activities?",
            "Sometimes it helps to talk about what's bothering you. I'm here to listen.",
            "Would you like me to recommend some calming music or content?"
        ],
        'angry': [
            "Would you like to try a brief breathing exercise?",
            "Taking a short break might help. Would you like some suggestions?",
            "I'm here to help you work through this. What would be most helpful?"
        ],
        'stressed': [
            "Would you like me to suggest a relaxation technique?",
            "It might help to break down what's overwhelming you. Shall we try?",
            "Would you like to set a reminder to take a break?"
        ],
        'anxious': [
            "Let's try a grounding exercise. Would that help?",
            "Would you like to talk through what's making you anxious?",
            "Sometimes structure helps. Would you like to make a plan together?"
        ],
        'tired': [
            "Would you like me to suggest some energizing activities?",
            "It might be time for a break. Would you like help planning one?",
            "Would you like to set a reminder to rest?"
        ]
    }

    @staticmethod
    def get_recommendations(
        emotion: str,
        confidence: float = 0.7
    ) -> Optional[List[str]]:
        """
        Get recommendations for an emotional state.

        Args:
            emotion: Detected emotion
            confidence: Confidence threshold

        Returns:
            List of recommendations or None
        """
        if confidence < 0.6:
            return None

        # Map emotions to recommendation categories
        emotion_map = {
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'anxious',
            'neutral': None,
            'happy': None,
            'surprised': None
        }

        category = emotion_map.get(emotion)
        if category is None:
            return None

        return ResponseRecommendations.RECOMMENDATIONS.get(category, [])

    @staticmethod
    def format_recommendation(
        recommendations: List[str],
        style: str = 'gentle'
    ) -> str:
        """
        Format recommendations for inclusion in response.

        Args:
            recommendations: List of recommendation texts
            style: Formatting style

        Returns:
            Formatted recommendation text
        """
        if not recommendations:
            return ""

        if style == 'gentle':
            return "\n\n" + recommendations[0]
        elif style == 'list':
            return "\n\nI have a few suggestions:\n" + "\n".join(
                f"- {rec}" for rec in recommendations
            )
        else:
            return "\n\n" + recommendations[0]
