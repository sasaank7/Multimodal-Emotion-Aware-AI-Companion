"""
Multimodal Fusion Module.
Combines facial, voice, and text emotion signals into a unified emotion assessment.
"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)


class MultimodalEmotionFusion:
    """
    Fuse multiple emotion signals into a unified emotion assessment.
    """

    def __init__(self):
        """Initialize the multimodal fusion system."""
        self.config = get_config()
        self.fusion_config = self.config.get_section("fusion")

        # Modality weights
        self.weights = {
            'face': self.fusion_config['weights']['face'],
            'voice': self.fusion_config['weights']['voice'],
            'text': self.fusion_config['weights']['text']
        }

        # Fusion strategy
        self.strategy = self.fusion_config.get('strategy', 'weighted_average')
        self.min_confidence = self.fusion_config.get('min_confidence', 0.4)

        # Emotion history for temporal fusion
        self.emotion_history: List[Dict] = []
        self.max_history = 20

        logger.info(
            f"Multimodal fusion initialized with strategy: {self.strategy}, "
            f"weights: {self.weights}"
        )

    def fuse_emotions(
        self,
        face_emotion: Optional[Dict] = None,
        voice_emotion: Optional[Dict] = None,
        text_emotion: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Fuse emotions from multiple modalities.

        Args:
            face_emotion: Facial emotion detection result
            voice_emotion: Voice emotion detection result
            text_emotion: Text sentiment analysis result

        Returns:
            Fused emotion result with:
                - dominant_emotion: Primary emotion
                - emotions: Emotion probability distribution
                - confidence: Overall confidence
                - modalities_used: Which modalities contributed
                - individual_results: Original modality results
                - timestamp: Fusion timestamp
        """
        # Collect available modalities
        modalities = {}
        if face_emotion and face_emotion.get('confidence', 0) > 0:
            modalities['face'] = face_emotion
        if voice_emotion and voice_emotion.get('confidence', 0) > 0:
            modalities['voice'] = voice_emotion
        if text_emotion and text_emotion.get('confidence', 0) > 0:
            modalities['text'] = text_emotion

        if not modalities:
            logger.debug("No valid emotion signals to fuse")
            return None

        # Apply fusion strategy
        if self.strategy == 'weighted_average':
            fused = self._weighted_average_fusion(modalities)
        elif self.strategy == 'max_confidence':
            fused = self._max_confidence_fusion(modalities)
        elif self.strategy == 'voting':
            fused = self._voting_fusion(modalities)
        else:
            logger.warning(f"Unknown fusion strategy: {self.strategy}, using weighted_average")
            fused = self._weighted_average_fusion(modalities)

        # Check minimum confidence
        if fused['confidence'] < self.min_confidence:
            logger.debug(
                f"Fused confidence {fused['confidence']:.2f} below threshold "
                f"{self.min_confidence}"
            )
            return None

        # Add metadata
        fused['modalities_used'] = list(modalities.keys())
        fused['individual_results'] = modalities
        fused['timestamp'] = datetime.now()

        # Add to history
        self._add_to_history(fused)

        logger.debug(
            f"Fused emotion: {fused['dominant_emotion']} "
            f"(confidence: {fused['confidence']:.2f}, "
            f"modalities: {fused['modalities_used']})"
        )

        return fused

    def _weighted_average_fusion(self, modalities: Dict) -> Dict:
        """
        Fuse emotions using weighted average of probabilities.

        Args:
            modalities: Dictionary of modality results

        Returns:
            Fused emotion result
        """
        # Collect all unique emotions
        all_emotions = set()
        for modality_data in modalities.values():
            all_emotions.update(modality_data.get('emotions', {}).keys())

        # Calculate weighted average for each emotion
        fused_emotions = {}
        total_weight = 0.0

        for emotion in all_emotions:
            weighted_sum = 0.0

            for modality, data in modalities.items():
                weight = self.weights.get(modality, 0.0)
                emotion_prob = data.get('emotions', {}).get(emotion, 0.0)
                confidence = data.get('confidence', 1.0)

                # Weight by both modality weight and confidence
                weighted_sum += weight * emotion_prob * confidence
                total_weight += weight * confidence

            if total_weight > 0:
                fused_emotions[emotion] = weighted_sum / total_weight

        # Normalize to sum to 1.0
        total_prob = sum(fused_emotions.values())
        if total_prob > 0:
            fused_emotions = {
                k: v / total_prob for k, v in fused_emotions.items()
            }

        # Determine dominant emotion
        if fused_emotions:
            dominant_emotion = max(fused_emotions, key=fused_emotions.get)
            confidence = fused_emotions[dominant_emotion]
        else:
            dominant_emotion = 'neutral'
            confidence = 0.5
            fused_emotions = {'neutral': 1.0}

        return {
            'dominant_emotion': dominant_emotion,
            'emotions': fused_emotions,
            'confidence': confidence
        }

    def _max_confidence_fusion(self, modalities: Dict) -> Dict:
        """
        Use the modality with highest confidence.

        Args:
            modalities: Dictionary of modality results

        Returns:
            Fused emotion result
        """
        # Find modality with highest confidence
        best_modality = max(
            modalities.items(),
            key=lambda x: x[1].get('confidence', 0.0)
        )

        modality_name, data = best_modality

        return {
            'dominant_emotion': data['dominant_emotion'],
            'emotions': data['emotions'].copy(),
            'confidence': data['confidence'],
            'primary_modality': modality_name
        }

    def _voting_fusion(self, modalities: Dict) -> Dict:
        """
        Fuse emotions using weighted voting.

        Args:
            modalities: Dictionary of modality results

        Returns:
            Fused emotion result
        """
        # Count votes for each emotion (weighted by confidence and modality weight)
        emotion_votes = defaultdict(float)

        for modality, data in modalities.items():
            emotion = data['dominant_emotion']
            confidence = data.get('confidence', 1.0)
            weight = self.weights.get(modality, 0.0)

            emotion_votes[emotion] += weight * confidence

        # Find emotion with most votes
        if emotion_votes:
            dominant_emotion = max(emotion_votes, key=emotion_votes.get)
            total_votes = sum(emotion_votes.values())
            confidence = emotion_votes[dominant_emotion] / total_votes if total_votes > 0 else 0.5

            # Build emotion distribution from votes
            emotions = {
                emotion: votes / total_votes
                for emotion, votes in emotion_votes.items()
            }
        else:
            dominant_emotion = 'neutral'
            confidence = 0.5
            emotions = {'neutral': 1.0}

        return {
            'dominant_emotion': dominant_emotion,
            'emotions': emotions,
            'confidence': confidence
        }

    def get_temporal_emotion(
        self,
        window_seconds: int = 10
    ) -> Optional[Dict]:
        """
        Get emotion averaged over a time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Temporally smoothed emotion
        """
        if not self.emotion_history:
            return None

        # Filter to recent history
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_emotions = [
            e for e in self.emotion_history
            if e['timestamp'] > cutoff_time
        ]

        if not recent_emotions:
            return self.emotion_history[-1] if self.emotion_history else None

        # Average emotions over time
        all_emotions = set()
        for emotion_data in recent_emotions:
            all_emotions.update(emotion_data['emotions'].keys())

        avg_emotions = {}
        for emotion in all_emotions:
            scores = [
                e['emotions'].get(emotion, 0.0)
                for e in recent_emotions
            ]
            avg_emotions[emotion] = sum(scores) / len(scores)

        # Determine dominant emotion
        dominant_emotion = max(avg_emotions, key=avg_emotions.get)
        confidence = avg_emotions[dominant_emotion]

        return {
            'dominant_emotion': dominant_emotion,
            'emotions': avg_emotions,
            'confidence': confidence,
            'sample_count': len(recent_emotions),
            'timestamp': datetime.now()
        }

    def get_emotion_stability(
        self,
        window_seconds: int = 30
    ) -> float:
        """
        Calculate emotion stability (how consistent emotions have been).

        Args:
            window_seconds: Time window to analyze

        Returns:
            Stability score (0.0 - 1.0, higher = more stable)
        """
        if len(self.emotion_history) < 2:
            return 1.0

        # Filter to recent history
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_emotions = [
            e['dominant_emotion']
            for e in self.emotion_history
            if e['timestamp'] > cutoff_time
        ]

        if len(recent_emotions) < 2:
            return 1.0

        # Calculate what percentage of time the same emotion was dominant
        from collections import Counter
        counts = Counter(recent_emotions)
        most_common_count = counts.most_common(1)[0][1]

        stability = most_common_count / len(recent_emotions)

        return stability

    def get_emotion_transition(self) -> Optional[Dict]:
        """
        Detect emotion transitions (changes in emotional state).

        Returns:
            Transition information if detected
        """
        if len(self.emotion_history) < 2:
            return None

        current = self.emotion_history[-1]
        previous = self.emotion_history[-2]

        if current['dominant_emotion'] != previous['dominant_emotion']:
            return {
                'from_emotion': previous['dominant_emotion'],
                'to_emotion': current['dominant_emotion'],
                'from_confidence': previous['confidence'],
                'to_confidence': current['confidence'],
                'timestamp': current['timestamp']
            }

        return None

    def _add_to_history(self, emotion_data: Dict):
        """
        Add emotion data to history.

        Args:
            emotion_data: Emotion data to add
        """
        self.emotion_history.append(emotion_data)

        # Trim history if needed
        if len(self.emotion_history) > self.max_history:
            self.emotion_history = self.emotion_history[-self.max_history:]

    def get_current_emotion(self) -> Optional[Dict]:
        """
        Get the most recent fused emotion.

        Returns:
            Current emotion data or None
        """
        if not self.emotion_history:
            return None

        return self.emotion_history[-1].copy()

    def get_emotion_summary(
        self,
        window_seconds: int = 60
    ) -> Dict:
        """
        Get a summary of emotions over a time period.

        Args:
            window_seconds: Time window to summarize

        Returns:
            Summary statistics
        """
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_emotions = [
            e for e in self.emotion_history
            if e['timestamp'] > cutoff_time
        ]

        if not recent_emotions:
            return {
                'count': 0,
                'emotions': {},
                'avg_confidence': 0.0,
                'stability': 1.0
            }

        # Count emotion occurrences
        from collections import Counter
        emotion_counts = Counter([e['dominant_emotion'] for e in recent_emotions])

        # Average confidence
        avg_confidence = sum(e['confidence'] for e in recent_emotions) / len(recent_emotions)

        # Stability
        stability = self.get_emotion_stability(window_seconds)

        return {
            'count': len(recent_emotions),
            'emotions': dict(emotion_counts),
            'avg_confidence': avg_confidence,
            'stability': stability,
            'window_seconds': window_seconds
        }

    def clear_history(self):
        """Clear emotion history."""
        self.emotion_history.clear()
        logger.info("Emotion history cleared")
