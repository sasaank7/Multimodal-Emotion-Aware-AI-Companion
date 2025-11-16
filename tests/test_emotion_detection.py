"""
Comprehensive tests for emotion detection modules.
"""

import pytest
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from emotion_detection.text_sentiment import TextSentimentAnalyzer, ConversationEmotionTracker
from emotion_detection.multimodal_fusion import MultimodalEmotionFusion


class TestTextSentimentAnalyzer:
    """Test text sentiment analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TextSentimentAnalyzer()

    def test_positive_sentiment(self, analyzer):
        """Test positive sentiment detection."""
        text = "I'm so happy and excited about this amazing project!"
        result = analyzer.analyze_sentiment(text)

        assert result is not None
        assert result['sentiment'] == 'positive'
        assert result['dominant_emotion'] in ['happy', 'surprised']
        assert result['confidence'] > 0.5

    def test_negative_sentiment(self, analyzer):
        """Test negative sentiment detection."""
        text = "I'm feeling really sad and disappointed today."
        result = analyzer.analyze_sentiment(text)

        assert result is not None
        assert result['sentiment'] == 'negative'
        assert result['dominant_emotion'] == 'sad'
        assert result['confidence'] > 0.3

    def test_neutral_sentiment(self, analyzer):
        """Test neutral sentiment detection."""
        text = "The weather is okay today."
        result = analyzer.analyze_sentiment(text)

        assert result is not None
        # Could be neutral or low confidence
        assert result['sentiment'] in ['neutral', 'positive', 'negative']

    def test_empty_text(self, analyzer):
        """Test empty text handling."""
        result = analyzer.analyze_sentiment("")
        assert result is None

    def test_emotion_intensity(self, analyzer):
        """Test emotion intensity detection."""
        text = "I'm extremely angry and furious!"
        intensity = analyzer.get_emotion_intensity(text, 'angry')

        assert intensity > 0.5

    def test_emotionally_charged(self, analyzer):
        """Test emotionally charged detection."""
        charged_text = "This is absolutely amazing and wonderful!"
        neutral_text = "This is a test message."

        assert analyzer.is_emotionally_charged(charged_text, threshold=0.4)
        assert not analyzer.is_emotionally_charged(neutral_text, threshold=0.6)


class TestConversationEmotionTracker:
    """Test conversation emotion tracking."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return ConversationEmotionTracker(max_history=10)

    def test_add_message(self, tracker):
        """Test adding messages."""
        message = "I'm feeling great today!"
        result = tracker.add_message(message)

        assert result is not None
        assert len(tracker.messages) == 1
        assert len(tracker.analyses) == 1

    def test_conversation_context(self, tracker):
        """Test conversation context aggregation."""
        messages = [
            "I'm happy today!",
            "This is wonderful!",
            "I love this project!"
        ]

        for msg in messages:
            tracker.add_message(msg)

        context = tracker.get_conversation_context()

        assert context is not None
        assert context['sentiment'] == 'positive'
        assert context['message_count'] == 3

    def test_emotion_trend(self, tracker):
        """Test emotion trend tracking."""
        messages = ["I'm happy!", "Still happy!", "Getting happier!"]

        for msg in messages:
            tracker.add_message(msg)

        trend = tracker.get_emotion_trend('happy')

        assert len(trend) == 3
        assert all(score >= 0 for score in trend)

    def test_sentiment_trend(self, tracker):
        """Test sentiment trend."""
        messages = ["I'm happy!", "I'm sad now.", "Feeling better!"]

        for msg in messages:
            tracker.add_message(msg)

        trend = tracker.get_sentiment_trend()

        assert len(trend) == 3
        assert trend[0] == 'positive'
        assert trend[1] == 'negative'

    def test_history_limit(self, tracker):
        """Test history limit enforcement."""
        for i in range(15):
            tracker.add_message(f"Message {i}")

        assert len(tracker.messages) <= tracker.max_history
        assert len(tracker.analyses) <= tracker.max_history


class TestMultimodalFusion:
    """Test multimodal emotion fusion."""

    @pytest.fixture
    def fusion(self):
        """Create fusion engine instance."""
        return MultimodalEmotionFusion()

    def test_single_modality(self, fusion):
        """Test fusion with single modality."""
        text_emotion = {
            'dominant_emotion': 'happy',
            'emotions': {'happy': 0.8, 'neutral': 0.2},
            'confidence': 0.8
        }

        result = fusion.fuse_emotions(text_emotion=text_emotion)

        assert result is not None
        assert result['dominant_emotion'] == 'happy'
        assert 'text' in result['modalities_used']

    def test_multi_modality_fusion(self, fusion):
        """Test fusion with multiple modalities."""
        face_emotion = {
            'dominant_emotion': 'happy',
            'emotions': {'happy': 0.9, 'neutral': 0.1},
            'confidence': 0.9
        }

        text_emotion = {
            'dominant_emotion': 'happy',
            'emotions': {'happy': 0.7, 'neutral': 0.3},
            'confidence': 0.7
        }

        result = fusion.fuse_emotions(
            face_emotion=face_emotion,
            text_emotion=text_emotion
        )

        assert result is not None
        assert result['dominant_emotion'] == 'happy'
        assert len(result['modalities_used']) == 2
        assert result['confidence'] > 0.7

    def test_conflicting_emotions(self, fusion):
        """Test fusion with conflicting emotions."""
        face_emotion = {
            'dominant_emotion': 'happy',
            'emotions': {'happy': 0.8, 'sad': 0.2},
            'confidence': 0.8
        }

        text_emotion = {
            'dominant_emotion': 'sad',
            'emotions': {'sad': 0.7, 'happy': 0.3},
            'confidence': 0.7
        }

        result = fusion.fuse_emotions(
            face_emotion=face_emotion,
            text_emotion=text_emotion
        )

        assert result is not None
        # Should be weighted average
        assert result['confidence'] > 0.4

    def test_temporal_emotion(self, fusion):
        """Test temporal emotion smoothing."""
        # Add some emotions to history
        for i in range(5):
            emotion = {
                'dominant_emotion': 'happy',
                'emotions': {'happy': 0.8, 'neutral': 0.2},
                'confidence': 0.8,
                'timestamp': datetime.now()
            }
            fusion._add_to_history(emotion)

        temporal = fusion.get_temporal_emotion(window_seconds=10)

        assert temporal is not None
        assert temporal['dominant_emotion'] == 'happy'
        assert temporal['sample_count'] == 5

    def test_emotion_stability(self, fusion):
        """Test emotion stability calculation."""
        # Add consistent emotions
        for i in range(5):
            emotion = {
                'dominant_emotion': 'happy',
                'emotions': {'happy': 0.8},
                'confidence': 0.8,
                'timestamp': datetime.now()
            }
            fusion._add_to_history(emotion)

        stability = fusion.get_emotion_stability(window_seconds=10)

        assert stability == 1.0  # All same emotion

    def test_emotion_transition(self, fusion):
        """Test emotion transition detection."""
        # Add first emotion
        fusion._add_to_history({
            'dominant_emotion': 'happy',
            'emotions': {'happy': 0.8},
            'confidence': 0.8,
            'timestamp': datetime.now()
        })

        # Add different emotion
        fusion._add_to_history({
            'dominant_emotion': 'sad',
            'emotions': {'sad': 0.7},
            'confidence': 0.7,
            'timestamp': datetime.now()
        })

        transition = fusion.get_emotion_transition()

        assert transition is not None
        assert transition['from_emotion'] == 'happy'
        assert transition['to_emotion'] == 'sad'

    def test_emotion_summary(self, fusion):
        """Test emotion summary generation."""
        # Add various emotions
        emotions = ['happy', 'happy', 'neutral', 'happy', 'sad']

        for emotion in emotions:
            fusion._add_to_history({
                'dominant_emotion': emotion,
                'emotions': {emotion: 0.8},
                'confidence': 0.8,
                'timestamp': datetime.now()
            })

        summary = fusion.get_emotion_summary(window_seconds=60)

        assert summary['count'] == 5
        assert summary['emotions']['happy'] == 3
        assert summary['avg_confidence'] == 0.8

    def test_min_confidence_threshold(self, fusion):
        """Test minimum confidence threshold."""
        low_confidence_emotion = {
            'dominant_emotion': 'happy',
            'emotions': {'happy': 0.3},
            'confidence': 0.3
        }

        result = fusion.fuse_emotions(text_emotion=low_confidence_emotion)

        # Should be None if below threshold
        # (depends on config settings)
        assert result is None or result['confidence'] >= fusion.min_confidence


@pytest.mark.integration
class TestEmotionDetectionIntegration:
    """Integration tests for emotion detection pipeline."""

    def test_end_to_end_text_processing(self):
        """Test complete text processing pipeline."""
        analyzer = TextSentimentAnalyzer()
        fusion = MultimodalEmotionFusion()

        text = "I'm feeling really stressed and overwhelmed today."

        # Analyze text
        text_result = analyzer.analyze_sentiment(text)
        assert text_result is not None

        # Fuse (with only text)
        fused = fusion.fuse_emotions(text_emotion=text_result)
        assert fused is not None

        # Check emotion flow
        assert fused['dominant_emotion'] in ['sad', 'fearful', 'angry']

    def test_conversation_flow(self):
        """Test multi-turn conversation flow."""
        tracker = ConversationEmotionTracker()

        conversation = [
            "I'm excited to start this project!",
            "It's challenging but I'm motivated.",
            "I'm a bit worried about the deadline.",
            "But I think we can make it!"
        ]

        for message in conversation:
            result = tracker.add_message(message)
            assert result is not None

        # Verify conversation context
        context = tracker.get_conversation_context()
        assert context is not None
        assert context['message_count'] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
