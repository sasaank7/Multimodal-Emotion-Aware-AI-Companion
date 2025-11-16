"""
Demo script for testing emotion detection components.
Run this to test individual modules without the full UI.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from emotion_detection.text_sentiment import TextSentimentAnalyzer
from emotion_detection.multimodal_fusion import MultimodalEmotionFusion
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


def demo_text_sentiment():
    """Demo text sentiment analysis."""
    print("\n" + "="*60)
    print("TEXT SENTIMENT ANALYSIS DEMO")
    print("="*60)

    analyzer = TextSentimentAnalyzer()

    test_messages = [
        "I'm feeling really happy today!",
        "This is so frustrating and annoying.",
        "I'm worried about the upcoming presentation.",
        "Everything is going great, I love it!",
        "I feel sad and disappointed.",
        "This is a normal message with no strong emotion."
    ]

    for message in test_messages:
        print(f"\nMessage: \"{message}\"")

        result = analyzer.analyze_sentiment(message)

        if result:
            print(f"  Sentiment: {result['sentiment']}")
            print(f"  Dominant Emotion: {result['dominant_emotion']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Scores: {result['sentiment_scores']}")
        else:
            print("  No strong emotion detected")


def demo_multimodal_fusion():
    """Demo multimodal fusion."""
    print("\n" + "="*60)
    print("MULTIMODAL FUSION DEMO")
    print("="*60)

    fusion = MultimodalEmotionFusion()

    # Simulate emotion data from different modalities
    face_emotion = {
        'dominant_emotion': 'happy',
        'emotions': {'happy': 0.8, 'neutral': 0.15, 'surprised': 0.05},
        'confidence': 0.8
    }

    text_emotion = {
        'dominant_emotion': 'sad',
        'emotions': {'sad': 0.6, 'neutral': 0.3, 'happy': 0.1},
        'confidence': 0.6
    }

    print("\nScenario 1: Conflicting emotions (face=happy, text=sad)")
    print(f"  Face: {face_emotion['dominant_emotion']} ({face_emotion['confidence']:.1%})")
    print(f"  Text: {text_emotion['dominant_emotion']} ({text_emotion['confidence']:.1%})")

    fused = fusion.fuse_emotions(
        face_emotion=face_emotion,
        text_emotion=text_emotion
    )

    if fused:
        print(f"\n  Fused Emotion: {fused['dominant_emotion']}")
        print(f"  Confidence: {fused['confidence']:.2%}")
        print(f"  Modalities Used: {fused['modalities_used']}")
        print(f"  Emotion Distribution: {fused['emotions']}")


def demo_conversation():
    """Demo a simple conversation."""
    print("\n" + "="*60)
    print("CONVERSATION DEMO")
    print("="*60)

    analyzer = TextSentimentAnalyzer()

    messages = [
        "I'm really excited about this new project!",
        "But I'm also a bit nervous about the deadline.",
        "I think we can make it if we work hard.",
        "I'm feeling more confident now."
    ]

    print("\nAnalyzing a conversation sequence:\n")

    for i, message in enumerate(messages, 1):
        print(f"Turn {i}: \"{message}\"")

        result = analyzer.analyze_sentiment(message)

        if result:
            print(f"  → Emotion: {result['dominant_emotion']} ({result['confidence']:.1%})")
        print()

    # Analyze conversation context
    print("\nOverall Conversation Analysis:")
    context = analyzer.analyze_conversation_context(messages)

    if context:
        print(f"  Overall Sentiment: {context['sentiment']}")
        print(f"  Dominant Emotion: {context['dominant_emotion']}")
        print(f"  Confidence: {context['confidence']:.2%}")
        print(f"  Sentiment Counts: {context['sentiment_counts']}")


def demo_personalization():
    """Demo personalization features."""
    print("\n" + "="*60)
    print("PERSONALIZATION DEMO")
    print("="*60)

    from utils.personalization import UserProfile

    profile = UserProfile(user_id="demo_user")

    # Simulate some interactions
    print("\nSimulating user interactions...\n")

    interactions = [
        ("I'm stressed about work", "supportive"),
        ("Tell me something cheerful", "cheerful"),
        ("I need help with a problem", "neutral"),
    ]

    for msg, mode in interactions:
        profile.add_interaction(
            user_message=msg,
            assistant_response=f"[Response in {mode} mode]",
            emotion_context={'dominant_emotion': 'neutral', 'confidence': 0.5},
            response_mode=mode
        )
        print(f"Added: \"{msg}\" ({mode} mode)")

    # Get stats
    print("\nInteraction Statistics:")
    stats = profile.get_interaction_stats(days=7)
    print(f"  Total Interactions: {stats['total_interactions']}")
    print(f"  Response Modes: {stats['response_modes']}")

    # Clean up demo data
    profile.clear_history(keep_profile=False)
    profile.close()

    print("\n(Demo data cleared)")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("EMOTION-AWARE AI COMPANION - COMPONENT DEMOS")
    print("="*60)

    try:
        demo_text_sentiment()
        demo_multimodal_fusion()
        demo_conversation()
        demo_personalization()

        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nTo run the full application, use: python app.py")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nError running demos: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
