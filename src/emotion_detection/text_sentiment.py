"""
Text Sentiment Analysis Module.
Analyzes sentiment and emotional content from text messages.
"""

import re
from typing import Dict, Optional, List
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)


class TextSentimentAnalyzer:
    """
    Text sentiment and emotion analysis.
    """

    # Emotion keywords mapping
    EMOTION_KEYWORDS = {
        'happy': [
            'happy', 'joy', 'excited', 'thrilled', 'delighted', 'pleased',
            'glad', 'cheerful', 'wonderful', 'amazing', 'great', 'awesome',
            'fantastic', 'love', 'loving', 'ecstatic', 'elated'
        ],
        'sad': [
            'sad', 'unhappy', 'depressed', 'down', 'miserable', 'gloomy',
            'disappointed', 'heartbroken', 'upset', 'crying', 'tears',
            'sorrowful', 'grief', 'mourning', 'melancholy'
        ],
        'angry': [
            'angry', 'mad', 'furious', 'rage', 'annoyed', 'irritated',
            'frustrated', 'outraged', 'hostile', 'resentful', 'bitter',
            'infuriated', 'enraged', 'livid'
        ],
        'fearful': [
            'afraid', 'scared', 'fear', 'terrified', 'anxious', 'worried',
            'nervous', 'frightened', 'panic', 'dread', 'alarmed',
            'horrified', 'petrified', 'uneasy'
        ],
        'surprised': [
            'surprised', 'shocked', 'amazed', 'astonished', 'astounded',
            'stunned', 'startled', 'unexpected', 'wow', 'incredible',
            'unbelievable'
        ],
        'disgusted': [
            'disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated',
            'appalled', 'horrified', 'gross', 'yuck', 'ew'
        ]
    }

    def __init__(self):
        """Initialize the text sentiment analyzer."""
        self.config = get_config()
        self.text_config = self.config.get_section("emotion_detection")["text"]

        self.enabled = self.text_config["enabled"]
        self.model_type = self.text_config.get("model", "vader")
        self.confidence_threshold = self.text_config["confidence_threshold"]

        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()

        # Download required NLTK data
        self._download_nltk_data()

        logger.info(f"Text sentiment analyzer initialized with model: {self.model_type}")

    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('brown', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")

    def analyze_sentiment(self, text: str) -> Optional[Dict]:
        """
        Analyze sentiment and emotion from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing:
                - sentiment: Sentiment classification (positive/negative/neutral)
                - sentiment_scores: Detailed sentiment scores
                - dominant_emotion: Primary detected emotion
                - emotions: Dictionary of emotion scores
                - confidence: Overall confidence
                - timestamp: Analysis timestamp
        """
        if not self.enabled or not text or not text.strip():
            return None

        try:
            # Clean text
            cleaned_text = self._clean_text(text)

            # Get VADER sentiment scores
            vader_scores = self.vader.polarity_scores(cleaned_text)

            # Determine overall sentiment
            compound = vader_scores['compound']
            if compound >= 0.05:
                sentiment = 'positive'
            elif compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'

            # Get TextBlob sentiment for comparison
            blob = TextBlob(cleaned_text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity

            # Detect emotions from keywords
            emotion_scores = self._detect_emotions_from_keywords(cleaned_text)

            # Determine dominant emotion
            if emotion_scores:
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[dominant_emotion]
            else:
                # Map sentiment to emotion if no specific emotion detected
                if sentiment == 'positive':
                    dominant_emotion = 'happy'
                    confidence = abs(compound)
                elif sentiment == 'negative':
                    dominant_emotion = 'sad'
                    confidence = abs(compound)
                else:
                    dominant_emotion = 'neutral'
                    confidence = 1.0 - abs(compound)

                emotion_scores = {dominant_emotion: confidence}

            # Check confidence threshold
            if confidence < self.confidence_threshold:
                logger.debug(
                    f"Text sentiment confidence {confidence:.2f} below threshold "
                    f"{self.confidence_threshold}"
                )
                return None

            # Build result
            result = {
                'sentiment': sentiment,
                'sentiment_scores': {
                    'positive': vader_scores['pos'],
                    'negative': vader_scores['neg'],
                    'neutral': vader_scores['neu'],
                    'compound': vader_scores['compound']
                },
                'dominant_emotion': dominant_emotion,
                'emotions': emotion_scores,
                'confidence': confidence,
                'text_features': {
                    'polarity': textblob_polarity,
                    'subjectivity': textblob_subjectivity,
                    'word_count': len(cleaned_text.split()),
                    'char_count': len(cleaned_text)
                },
                'timestamp': datetime.now()
            }

            logger.debug(
                f"Text sentiment: {sentiment}, emotion: {dominant_emotion} "
                f"(confidence: {confidence:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove mentions and hashtags (keep the text)
        text = re.sub(r'[@#]\w+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _detect_emotions_from_keywords(self, text: str) -> Dict[str, float]:
        """
        Detect emotions based on keyword matching.

        Args:
            text: Input text

        Returns:
            Dictionary of emotion scores
        """
        text_lower = text.lower()
        emotion_counts = {}

        # Count keyword matches for each emotion
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                emotion_counts[emotion] = count

        # Normalize to probabilities
        if emotion_counts:
            total = sum(emotion_counts.values())
            emotion_scores = {
                emotion: count / total
                for emotion, count in emotion_counts.items()
            }
            return emotion_scores

        return {}

    def analyze_conversation_context(
        self,
        messages: List[str],
        max_messages: int = 5
    ) -> Optional[Dict]:
        """
        Analyze sentiment across multiple messages for context.

        Args:
            messages: List of message texts
            max_messages: Maximum number of recent messages to analyze

        Returns:
            Aggregated sentiment analysis
        """
        if not messages:
            return None

        # Analyze recent messages
        recent_messages = messages[-max_messages:]
        analyses = [
            self.analyze_sentiment(msg)
            for msg in recent_messages
            if msg.strip()
        ]

        # Filter out None results
        analyses = [a for a in analyses if a is not None]

        if not analyses:
            return None

        # Aggregate sentiments
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for analysis in analyses:
            sentiment_counts[analysis['sentiment']] += 1

        # Average sentiment scores
        avg_compound = sum(
            a['sentiment_scores']['compound'] for a in analyses
        ) / len(analyses)

        # Aggregate emotions
        all_emotions = {}
        for analysis in analyses:
            for emotion, score in analysis['emotions'].items():
                if emotion not in all_emotions:
                    all_emotions[emotion] = []
                all_emotions[emotion].append(score)

        # Average emotion scores
        avg_emotions = {
            emotion: sum(scores) / len(scores)
            for emotion, scores in all_emotions.items()
        }

        # Determine dominant emotion
        if avg_emotions:
            dominant_emotion = max(avg_emotions, key=avg_emotions.get)
            confidence = avg_emotions[dominant_emotion]
        else:
            dominant_emotion = 'neutral'
            confidence = 0.5

        # Determine overall sentiment
        if avg_compound >= 0.05:
            sentiment = 'positive'
        elif avg_compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'sentiment': sentiment,
            'sentiment_counts': sentiment_counts,
            'avg_compound': avg_compound,
            'dominant_emotion': dominant_emotion,
            'emotions': avg_emotions,
            'confidence': confidence,
            'message_count': len(analyses),
            'timestamp': datetime.now()
        }

    def get_emotion_intensity(self, text: str, emotion: str) -> float:
        """
        Get the intensity of a specific emotion in the text.

        Args:
            text: Input text
            emotion: Emotion to check

        Returns:
            Intensity score (0.0 - 1.0)
        """
        analysis = self.analyze_sentiment(text)

        if analysis is None:
            return 0.0

        return analysis['emotions'].get(emotion, 0.0)

    def is_emotionally_charged(
        self,
        text: str,
        threshold: float = 0.5
    ) -> bool:
        """
        Check if text is emotionally charged.

        Args:
            text: Input text
            threshold: Emotion intensity threshold

        Returns:
            True if emotionally charged
        """
        analysis = self.analyze_sentiment(text)

        if analysis is None:
            return False

        return analysis['confidence'] >= threshold


class ConversationEmotionTracker:
    """
    Track emotional progression throughout a conversation.
    """

    def __init__(self, max_history: int = 50):
        """
        Initialize the conversation tracker.

        Args:
            max_history: Maximum conversation history to maintain
        """
        self.analyzer = TextSentimentAnalyzer()
        self.max_history = max_history

        # Conversation history
        self.messages: List[str] = []
        self.analyses: List[Dict] = []

        logger.info("Conversation emotion tracker initialized")

    def add_message(self, text: str) -> Optional[Dict]:
        """
        Add a message and analyze its sentiment.

        Args:
            text: Message text

        Returns:
            Sentiment analysis result
        """
        # Analyze the message
        analysis = self.analyzer.analyze_sentiment(text)

        # Add to history
        self.messages.append(text)
        if analysis:
            self.analyses.append(analysis)

        # Trim history if needed
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
            self.analyses = self.analyses[-self.max_history:]

        return analysis

    def get_conversation_context(self) -> Optional[Dict]:
        """
        Get aggregated sentiment analysis for the conversation.

        Returns:
            Conversation-level sentiment analysis
        """
        return self.analyzer.analyze_conversation_context(self.messages)

    def get_emotion_trend(self, emotion: str) -> List[float]:
        """
        Get the trend of a specific emotion over the conversation.

        Args:
            emotion: Emotion to track

        Returns:
            List of emotion intensities over time
        """
        return [
            analysis['emotions'].get(emotion, 0.0)
            for analysis in self.analyses
        ]

    def get_sentiment_trend(self) -> List[str]:
        """
        Get the sentiment trend over the conversation.

        Returns:
            List of sentiments over time
        """
        return [analysis['sentiment'] for analysis in self.analyses]

    def clear_history(self):
        """Clear conversation history."""
        self.messages.clear()
        self.analyses.clear()
