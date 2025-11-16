"""
Intelligent Recommendation Engine.
Provides personalized recommendations based on emotional state and user patterns.
"""

from typing import Dict, List, Optional
from datetime import datetime
import random

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)


class RecommendationEngine:
    """
    Generate context-aware recommendations based on emotions and user data.
    """

    # Emotion-based recommendations database
    RECOMMENDATIONS_DB = {
        'sad': {
            'activities': [
                "Take a short walk outside",
                "Listen to uplifting music",
                "Call a friend or family member",
                "Watch a favorite comedy show",
                "Practice gratitude journaling",
                "Do some light exercise or stretching",
                "Take a warm bath or shower",
                "Cook or eat a comfort food"
            ],
            'music_genres': ['uplifting', 'energetic', 'inspiring', 'happy'],
            'exercises': [
                "5-minute guided meditation",
                "Deep breathing exercise",
                "Progressive muscle relaxation",
                "Gentle yoga flow"
            ],
            'content': [
                "Inspiring TED talks",
                "Motivational podcasts",
                "Uplifting documentaries",
                "Funny videos or memes"
            ]
        },
        'stressed': {
            'activities': [
                "Take a 10-minute break",
                "Practice mindfulness meditation",
                "Go for a quick walk",
                "Do some desk stretches",
                "Listen to calming music",
                "Organize your workspace",
                "Make a to-do list",
                "Talk to someone about it"
            ],
            'music_genres': ['calm', 'ambient', 'classical', 'nature sounds'],
            'exercises': [
                "Box breathing (4-4-4-4)",
                "Body scan meditation",
                "Guided stress relief meditation",
                "Gentle stretching routine"
            ],
            'content': [
                "Stress management tips",
                "Time management techniques",
                "Productivity strategies",
                "Relaxation guides"
            ]
        },
        'angry': {
            'activities': [
                "Take deep breaths",
                "Go for a vigorous walk or run",
                "Do some physical exercise",
                "Write down your feelings",
                "Count to 10 slowly",
                "Step away from the situation",
                "Listen to calming music",
                "Talk it out with someone"
            ],
            'music_genres': ['calm', 'instrumental', 'meditation', 'nature sounds'],
            'exercises': [
                "Progressive muscle relaxation",
                "Anger management breathing",
                "Mindful walking",
                "Intense cardio workout"
            ],
            'content': [
                "Anger management techniques",
                "Conflict resolution strategies",
                "Emotional regulation guides",
                "Mindfulness resources"
            ]
        },
        'anxious': {
            'activities': [
                "Practice grounding techniques",
                "Do a breathing exercise",
                "Write down your worries",
                "Talk to someone supportive",
                "Engage in a calming hobby",
                "Listen to soothing music",
                "Take a warm shower",
                "Practice gentle movement"
            ],
            'music_genres': ['ambient', 'calm', 'classical', 'meditation'],
            'exercises': [
                "5-4-3-2-1 grounding technique",
                "Deep breathing exercises",
                "Guided anxiety meditation",
                "Gentle yoga for anxiety"
            ],
            'content': [
                "Anxiety coping strategies",
                "Grounding technique guides",
                "Mindfulness exercises",
                "Relaxation tutorials"
            ]
        },
        'tired': {
            'activities': [
                "Take a power nap (20 min)",
                "Get some fresh air",
                "Have a healthy snack",
                "Do some light stretching",
                "Drink water or green tea",
                "Take a short break",
                "Go for a brief walk",
                "Listen to energizing music"
            ],
            'music_genres': ['energetic', 'upbeat', 'motivational', 'pop'],
            'exercises': [
                "Quick energizing yoga",
                "Desk exercises",
                "Power poses",
                "Light cardio burst"
            ],
            'content': [
                "Energy-boosting tips",
                "Sleep hygiene guides",
                "Productivity hacks",
                "Healthy habits for energy"
            ]
        },
        'happy': {
            'activities': [
                "Share your happiness with others",
                "Document this positive moment",
                "Engage in creative activities",
                "Help someone else",
                "Try something new",
                "Celebrate your achievements",
                "Spend time with loved ones",
                "Exercise or play sports"
            ],
            'music_genres': ['upbeat', 'pop', 'dance', 'energetic'],
            'exercises': [
                "Dynamic yoga flow",
                "Dance workout",
                "High-energy cardio",
                "Fun sports activity"
            ],
            'content': [
                "Inspiring success stories",
                "Creative project ideas",
                "Fun challenges to try",
                "Positive psychology content"
            ]
        },
        'neutral': {
            'activities': [
                "Learn something new",
                "Work on a personal project",
                "Read an interesting article",
                "Practice a hobby",
                "Plan your goals",
                "Connect with friends",
                "Explore new interests",
                "Organize your space"
            ],
            'music_genres': ['varied', 'instrumental', 'lo-fi', 'jazz'],
            'exercises': [
                "Balanced yoga practice",
                "Moderate cardio",
                "Strength training",
                "Flexibility routine"
            ],
            'content': [
                "Educational content",
                "Skill-building tutorials",
                "Personal development",
                "Interesting documentaries"
            ]
        }
    }

    # Time-of-day specific recommendations
    TIME_BASED_RECOMMENDATIONS = {
        'morning': {
            'activities': [
                "Morning meditation or stretching",
                "Healthy breakfast",
                "Plan your day",
                "Morning walk or jog",
                "Read or journal"
            ],
            'energy': 'Start your day with positive energy'
        },
        'afternoon': {
            'activities': [
                "Take a lunch break",
                "Quick power nap",
                "Afternoon walk",
                "Healthy snack",
                "Desk stretches"
            ],
            'energy': 'Maintain your momentum'
        },
        'evening': {
            'activities': [
                "Wind down routine",
                "Reflect on the day",
                "Light dinner",
                "Relaxing hobby",
                "Quality time with family"
            ],
            'energy': 'Prepare for restful evening'
        },
        'night': {
            'activities': [
                "Relaxation techniques",
                "Light reading",
                "Prepare for sleep",
                "Gratitude practice",
                "Calm music"
            ],
            'energy': 'Get ready for restful sleep'
        }
    }

    def __init__(self):
        """Initialize recommendation engine."""
        self.config = get_config()
        logger.info("Recommendation engine initialized")

    def get_recommendations(
        self,
        emotion_context: Dict,
        user_profile: Optional[Dict] = None,
        limit: int = 3
    ) -> Dict:
        """
        Generate personalized recommendations.

        Args:
            emotion_context: Current emotion data
            user_profile: User profile and preferences
            limit: Maximum recommendations per category

        Returns:
            Dictionary of recommendations by category
        """
        if not emotion_context:
            return self._get_general_recommendations(limit)

        dominant_emotion = emotion_context.get('dominant_emotion', 'neutral')
        confidence = emotion_context.get('confidence', 0.0)

        # Map emotions to recommendation categories
        emotion_map = {
            'sad': 'sad',
            'angry': 'angry',
            'fearful': 'anxious',
            'happy': 'happy',
            'surprised': 'happy',
            'neutral': 'neutral',
            'disgusted': 'stressed'
        }

        rec_category = emotion_map.get(dominant_emotion, 'neutral')

        # Get base recommendations
        base_recs = self.RECOMMENDATIONS_DB.get(rec_category, {})

        # Get time-based recommendations
        time_recs = self._get_time_based_recommendations()

        # Personalize based on user profile
        if user_profile:
            base_recs = self._personalize_recommendations(base_recs, user_profile)

        # Build final recommendations
        recommendations = {
            'emotion': dominant_emotion,
            'confidence': confidence,
            'recommended_for': rec_category,
            'activities': self._sample_list(base_recs.get('activities', []), limit),
            'music': base_recs.get('music_genres', []),
            'exercises': self._sample_list(base_recs.get('exercises', []), limit),
            'content': self._sample_list(base_recs.get('content', []), limit),
            'time_based': time_recs,
            'quick_actions': self._get_quick_actions(rec_category),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Generated recommendations for emotion: {dominant_emotion}")

        return recommendations

    def get_activity_recommendation(
        self,
        emotion: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Get a single activity recommendation.

        Args:
            emotion: Current emotion
            context: Additional context

        Returns:
            Activity recommendation text
        """
        emotion_map = {
            'sad': 'sad',
            'stressed': 'stressed',
            'angry': 'angry',
            'anxious': 'anxious',
            'tired': 'tired',
            'happy': 'happy',
            'neutral': 'neutral'
        }

        category = emotion_map.get(emotion, 'neutral')
        activities = self.RECOMMENDATIONS_DB.get(category, {}).get('activities', [])

        if not activities:
            return "Take a moment to check in with yourself."

        return random.choice(activities)

    def get_breathing_exercise(self, emotion: str) -> Dict:
        """
        Get a breathing exercise for the emotion.

        Args:
            emotion: Current emotion

        Returns:
            Breathing exercise instructions
        """
        exercises = {
            'stressed': {
                'name': 'Box Breathing',
                'description': 'Calm your nervous system with 4-4-4-4 breathing',
                'steps': [
                    'Inhale slowly for 4 counts',
                    'Hold your breath for 4 counts',
                    'Exhale slowly for 4 counts',
                    'Hold empty for 4 counts',
                    'Repeat 4-5 times'
                ],
                'duration': '2 minutes'
            },
            'anxious': {
                'name': '4-7-8 Breathing',
                'description': 'Reduce anxiety with extended exhale',
                'steps': [
                    'Inhale through nose for 4 counts',
                    'Hold breath for 7 counts',
                    'Exhale through mouth for 8 counts',
                    'Repeat 4 times'
                ],
                'duration': '2 minutes'
            },
            'angry': {
                'name': 'Counted Breathing',
                'description': 'Regain control with slow, counted breaths',
                'steps': [
                    'Count slowly to 10',
                    'With each count, take a deep breath',
                    'Exhale slowly and completely',
                    'Repeat until calm'
                ],
                'duration': '3 minutes'
            },
            'default': {
                'name': 'Deep Breathing',
                'description': 'Simple deep breathing for general relaxation',
                'steps': [
                    'Sit or lie comfortably',
                    'Breathe in deeply through nose',
                    'Exhale slowly through mouth',
                    'Focus on your breath',
                    'Repeat 5-10 times'
                ],
                'duration': '2 minutes'
            }
        }

        return exercises.get(emotion, exercises['default'])

    def get_music_recommendation(
        self,
        emotion: str,
        user_preferences: Optional[Dict] = None
    ) -> Dict:
        """
        Get music recommendations.

        Args:
            emotion: Current emotion
            user_preferences: User music preferences

        Returns:
            Music recommendation
        """
        emotion_map = {
            'sad': 'sad',
            'stressed': 'stressed',
            'angry': 'angry',
            'anxious': 'anxious',
            'tired': 'tired',
            'happy': 'happy'
        }

        category = emotion_map.get(emotion, 'neutral')
        genres = self.RECOMMENDATIONS_DB.get(category, {}).get('music_genres', ['calm'])

        return {
            'emotion': emotion,
            'genres': genres,
            'suggested_playlist': f"{emotion.title()} Mood Booster",
            'purpose': self._get_music_purpose(emotion)
        }

    def _get_music_purpose(self, emotion: str) -> str:
        """Get purpose description for music recommendation."""
        purposes = {
            'sad': 'Uplift your mood',
            'stressed': 'Calm your mind',
            'angry': 'Soothe your emotions',
            'anxious': 'Reduce anxiety',
            'tired': 'Boost your energy',
            'happy': 'Enhance your joy'
        }
        return purposes.get(emotion, 'Support your current state')

    def _get_time_based_recommendations(self) -> Dict:
        """Get recommendations based on current time."""
        current_hour = datetime.now().hour

        if 5 <= current_hour < 12:
            period = 'morning'
        elif 12 <= current_hour < 17:
            period = 'afternoon'
        elif 17 <= current_hour < 22:
            period = 'evening'
        else:
            period = 'night'

        return self.TIME_BASED_RECOMMENDATIONS.get(period, {})

    def _get_quick_actions(self, emotion_category: str) -> List[str]:
        """Get quick action buttons."""
        quick_actions = {
            'sad': ['Play uplifting music', 'Start breathing exercise', 'Call a friend'],
            'stressed': ['Take a break', 'Start meditation', 'Go for a walk'],
            'angry': ['Cool down exercise', 'Step away', 'Deep breathing'],
            'anxious': ['Grounding technique', 'Calm breathing', 'Talk to someone'],
            'tired': ['Power nap timer', 'Energy boost', 'Quick stretch'],
            'happy': ['Share your joy', 'Celebrate', 'Be creative']
        }

        return quick_actions.get(emotion_category, ['Take a moment', 'Reflect', 'Self-care'])

    def _personalize_recommendations(
        self,
        base_recs: Dict,
        user_profile: Dict
    ) -> Dict:
        """
        Personalize recommendations based on user profile.

        Args:
            base_recs: Base recommendations
            user_profile: User profile data

        Returns:
            Personalized recommendations
        """
        # Filter out activities user doesn't like
        disliked = user_profile.get('disliked_activities', [])

        if 'activities' in base_recs:
            base_recs['activities'] = [
                a for a in base_recs['activities']
                if not any(d in a.lower() for d in disliked)
            ]

        # Add user's preferred activities
        preferred = user_profile.get('preferred_activities', [])
        if preferred and 'activities' in base_recs:
            base_recs['activities'] = preferred + base_recs['activities']

        return base_recs

    def _sample_list(self, items: List, limit: int) -> List:
        """Sample items from list."""
        if len(items) <= limit:
            return items

        return random.sample(items, limit)

    def _get_general_recommendations(self, limit: int) -> Dict:
        """Get general recommendations when no emotion context."""
        return {
            'activities': [
                'Take a mindful break',
                'Practice gratitude',
                'Connect with someone',
                'Learn something new'
            ][:limit],
            'message': 'General wellness suggestions',
            'timestamp': datetime.now().isoformat()
        }


class InsightGenerator:
    """
    Generate insights from user emotion patterns.
    """

    def __init__(self):
        """Initialize insight generator."""
        logger.info("Insight generator initialized")

    def generate_insights(
        self,
        mood_patterns: Dict,
        interaction_stats: Dict
    ) -> List[Dict]:
        """
        Generate insights from user data.

        Args:
            mood_patterns: User mood patterns
            interaction_stats: User interaction statistics

        Returns:
            List of insights
        """
        insights = []

        # Time-of-day pattern insights
        time_patterns = mood_patterns.get('time_patterns', {})
        if time_patterns:
            for period, emotion in time_patterns.items():
                if emotion in ['sad', 'stressed', 'angry']:
                    insights.append({
                        'type': 'pattern',
                        'severity': 'medium',
                        'title': f'{period.title()} Mood Pattern',
                        'description': f'You tend to feel {emotion} in the {period}',
                        'suggestion': f'Consider scheduling breaks or self-care during {period} time'
                    })

        # Dominant emotion insights
        dominant_emotions = mood_patterns.get('dominant_emotions', {})
        if dominant_emotions:
            top_emotion = list(dominant_emotions.keys())[0]

            if top_emotion in ['sad', 'stressed', 'angry']:
                insights.append({
                    'type': 'trend',
                    'severity': 'high',
                    'title': 'Recurring Emotion Detected',
                    'description': f'You\'ve frequently felt {top_emotion} recently',
                    'suggestion': 'Consider talking to someone or trying stress-relief techniques'
                })

        # Interaction insights
        total_interactions = interaction_stats.get('total_interactions', 0)
        if total_interactions > 10:
            insights.append({
                'type': 'milestone',
                'severity': 'low',
                'title': 'Active Engagement',
                'description': f'You\'ve had {total_interactions} meaningful conversations',
                'suggestion': 'Great job staying connected with yourself!'
            })

        return insights
