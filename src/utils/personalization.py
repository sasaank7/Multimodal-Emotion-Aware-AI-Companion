"""
User Personalization and History Tracking.
Manages user profiles, preferences, and mood history for personalized interactions.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from tinydb import TinyDB, Query

from .logger import get_logger
from .config_loader import get_config

logger = get_logger(__name__)


class UserProfile:
    """
    Manages user profile and personalization data.
    """

    def __init__(self, user_id: str = "default"):
        """
        Initialize user profile.

        Args:
            user_id: Unique user identifier
        """
        self.config = get_config()
        self.personalization_config = self.config.get_section("personalization")

        self.user_id = user_id
        self.enabled = self.personalization_config.get("enabled", True)

        # Storage setup
        storage_path = self.personalization_config.get(
            "storage_path",
            "data/user_data/user_profile.json"
        )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db = TinyDB(str(self.storage_path))

        # Tables
        self.profile_table = self.db.table('profile')
        self.mood_history_table = self.db.table('mood_history')
        self.interactions_table = self.db.table('interactions')
        self.preferences_table = self.db.table('preferences')

        # Load or create profile
        self.profile_data = self._load_profile()

        logger.info(f"User profile initialized for user: {user_id}")

    def _load_profile(self) -> Dict:
        """
        Load user profile or create new one.

        Returns:
            Profile data dictionary
        """
        User = Query()
        result = self.profile_table.search(User.user_id == self.user_id)

        if result:
            return result[0]

        # Create new profile
        profile = {
            'user_id': self.user_id,
            'created_at': datetime.now().isoformat(),
            'last_active': datetime.now().isoformat(),
            'total_interactions': 0
        }

        self.profile_table.insert(profile)
        return profile

    def _save_profile(self):
        """Save profile data to storage."""
        User = Query()
        self.profile_table.update(
            self.profile_data,
            User.user_id == self.user_id
        )

    def add_mood_entry(self, emotion_data: Dict):
        """
        Add a mood entry to history.

        Args:
            emotion_data: Emotion detection result
        """
        if not self.enabled:
            return

        entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': self.user_id,
            'dominant_emotion': emotion_data.get('dominant_emotion'),
            'confidence': emotion_data.get('confidence'),
            'emotions': emotion_data.get('emotions', {}),
            'modalities': emotion_data.get('modalities_used', [])
        }

        self.mood_history_table.insert(entry)

        # Clean old entries
        self._cleanup_old_entries(self.mood_history_table)

    def add_interaction(
        self,
        user_message: str,
        assistant_response: str,
        emotion_context: Optional[Dict] = None,
        response_mode: Optional[str] = None
    ):
        """
        Record an interaction.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            emotion_context: Emotion context
            response_mode: Response mode used
        """
        if not self.enabled:
            return

        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_id': self.user_id,
            'user_message': user_message,
            'assistant_response': assistant_response,
            'emotion_context': emotion_context,
            'response_mode': response_mode
        }

        self.interactions_table.insert(interaction)

        # Update profile
        self.profile_data['total_interactions'] += 1
        self.profile_data['last_active'] = datetime.now().isoformat()
        self._save_profile()

        # Clean old entries
        self._cleanup_old_entries(self.interactions_table)

    def set_preference(self, key: str, value: Any):
        """
        Set a user preference.

        Args:
            key: Preference key
            value: Preference value
        """
        Pref = Query()
        existing = self.preferences_table.search(
            (Pref.user_id == self.user_id) & (Pref.key == key)
        )

        pref_data = {
            'user_id': self.user_id,
            'key': key,
            'value': value,
            'updated_at': datetime.now().isoformat()
        }

        if existing:
            self.preferences_table.update(
                pref_data,
                (Pref.user_id == self.user_id) & (Pref.key == key)
            )
        else:
            self.preferences_table.insert(pref_data)

        logger.debug(f"Preference set: {key} = {value}")

    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference.

        Args:
            key: Preference key
            default: Default value if not found

        Returns:
            Preference value or default
        """
        Pref = Query()
        result = self.preferences_table.search(
            (Pref.user_id == self.user_id) & (Pref.key == key)
        )

        if result:
            return result[0]['value']

        return default

    def get_mood_history(
        self,
        days: int = 7,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get mood history for a time period.

        Args:
            days: Number of days to retrieve
            limit: Maximum number of entries

        Returns:
            List of mood entries
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()

        Entry = Query()
        results = self.mood_history_table.search(
            (Entry.user_id == self.user_id) & (Entry.timestamp >= cutoff_str)
        )

        # Sort by timestamp
        results.sort(key=lambda x: x['timestamp'], reverse=True)

        if limit:
            results = results[:limit]

        return results

    def get_mood_patterns(self, days: int = 7) -> Dict:
        """
        Analyze mood patterns over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary of mood pattern statistics
        """
        history = self.get_mood_history(days=days)

        if not history:
            return {
                'dominant_emotions': {},
                'avg_confidence': 0.0,
                'emotion_transitions': [],
                'time_patterns': {}
            }

        # Count emotions
        emotion_counts = Counter([
            entry['dominant_emotion']
            for entry in history
            if entry.get('dominant_emotion')
        ])

        # Average confidence
        confidences = [
            entry['confidence']
            for entry in history
            if entry.get('confidence') is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Time-of-day patterns
        time_emotions = defaultdict(list)
        for entry in history:
            try:
                timestamp = datetime.fromisoformat(entry['timestamp'])
                hour = timestamp.hour

                # Categorize by time of day
                if 5 <= hour < 12:
                    period = 'morning'
                elif 12 <= hour < 17:
                    period = 'afternoon'
                elif 17 <= hour < 22:
                    period = 'evening'
                else:
                    period = 'night'

                emotion = entry.get('dominant_emotion')
                if emotion:
                    time_emotions[period].append(emotion)
            except:
                continue

        # Most common emotion by time of day
        time_patterns = {
            period: Counter(emotions).most_common(1)[0][0]
            if emotions else 'neutral'
            for period, emotions in time_emotions.items()
        }

        return {
            'dominant_emotions': dict(emotion_counts.most_common(5)),
            'avg_confidence': avg_confidence,
            'total_entries': len(history),
            'time_patterns': time_patterns,
            'days_analyzed': days
        }

    def get_interaction_stats(self, days: int = 7) -> Dict:
        """
        Get interaction statistics.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary of statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()

        Entry = Query()
        interactions = self.interactions_table.search(
            (Entry.user_id == self.user_id) & (Entry.timestamp >= cutoff_str)
        )

        # Count response modes used
        mode_counts = Counter([
            i['response_mode']
            for i in interactions
            if i.get('response_mode')
        ])

        # Average message length
        message_lengths = [
            len(i['user_message'])
            for i in interactions
            if i.get('user_message')
        ]
        avg_message_length = (
            sum(message_lengths) / len(message_lengths)
            if message_lengths else 0
        )

        return {
            'total_interactions': len(interactions),
            'response_modes': dict(mode_counts),
            'avg_message_length': avg_message_length,
            'days_analyzed': days
        }

    def get_personalization_context(self) -> Dict:
        """
        Get context for personalized responses.

        Returns:
            Dictionary of personalization context
        """
        # Get recent mood patterns
        mood_patterns = self.get_mood_patterns(days=7)

        # Get interaction stats
        interaction_stats = self.get_interaction_stats(days=7)

        # Get preferences
        all_prefs = self.preferences_table.search(
            Query().user_id == self.user_id
        )
        preferences = {
            pref['key']: pref['value']
            for pref in all_prefs
        }

        return {
            'user_id': self.user_id,
            'total_interactions': self.profile_data.get('total_interactions', 0),
            'mood_patterns': mood_patterns,
            'interaction_stats': interaction_stats,
            'preferences': preferences
        }

    def _cleanup_old_entries(self, table):
        """
        Remove entries older than retention period.

        Args:
            table: Database table to clean
        """
        max_days = self.personalization_config.get('max_history_days', 30)
        cutoff_date = datetime.now() - timedelta(days=max_days)
        cutoff_str = cutoff_date.isoformat()

        Entry = Query()
        table.remove(
            (Entry.user_id == self.user_id) & (Entry.timestamp < cutoff_str)
        )

        # Also enforce max interaction limit
        max_interactions = self.personalization_config.get('max_interactions', 1000)
        all_entries = table.search(Entry.user_id == self.user_id)

        if len(all_entries) > max_interactions:
            # Sort by timestamp and keep only recent ones
            all_entries.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            entries_to_remove = all_entries[max_interactions:]

            for entry in entries_to_remove:
                table.remove(doc_ids=[entry.doc_id])

    def export_data(self, filepath: Optional[str] = None) -> str:
        """
        Export user data to JSON file.

        Args:
            filepath: Export file path (auto-generated if None)

        Returns:
            Path to exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"data/user_data/export_{self.user_id}_{timestamp}.json"

        export_data = {
            'profile': self.profile_data,
            'mood_history': self.get_mood_history(days=30),
            'mood_patterns': self.get_mood_patterns(days=30),
            'interaction_stats': self.get_interaction_stats(days=30),
            'preferences': {
                pref['key']: pref['value']
                for pref in self.preferences_table.search(
                    Query().user_id == self.user_id
                )
            },
            'exported_at': datetime.now().isoformat()
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"User data exported to: {filepath}")
        return filepath

    def clear_history(self, keep_profile: bool = True):
        """
        Clear user history.

        Args:
            keep_profile: Whether to keep profile data
        """
        Entry = Query()

        self.mood_history_table.remove(Entry.user_id == self.user_id)
        self.interactions_table.remove(Entry.user_id == self.user_id)

        if not keep_profile:
            self.preferences_table.remove(Entry.user_id == self.user_id)
            self.profile_table.remove(Entry.user_id == self.user_id)

        logger.info("User history cleared")

    def close(self):
        """Close the database connection."""
        self.db.close()
