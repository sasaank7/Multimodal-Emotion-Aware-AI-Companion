"""
Streamlit UI for Emotion-Aware AI Companion.
Main interface with real-time emotion detection and adaptive conversation.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Optional, Dict

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from emotion_detection.facial_emotion import FacialEmotionDetector
from emotion_detection.text_sentiment import TextSentimentAnalyzer
from llm_engine.adaptive_response import AdaptiveResponseSystem
from utils.personalization import UserProfile
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


# Page configuration
st.set_page_config(
    page_title="Emotion-Aware AI Companion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .emotion-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }

    .mode-indicator {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }

    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }

    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }

    .assistant-message {
        background-color: #f3e5f5;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.config = get_config()
        st.session_state.user_profile = UserProfile()
        st.session_state.adaptive_system = AdaptiveResponseSystem()
        st.session_state.facial_detector = FacialEmotionDetector()
        st.session_state.text_analyzer = TextSentimentAnalyzer()
        st.session_state.chat_history = []
        st.session_state.emotion_history = []
        st.session_state.current_emotion = None
        st.session_state.webcam_enabled = False
        st.session_state.conversation_started = False


def create_emotion_gauge(emotion_data: Optional[Dict]) -> go.Figure:
    """
    Create emotion confidence gauge chart.

    Args:
        emotion_data: Emotion detection result

    Returns:
        Plotly figure
    """
    if emotion_data is None:
        confidence = 0
        emotion = "None"
    else:
        confidence = emotion_data.get('confidence', 0) * 100
        emotion = emotion_data.get('dominant_emotion', 'unknown').title()

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Current Emotion: {emotion}"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_emotion_distribution_chart(emotion_data: Optional[Dict]) -> go.Figure:
    """
    Create emotion distribution bar chart.

    Args:
        emotion_data: Emotion detection result

    Returns:
        Plotly figure
    """
    if emotion_data is None or 'emotions' not in emotion_data:
        emotions = {'neutral': 1.0}
    else:
        emotions = emotion_data['emotions']

    # Sort by value
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)

    labels = [e[0].title() for e in sorted_emotions]
    values = [e[1] * 100 for e in sorted_emotions]

    # Get colors from config
    config = get_config()
    emotion_colors = config.get('dashboard.emotion_colors', {})
    colors = [emotion_colors.get(e[0], '#808080') for e in sorted_emotions]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="Emotion Distribution",
        xaxis_title="Emotion",
        yaxis_title="Confidence (%)",
        height=300,
        showlegend=False
    )

    return fig


def create_mood_timeline(emotion_history: list) -> go.Figure:
    """
    Create mood timeline chart.

    Args:
        emotion_history: List of emotion data over time

    Returns:
        Plotly figure
    """
    if not emotion_history:
        fig = go.Figure()
        fig.update_layout(title="Mood Timeline (No data yet)")
        return fig

    # Extract data
    timestamps = [e.get('timestamp', datetime.now()) for e in emotion_history]
    confidences = [e.get('confidence', 0) * 100 for e in emotion_history]
    emotions = [e.get('dominant_emotion', 'unknown') for e in emotion_history]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=confidences,
        mode='lines+markers',
        name='Emotion Confidence',
        text=emotions,
        hovertemplate='<b>%{text}</b><br>Confidence: %{y:.1f}%<br>%{x}<extra></extra>',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Mood Timeline",
        xaxis_title="Time",
        yaxis_title="Confidence (%)",
        height=300,
        hovermode='closest'
    )

    return fig


def render_sidebar():
    """Render the sidebar with controls and settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Webcam toggle
        webcam_enabled = st.checkbox(
            "Enable Webcam",
            value=st.session_state.webcam_enabled,
            help="Enable facial emotion detection"
        )
        st.session_state.webcam_enabled = webcam_enabled

        # Modality toggles
        st.markdown("### Emotion Detection")
        face_enabled = st.checkbox("Facial Detection", value=True)
        text_enabled = st.checkbox("Text Sentiment", value=True)

        st.markdown("---")

        # User stats
        st.markdown("### 📊 Your Stats")
        context = st.session_state.user_profile.get_personalization_context()

        st.metric(
            "Total Interactions",
            context.get('total_interactions', 0)
        )

        mood_patterns = context.get('mood_patterns', {})
        dominant_emotions = mood_patterns.get('dominant_emotions', {})

        if dominant_emotions:
            top_emotion = list(dominant_emotions.keys())[0]
            st.metric(
                "Most Common Emotion (7d)",
                top_emotion.title()
            )

        st.markdown("---")

        # Export data
        if st.button("📥 Export My Data"):
            filepath = st.session_state.user_profile.export_data()
            st.success(f"Data exported to: {filepath}")

        # Clear history
        if st.button("🗑️ Clear History"):
            if st.confirm("Are you sure?"):
                st.session_state.user_profile.clear_history(keep_profile=True)
                st.session_state.chat_history = []
                st.session_state.emotion_history = []
                st.success("History cleared!")


def render_main_interface():
    """Render the main chat interface."""
    # Header
    st.markdown(
        '<div class="main-header">🤖 Emotion-Aware AI Companion</div>',
        unsafe_allow_html=True
    )

    # Create main columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💬 Conversation")

        # Chat container
        chat_container = st.container()

        with chat_container:
            # Display chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg['role']):
                    st.write(msg['content'])

                    if msg['role'] == 'assistant' and 'metadata' in msg:
                        metadata = msg['metadata']
                        mode = metadata.get('response_mode', 'neutral')

                        st.caption(f"🎭 Mode: {mode.title()}")

        # Chat input
        user_input = st.chat_input("Type your message here...")

        if user_input:
            # Add user message to chat
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })

            # Analyze text sentiment
            text_emotion = st.session_state.text_analyzer.analyze_sentiment(user_input)

            # Get facial emotion if webcam enabled
            face_emotion = st.session_state.current_emotion

            # Generate adaptive response
            with st.spinner("Thinking..."):
                result = st.session_state.adaptive_system.generate_response(
                    user_message=user_input,
                    face_emotion=face_emotion,
                    voice_emotion=None,  # Voice not implemented in this UI version
                    text_emotion=text_emotion
                )

            # Add assistant response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': result['response'],
                'timestamp': datetime.now(),
                'metadata': {
                    'emotion_context': result['emotion_context'],
                    'response_mode': result['response_mode'],
                    'confidence': result['confidence']
                }
            })

            # Track in user profile
            st.session_state.user_profile.add_interaction(
                user_message=user_input,
                assistant_response=result['response'],
                emotion_context=result['emotion_context'],
                response_mode=result['response_mode']
            )

            # Update emotion history
            if result['emotion_context']:
                st.session_state.emotion_history.append(result['emotion_context'])
                st.session_state.user_profile.add_mood_entry(result['emotion_context'])

            # Rerun to update UI
            st.rerun()

    with col2:
        st.markdown("### 📈 Emotion Analytics")

        # Current emotion display
        current_emotion = st.session_state.current_emotion

        # Emotion gauge
        gauge_fig = create_emotion_gauge(current_emotion)
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Emotion distribution
        if current_emotion:
            dist_fig = create_emotion_distribution_chart(current_emotion)
            st.plotly_chart(dist_fig, use_container_width=True)

        # Response mode indicator
        if st.session_state.chat_history:
            last_msg = st.session_state.chat_history[-1]
            if last_msg['role'] == 'assistant' and 'metadata' in last_msg:
                mode = last_msg['metadata']['response_mode']

                st.markdown(f"""
                <div class="mode-indicator">
                    <b>🎭 Current Mode:</b> {mode.title()}<br>
                    <small>The companion adapts its tone based on your emotions</small>
                </div>
                """, unsafe_allow_html=True)


def render_dashboard():
    """Render the analytics dashboard."""
    st.markdown("## 📊 Mood & Interaction Dashboard")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Mood Timeline", "Patterns", "Statistics"])

    with tab1:
        st.markdown("### Mood Over Time")

        # Mood timeline
        timeline_fig = create_mood_timeline(st.session_state.emotion_history)
        st.plotly_chart(timeline_fig, use_container_width=True)

        # Recent emotions
        if st.session_state.emotion_history:
            st.markdown("### Recent Emotions")

            recent = st.session_state.emotion_history[-10:]
            for i, emotion in enumerate(reversed(recent)):
                timestamp = emotion.get('timestamp', datetime.now())
                dominant = emotion.get('dominant_emotion', 'unknown')
                confidence = emotion.get('confidence', 0) * 100

                st.write(f"**{timestamp.strftime('%H:%M:%S')}** - {dominant.title()} ({confidence:.0f}%)")

    with tab2:
        st.markdown("### Emotion Patterns (Last 7 Days)")

        mood_patterns = st.session_state.user_profile.get_mood_patterns(days=7)

        # Dominant emotions
        dominant_emotions = mood_patterns.get('dominant_emotions', {})
        if dominant_emotions:
            fig = px.pie(
                values=list(dominant_emotions.values()),
                names=[e.title() for e in dominant_emotions.keys()],
                title="Emotion Distribution (7 days)"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Time of day patterns
        time_patterns = mood_patterns.get('time_patterns', {})
        if time_patterns:
            st.markdown("### Emotion by Time of Day")

            for period, emotion in time_patterns.items():
                st.write(f"**{period.title()}**: {emotion.title()}")

    with tab3:
        st.markdown("### Interaction Statistics")

        stats = st.session_state.user_profile.get_interaction_stats(days=7)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Interactions (7d)",
                stats.get('total_interactions', 0)
            )

        with col2:
            st.metric(
                "Avg Message Length",
                f"{stats.get('avg_message_length', 0):.0f} chars"
            )

        # Response modes used
        response_modes = stats.get('response_modes', {})
        if response_modes:
            st.markdown("### Response Modes Used")

            fig = px.bar(
                x=list(response_modes.keys()),
                y=list(response_modes.values()),
                labels={'x': 'Mode', 'y': 'Count'},
                title="Companion Response Modes"
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main application entry point."""
    # Initialize
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Main content tabs
    main_tab, dashboard_tab = st.tabs(["💬 Chat", "📊 Dashboard"])

    with main_tab:
        render_main_interface()

    with dashboard_tab:
        render_dashboard()

    # Webcam processing (if enabled)
    if st.session_state.webcam_enabled:
        # Note: Real-time webcam in Streamlit requires additional setup
        # This is a placeholder for the webcam functionality
        st.info("Webcam emotion detection is available. For real-time processing, use the standalone version.")


if __name__ == "__main__":
    main()
